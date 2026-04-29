from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as onp
import pandas as pd

from aerosandbox.aerodynamics.aero_2D.afc_dataset import (
    KULFAN_LOWER_COLUMNS,
    KULFAN_UPPER_COLUMNS,
    afc_dataset_statistics,
    check_duplicate_afc_samples,
    read_afc_dataset,
    read_afc_dataset_manifest,
    split_afc_dataset,
    validate_afc_dataframe,
)
from aerosandbox.aerodynamics.aero_2D.afc_confidence import (
    make_afc_training_statistics,
    write_afc_training_statistics,
)
from aerosandbox.aerodynamics.aero_2D.afc_neuralfoil import (
    AFC_N_INPUTS,
    AFC_N_OUTPUTS,
)

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except ImportError:  # pragma: no cover - exercised only in environments without PyTorch
    torch = None
    nn = None
    F = None
    DataLoader = None
    Dataset = object


BASELINE_COLUMNS = [
    "CL_baseline",
    "CD_baseline",
    "CM_baseline",
    "Top_Xtr_baseline",
    "Bot_Xtr_baseline",
    "analysis_confidence_baseline",
]


MODEL_SIZE_DEFAULTS = {
    "small": {"hidden_width": 32, "hidden_depth": 2},
    "medium": {"hidden_width": 64, "hidden_depth": 3},
    "large": {"hidden_width": 128, "hidden_depth": 4},
}


def _require_torch() -> None:
    if torch is None:
        raise ImportError(
            "AFC-NeuralFoil training requires PyTorch. Install `torch` in the "
            "training environment; AeroSandbox inference itself does not require it."
        )


@dataclass
class TrainingConfig:
    data_path: str = "examples/generated_afc_dataset/afc_airfoil_synthetic.csv"
    dataset_manifest_path: Optional[str] = None
    output_directory: str = "runs/afc_neuralfoil"
    model_size: str = "small"
    hidden_width: Optional[int] = None
    hidden_depth: Optional[int] = None
    baseline_model_size: str = "large"
    overwrite_baseline_columns: bool = False
    batch_size: int = 256
    epochs: int = 200
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    train_fraction: float = 0.80
    val_fraction: float = 0.10
    test_fraction: float = 0.10
    random_seed: int = 0
    num_workers: int = 0
    device: str = "auto"
    early_stopping_patience: int = 30
    early_stopping_min_delta: float = 1e-5
    lr_plateau_patience: int = 10
    lr_plateau_factor: float = 0.5
    min_gate_for_output_norm: float = 0.10
    loss_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "CL": 1.0,
            "logCD": 1.0,
            "CM": 0.5,
            "Top_Xtr": 0.05,
            "Bot_Xtr": 0.05,
            "confidence": 0.1,
        }
    )
    export_best_npz: bool = True
    save_prediction_plots: bool = True
    n_prediction_plot_points: int = 5000

    def resolved_hidden_width(self) -> int:
        if self.hidden_width is not None:
            return self.hidden_width
        return MODEL_SIZE_DEFAULTS.get(self.model_size, MODEL_SIZE_DEFAULTS["small"])[
            "hidden_width"
        ]

    def resolved_hidden_depth(self) -> int:
        if self.hidden_depth is not None:
            return self.hidden_depth
        return MODEL_SIZE_DEFAULTS.get(self.model_size, MODEL_SIZE_DEFAULTS["small"])[
            "hidden_depth"
        ]


def load_config(filepath: Optional[str]) -> TrainingConfig:
    config = TrainingConfig()
    if filepath is None:
        return config

    with open(filepath, "r", encoding="utf-8") as f:
        user_config = json.load(f)

    merged = asdict(config)
    merged.update(user_config)
    return TrainingConfig(**merged)


def write_config(config: TrainingConfig, filepath: Path) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)


def _safe_std(values: onp.ndarray) -> onp.ndarray:
    std = onp.nanstd(values, axis=0)
    return onp.where(std < 1e-8, 1.0, std)


def _logit(probability: onp.ndarray) -> onp.ndarray:
    p = onp.clip(probability, 1e-5, 1 - 1e-5)
    return onp.log(p / (1 - p))


def _scalar(value) -> float:
    return float(onp.ravel(onp.asarray(value, dtype=float))[0])


def build_feature_array(df: pd.DataFrame) -> onp.ndarray:
    """
    Builds the same 30-column AFC-NeuralFoil feature matrix used by the NumPy inference path.
    """
    features = [
        *[df[column].to_numpy(float) for column in KULFAN_UPPER_COLUMNS],
        *[df[column].to_numpy(float) for column in KULFAN_LOWER_COLUMNS],
        df["leading_edge_weight"].to_numpy(float),
        50 * df["TE_thickness"].to_numpy(float),
        onp.sin(onp.deg2rad(df["alpha"].to_numpy(float))),
        onp.cos(onp.deg2rad(df["alpha"].to_numpy(float))),
        onp.sin(onp.deg2rad(2 * df["alpha"].to_numpy(float))),
        (onp.log(df["Re"].to_numpy(float)) - 12.5) / 3.5,
        df["Mach"].to_numpy(float),
        (df["n_crit"].to_numpy(float) - 9) / 4.5,
        df["xtr_upper"].to_numpy(float),
        df["xtr_lower"].to_numpy(float),
        df["C_mu"].to_numpy(float),
        df["x_jet"].to_numpy(float),
        onp.sin(onp.deg2rad(df["theta_jet"].to_numpy(float))),
        onp.cos(onp.deg2rad(df["theta_jet"].to_numpy(float))),
    ]
    x = onp.stack(features, axis=1).astype(onp.float32)
    if x.shape[1] != AFC_N_INPUTS:
        raise RuntimeError(f"Expected {AFC_N_INPUTS} input features, got {x.shape[1]}.")
    return x


def ensure_neuralfoil_baseline_columns(
    df: pd.DataFrame,
    *,
    model_size: str = "large",
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Adds NeuralFoil baseline columns needed by the current correction-model inference convention.

    This is deliberately straightforward rather than highly optimized. For large datasets, precompute these columns
    once and store them in the parquet/csv used for training.
    """
    if all(column in df.columns for column in BASELINE_COLUMNS) and not overwrite:
        return df

    import aerosandbox as asb

    df = df.copy()
    values = {column: [] for column in BASELINE_COLUMNS}

    for _, row in df.iterrows():
        airfoil = asb.KulfanAirfoil(
            upper_weights=[row[column] for column in KULFAN_UPPER_COLUMNS],
            lower_weights=[row[column] for column in KULFAN_LOWER_COLUMNS],
            leading_edge_weight=row["leading_edge_weight"],
            TE_thickness=row["TE_thickness"],
        )
        aero = airfoil.get_aero_from_neuralfoil(
            alpha=row["alpha"],
            Re=row["Re"],
            mach=row["Mach"],
            n_crit=row["n_crit"],
            xtr_upper=row["xtr_upper"],
            xtr_lower=row["xtr_lower"],
            model_size=model_size,
        )
        values["CL_baseline"].append(_scalar(aero["CL"]))
        values["CD_baseline"].append(_scalar(aero["CD"]))
        values["CM_baseline"].append(_scalar(aero["CM"]))
        values["Top_Xtr_baseline"].append(_scalar(aero["Top_Xtr"]))
        values["Bot_Xtr_baseline"].append(_scalar(aero["Bot_Xtr"]))
        values["analysis_confidence_baseline"].append(
            _scalar(aero["analysis_confidence"])
        )

    for column, column_values in values.items():
        df[column] = column_values

    return df


def build_raw_output_targets(
    df: pd.DataFrame,
    *,
    min_gate: float = 0.10,
) -> onp.ndarray:
    """
    Estimates the raw MLP outputs implied by the training data.

    These targets are used for output normalization only; the main training loss is applied after decoding back to
    physical `CL`, `logCD`, and `CM`.
    """
    gate = 1 - onp.exp(-60 * df["C_mu"].to_numpy(float))
    gate_ok = gate > min_gate
    gate_safe = onp.where(gate_ok, gate, 1.0)

    y = onp.zeros((len(df), AFC_N_OUTPUTS), dtype=onp.float32)
    y[:, 0] = _logit(df["confidence_label"].to_numpy(float)).astype(onp.float32)
    y[:, 1] = onp.where(
        gate_ok,
        2 * (df["CL"].to_numpy(float) - df["CL_baseline"].to_numpy(float)) / gate_safe,
        0.0,
    )
    y[:, 2] = onp.where(
        gate_ok,
        5
        * (
            onp.log(onp.clip(df["CD"].to_numpy(float), 1e-12, None))
            - onp.log(onp.clip(df["CD_baseline"].to_numpy(float), 1e-12, None))
        )
        / gate_safe,
        0.0,
    )
    y[:, 3] = onp.where(
        gate_ok,
        20 * (df["CM"].to_numpy(float) - df["CM_baseline"].to_numpy(float)) / gate_safe,
        0.0,
    )
    y[:, 4] = onp.where(
        gate_ok,
        5
        * (df["Top_Xtr"].to_numpy(float) - df["Top_Xtr_baseline"].to_numpy(float))
        / gate_safe,
        0.0,
    )
    y[:, 5] = onp.where(
        gate_ok,
        5
        * (df["Bot_Xtr"].to_numpy(float) - df["Bot_Xtr_baseline"].to_numpy(float))
        / gate_safe,
        0.0,
    )
    return y


def make_normalization_stats(
    train_df: pd.DataFrame,
    *,
    min_gate_for_output_norm: float,
) -> Dict[str, onp.ndarray]:
    x = build_feature_array(train_df)
    y = build_raw_output_targets(train_df, min_gate=min_gate_for_output_norm)
    return {
        "input_mean": onp.mean(x, axis=0).astype(onp.float32),
        "input_std": _safe_std(x).astype(onp.float32),
        "output_mean": onp.nanmean(y, axis=0).astype(onp.float32),
        "output_std": _safe_std(y).astype(onp.float32),
    }


class AFCNeuralFoilTorchDataset(Dataset):
    """
    Thin PyTorch Dataset wrapper around the canonical flat AFC dataframe schema.
    """

    def __init__(self, df: pd.DataFrame):
        _require_torch()
        self.df = df.reset_index(drop=True)
        self.x = torch.as_tensor(build_feature_array(self.df), dtype=torch.float32)

        self.targets = {
            "CL": torch.as_tensor(self.df["CL"].to_numpy(float), dtype=torch.float32),
            "logCD": torch.as_tensor(
                onp.log(onp.clip(self.df["CD"].to_numpy(float), 1e-12, None)),
                dtype=torch.float32,
            ),
            "CM": torch.as_tensor(self.df["CM"].to_numpy(float), dtype=torch.float32),
            "Top_Xtr": torch.as_tensor(
                self.df["Top_Xtr"].to_numpy(float), dtype=torch.float32
            ),
            "Bot_Xtr": torch.as_tensor(
                self.df["Bot_Xtr"].to_numpy(float), dtype=torch.float32
            ),
            "confidence": torch.as_tensor(
                self.df["confidence_label"].to_numpy(float), dtype=torch.float32
            ),
        }
        self.baseline = {
            "CL": torch.as_tensor(
                self.df["CL_baseline"].to_numpy(float), dtype=torch.float32
            ),
            "logCD": torch.as_tensor(
                onp.log(onp.clip(self.df["CD_baseline"].to_numpy(float), 1e-12, None)),
                dtype=torch.float32,
            ),
            "CM": torch.as_tensor(
                self.df["CM_baseline"].to_numpy(float), dtype=torch.float32
            ),
            "Top_Xtr": torch.as_tensor(
                self.df["Top_Xtr_baseline"].to_numpy(float), dtype=torch.float32
            ),
            "Bot_Xtr": torch.as_tensor(
                self.df["Bot_Xtr_baseline"].to_numpy(float), dtype=torch.float32
            ),
            "analysis_confidence": torch.as_tensor(
                self.df["analysis_confidence_baseline"].to_numpy(float),
                dtype=torch.float32,
            ),
        }
        self.gate = torch.as_tensor(
            1 - onp.exp(-60 * self.df["C_mu"].to_numpy(float)),
            dtype=torch.float32,
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Dict[str, object]:
        return {
            "x": self.x[index],
            "targets": {key: value[index] for key, value in self.targets.items()},
            "baseline": {key: value[index] for key, value in self.baseline.items()},
            "gate": self.gate[index],
        }


if nn is not None:

    class Swish(nn.Module):
        def forward(self, x):
            return x * torch.sigmoid(x)


    class AFCNeuralFoilMLP(nn.Module):
        """
        MLP that keeps input/output normalization inside the PyTorch module.

        The exporter folds these normalizers into the first and last linear layers so the runtime `.npz` can be
        evaluated directly by `aerosandbox.numpy`.
        """

        def __init__(
            self,
            *,
            hidden_width: int,
            hidden_depth: int,
            input_mean: Iterable[float],
            input_std: Iterable[float],
            output_mean: Iterable[float],
            output_std: Iterable[float],
        ):
            super().__init__()
            self.register_buffer(
                "input_mean",
                torch.as_tensor(onp.asarray(input_mean), dtype=torch.float32),
            )
            self.register_buffer(
                "input_std",
                torch.as_tensor(onp.asarray(input_std), dtype=torch.float32),
            )
            self.register_buffer(
                "output_mean",
                torch.as_tensor(onp.asarray(output_mean), dtype=torch.float32),
            )
            self.register_buffer(
                "output_std",
                torch.as_tensor(onp.asarray(output_std), dtype=torch.float32),
            )

            layer_sizes = [
                AFC_N_INPUTS,
                *([hidden_width] * hidden_depth),
                AFC_N_OUTPUTS,
            ]
            self.linears = nn.ModuleList(
                [
                    nn.Linear(n_in, n_out)
                    for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:])
                ]
            )
            self.activation = Swish()

        def forward(self, x):
            z = (x - self.input_mean) / self.input_std
            for i, layer in enumerate(self.linears):
                z = layer(z)
                if i != len(self.linears) - 1:
                    z = self.activation(z)
            return z * self.output_std + self.output_mean

else:

    class Swish:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            _require_torch()


    class AFCNeuralFoilMLP:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            _require_torch()


def decode_raw_outputs(raw, batch) -> Dict[str, object]:
    gate = batch["gate"]
    baseline = batch["baseline"]
    model_confidence = torch.sigmoid(raw[:, 0])

    return {
        "analysis_confidence": baseline["analysis_confidence"]
        * (1 - gate * (1 - model_confidence)),
        "CL": baseline["CL"] + gate * raw[:, 1] / 2,
        "logCD": baseline["logCD"] + gate * raw[:, 2] / 5,
        "CM": baseline["CM"] + gate * raw[:, 3] / 20,
        "Top_Xtr": torch.clamp(baseline["Top_Xtr"] + gate * raw[:, 4] / 5, 0, 1),
        "Bot_Xtr": torch.clamp(baseline["Bot_Xtr"] + gate * raw[:, 5] / 5, 0, 1),
        "confidence_logit": raw[:, 0],
    }


def move_batch_to_device(batch: Dict[str, object], device: str) -> Dict[str, object]:
    return {
        "x": batch["x"].to(device),
        "targets": {
            key: value.to(device) for key, value in batch["targets"].items()
        },
        "baseline": {
            key: value.to(device) for key, value in batch["baseline"].items()
        },
        "gate": batch["gate"].to(device),
    }


def compute_loss(predictions: Dict[str, object], batch: Dict[str, object], config: TrainingConfig):
    targets = batch["targets"]
    weights = config.loss_weights

    loss_terms = {
        "CL": F.huber_loss(predictions["CL"], targets["CL"]),
        "logCD": F.huber_loss(predictions["logCD"], targets["logCD"]),
        "CM": F.huber_loss(predictions["CM"], targets["CM"]),
        "Top_Xtr": F.huber_loss(predictions["Top_Xtr"], targets["Top_Xtr"]),
        "Bot_Xtr": F.huber_loss(predictions["Bot_Xtr"], targets["Bot_Xtr"]),
        "confidence": F.binary_cross_entropy_with_logits(
            predictions["confidence_logit"], targets["confidence"]
        ),
    }
    total = sum(weights.get(key, 0.0) * value for key, value in loss_terms.items())
    return total, loss_terms


def run_epoch(
    model,
    loader,
    *,
    config: TrainingConfig,
    device: str,
    optimizer=None,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    totals: Dict[str, float] = {}
    n_batches = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            raw = model(batch["x"])
            predictions = decode_raw_outputs(raw, batch)
            loss, loss_terms = compute_loss(predictions, batch, config)
            if is_train:
                loss.backward()
                optimizer.step()

        totals["loss"] = totals.get("loss", 0.0) + float(loss.detach().cpu())
        for key, value in loss_terms.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
        n_batches += 1

    return {key: value / max(n_batches, 1) for key, value in totals.items()}


def _tensor_to_numpy(value) -> onp.ndarray:
    return value.detach().cpu().numpy()


def predict_dataframe(
    model,
    df: pd.DataFrame,
    *,
    config: TrainingConfig,
    device: str,
) -> pd.DataFrame:
    """
    Runs the trained PyTorch model over a DataFrame and returns prediction-vs-truth rows.
    """
    _require_torch()
    if len(df) == 0:
        return pd.DataFrame()

    dataset = AFCNeuralFoilTorchDataset(df)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    prediction_parts: Dict[str, List[onp.ndarray]] = {
        "analysis_confidence": [],
        "CL": [],
        "logCD": [],
        "CM": [],
        "Top_Xtr": [],
        "Bot_Xtr": [],
    }

    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            raw = model(batch["x"])
            predictions = decode_raw_outputs(raw, batch)
            for key in prediction_parts:
                prediction_parts[key].append(_tensor_to_numpy(predictions[key]))

    pred = {
        key: onp.concatenate(values) if values else onp.array([])
        for key, values in prediction_parts.items()
    }
    pred["CD"] = onp.exp(pred["logCD"])

    out = pd.DataFrame(
        {
            "alpha": df["alpha"].to_numpy(float),
            "Re": df["Re"].to_numpy(float),
            "Mach": df["Mach"].to_numpy(float),
            "C_mu": df["C_mu"].to_numpy(float),
            "x_jet": df["x_jet"].to_numpy(float),
            "theta_jet": df["theta_jet"].to_numpy(float),
            "source": df["source"].astype(str).to_numpy(),
            "truth_CL": df["CL"].to_numpy(float),
            "predicted_CL": pred["CL"],
            "truth_CD": df["CD"].to_numpy(float),
            "predicted_CD": pred["CD"],
            "truth_logCD": onp.log(onp.clip(df["CD"].to_numpy(float), 1e-12, None)),
            "predicted_logCD": pred["logCD"],
            "truth_CM": df["CM"].to_numpy(float),
            "predicted_CM": pred["CM"],
            "truth_Top_Xtr": df["Top_Xtr"].to_numpy(float),
            "predicted_Top_Xtr": pred["Top_Xtr"],
            "truth_Bot_Xtr": df["Bot_Xtr"].to_numpy(float),
            "predicted_Bot_Xtr": pred["Bot_Xtr"],
            "truth_confidence_label": df["confidence_label"].to_numpy(float),
            "predicted_analysis_confidence": pred["analysis_confidence"],
        }
    )
    for field in ["CL", "CD", "logCD", "CM", "Top_Xtr", "Bot_Xtr"]:
        out[f"error_{field}"] = out[f"predicted_{field}"] - out[f"truth_{field}"]
    out["error_confidence"] = (
        out["predicted_analysis_confidence"] - out["truth_confidence_label"]
    )
    return out


def make_test_error_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Computes compact test-set error metrics from `predict_dataframe()`.
    """
    rows = []
    field_pairs = [
        ("CL", "truth_CL", "predicted_CL"),
        ("CD", "truth_CD", "predicted_CD"),
        ("logCD", "truth_logCD", "predicted_logCD"),
        ("CM", "truth_CM", "predicted_CM"),
        ("Top_Xtr", "truth_Top_Xtr", "predicted_Top_Xtr"),
        ("Bot_Xtr", "truth_Bot_Xtr", "predicted_Bot_Xtr"),
        (
            "analysis_confidence",
            "truth_confidence_label",
            "predicted_analysis_confidence",
        ),
    ]
    for field, truth_column, predicted_column in field_pairs:
        if truth_column not in predictions or predicted_column not in predictions:
            continue
        truth = predictions[truth_column].to_numpy(float)
        predicted = predictions[predicted_column].to_numpy(float)
        error = predicted - truth
        rows.append(
            {
                "field": field,
                "n": int(len(error)),
                "truth_mean": float(onp.mean(truth)) if len(error) else onp.nan,
                "predicted_mean": float(onp.mean(predicted)) if len(error) else onp.nan,
                "bias": float(onp.mean(error)) if len(error) else onp.nan,
                "mae": float(onp.mean(onp.abs(error))) if len(error) else onp.nan,
                "rmse": float(onp.sqrt(onp.mean(error**2))) if len(error) else onp.nan,
                "max_abs_error": float(onp.max(onp.abs(error))) if len(error) else onp.nan,
            }
        )
    return pd.DataFrame(rows)


def save_prediction_vs_truth_plots(
    predictions: pd.DataFrame,
    output_directory: Path,
    *,
    max_points: int = 5000,
) -> List[Path]:
    """
    Saves basic prediction-vs-truth scatter plots for acceptance review.
    """
    if len(predictions) == 0:
        return []

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    output_directory.mkdir(parents=True, exist_ok=True)
    if len(predictions) > max_points:
        plot_df = predictions.sample(n=max_points, random_state=0)
    else:
        plot_df = predictions

    fields = ["CL", "CD", "CM", "Top_Xtr", "Bot_Xtr"]
    paths = []
    for field in fields:
        truth = plot_df[f"truth_{field}"].to_numpy(float)
        predicted = plot_df[f"predicted_{field}"].to_numpy(float)
        lower = float(min(onp.min(truth), onp.min(predicted)))
        upper = float(max(onp.max(truth), onp.max(predicted)))
        pad = 0.04 * (upper - lower + 1e-12)

        fig, ax = plt.subplots(figsize=(4.8, 4.2))
        ax.scatter(truth, predicted, s=10, alpha=0.45, linewidths=0)
        ax.plot([lower - pad, upper + pad], [lower - pad, upper + pad], "k--", lw=1)
        ax.set_xlabel(f"Truth {field}")
        ax.set_ylabel(f"Predicted {field}")
        ax.set_title(f"AFC-NeuralFoil test: {field}")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()

        path = output_directory / f"prediction_vs_truth_{field}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)

    return paths


def _read_optional_dataset_manifest(config: TrainingConfig) -> Optional[Dict[str, object]]:
    candidate_paths = []
    if config.dataset_manifest_path is not None:
        candidate_paths.append(Path(config.dataset_manifest_path))

    data_path = Path(config.data_path)
    candidate_paths.append(data_path.with_suffix(data_path.suffix + ".manifest.json"))
    candidate_paths.append(data_path.parent / "manifest.json")

    for path in candidate_paths:
        if path.exists():
            return read_afc_dataset_manifest(path)
    return None


def _dataframe_to_markdown_table(df: pd.DataFrame) -> str:
    if len(df) == 0:
        return "No rows."
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        formatted = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                formatted.append(f"{value:.6g}")
            else:
                formatted.append(str(value))
        lines.append("| " + " | ".join(formatted) + " |")
    return "\n".join(lines)


def write_model_card(
    filepath: Path,
    *,
    config: TrainingConfig,
    dataset_manifest: Optional[Dict[str, object]],
    full_df: pd.DataFrame,
    split_dfs: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    test_metrics: Dict[str, float],
    error_summary: pd.DataFrame,
    predictions_path: Path,
    error_summary_path: Path,
    plot_paths: Iterable[Path],
    exported_npz: Optional[Path],
    elapsed_seconds: float,
) -> Path:
    """
    Writes a minimal Markdown model card for the trained AFC-NeuralFoil checkpoint.
    """
    train_df, val_df, test_df = split_dfs
    stats = afc_dataset_statistics(full_df, validate=False)
    duplicate_report = check_duplicate_afc_samples(full_df, decimals=8, validate=False)

    lines = [
        "# AFC-NeuralFoil Model Card",
        "",
        "## Model",
        "",
        f"- model_size: `{config.model_size}`",
        f"- hidden_width: `{config.resolved_hidden_width()}`",
        f"- hidden_depth: `{config.resolved_hidden_depth()}`",
        f"- baseline_model_size: `{config.baseline_model_size}`",
        f"- exported_npz: `{exported_npz}`",
        "",
        "## Dataset",
        "",
        f"- data_path: `{config.data_path}`",
        f"- n_cases: `{len(full_df)}`",
        f"- n_train / n_val / n_test: `{len(train_df)} / {len(val_df)} / {len(test_df)}`",
        f"- sources: `{stats['sources']}`",
        f"- duplicate_rows: `{duplicate_report['n_duplicate_rows']}`",
    ]
    if dataset_manifest is not None:
        lines.extend(
            [
                f"- dataset_id: `{dataset_manifest.get('dataset_id')}`",
                f"- schema_version: `{dataset_manifest.get('schema_version')}`",
                f"- manifest_created_at: `{dataset_manifest.get('created_at')}`",
                f"- split_method: `{dataset_manifest.get('split_method')}`",
            ]
        )
        limitations = dataset_manifest.get("known_limitations", [])
        if limitations:
            lines.extend(["", "## Known Limitations", ""])
            lines.extend([f"- {item}" for item in limitations])

    lines.extend(
        [
            "",
            "## Training",
            "",
            f"- epochs_requested: `{config.epochs}`",
            f"- optimizer: `{config.optimizer}`",
            f"- learning_rate: `{config.learning_rate}`",
            f"- elapsed_seconds: `{elapsed_seconds:.3f}`",
            "",
            "## Test Metrics",
            "",
        ]
    )
    for key, value in test_metrics.items():
        lines.append(f"- {key}: `{value:.6g}`")

    lines.extend(
        [
            "",
            "## Test Error Summary",
            "",
            f"- csv: `{error_summary_path}`",
            f"- predictions: `{predictions_path}`",
            "",
            _dataframe_to_markdown_table(error_summary),
            "",
            "## Prediction Plots",
            "",
        ]
    )
    plot_paths = list(plot_paths)
    if plot_paths:
        lines.extend([f"- `{path}`" for path in plot_paths])
    else:
        lines.append("- No plots generated.")

    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return filepath


def resolve_device(config: TrainingConfig) -> str:
    _require_torch()
    if config.device != "auto":
        return config.device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def make_loaders(
    df: pd.DataFrame,
    config: TrainingConfig,
) -> Tuple[object, object, object, Dict[str, onp.ndarray], Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    train_df, val_df, test_df = split_afc_dataset(
        df,
        train_fraction=config.train_fraction,
        val_fraction=config.val_fraction,
        test_fraction=config.test_fraction,
        random_seed=config.random_seed,
    )
    stats = make_normalization_stats(
        train_df, min_gate_for_output_norm=config.min_gate_for_output_norm
    )

    train_dataset = AFCNeuralFoilTorchDataset(train_df)
    val_dataset = AFCNeuralFoilTorchDataset(val_df)
    test_dataset = AFCNeuralFoilTorchDataset(test_df)

    generator = torch.Generator()
    generator.manual_seed(config.random_seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    return train_loader, val_loader, test_loader, stats, (train_df, val_df, test_df)


def save_checkpoint(
    filepath: Path,
    *,
    model,
    optimizer,
    scheduler,
    config: TrainingConfig,
    epoch: int,
    metrics: Dict[str, float],
) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "config": asdict(config),
            "architecture": {
                "hidden_width": config.resolved_hidden_width(),
                "hidden_depth": config.resolved_hidden_depth(),
                "input_dim": AFC_N_INPUTS,
                "output_dim": AFC_N_OUTPUTS,
            },
            "epoch": epoch,
            "metrics": metrics,
        },
        filepath,
    )


def train(config: TrainingConfig) -> Dict[str, object]:
    _require_torch()
    torch.manual_seed(config.random_seed)
    onp.random.seed(config.random_seed)

    output_dir = Path(config.output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_config(config, output_dir / "resolved_config.json")

    df = read_afc_dataset(config.data_path)
    dataset_manifest = _read_optional_dataset_manifest(config)
    validate_afc_dataframe(df)
    df = ensure_neuralfoil_baseline_columns(
        df,
        model_size=config.baseline_model_size,
        overwrite=config.overwrite_baseline_columns,
    )

    train_loader, val_loader, test_loader, stats, split_dfs = make_loaders(df, config)
    train_df, val_df, test_df = split_dfs
    confidence_statistics = make_afc_training_statistics(
        train_df,
        dataset_id=Path(config.data_path).stem,
        validate=False,
    )
    confidence_statistics_path = write_afc_training_statistics(
        confidence_statistics,
        output_dir / "afc_confidence_training_statistics.json",
    )
    device = resolve_device(config)

    model = AFCNeuralFoilMLP(
        hidden_width=config.resolved_hidden_width(),
        hidden_depth=config.resolved_hidden_depth(),
        input_mean=stats["input_mean"],
        input_std=stats["input_std"],
        output_mean=stats["output_mean"],
        output_std=stats["output_std"],
    ).to(device)

    optimizer_name = config.optimizer.lower()
    if optimizer_name == "radam":
        optimizer = torch.optim.RAdam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    else:
        raise ValueError('`optimizer` must be either "adamw" or "radam".')

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.lr_plateau_factor,
        patience=config.lr_plateau_patience,
    )

    best_val_loss = math.inf
    epochs_without_improvement = 0
    history = []
    start_time = time.time()

    latest_checkpoint = output_dir / "checkpoint_latest.pt"
    best_checkpoint = output_dir / "checkpoint_best.pt"

    for epoch in range(1, config.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            config=config,
            device=device,
            optimizer=optimizer,
        )
        val_metrics = run_epoch(model, val_loader, config=config, device=device)
        scheduler.step(val_metrics["loss"])

        metrics = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            **{f"train_{k}": v for k, v in train_metrics.items() if k != "loss"},
            **{f"val_{k}": v for k, v in val_metrics.items() if k != "loss"},
        }
        history.append(metrics)

        save_checkpoint(
            latest_checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            epoch=epoch,
            metrics=metrics,
        )

        improved = (
            best_val_loss - val_metrics["loss"] > config.early_stopping_min_delta
        )
        if improved:
            best_val_loss = val_metrics["loss"]
            epochs_without_improvement = 0
            save_checkpoint(
                best_checkpoint,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config,
                epoch=epoch,
                metrics=metrics,
            )
        else:
            epochs_without_improvement += 1

        print(
            f"epoch={epoch:04d} "
            f"train_loss={train_metrics['loss']:.6g} "
            f"val_loss={val_metrics['loss']:.6g} "
            f"lr={optimizer.param_groups[0]['lr']:.3g}"
        )

        if epochs_without_improvement >= config.early_stopping_patience:
            print(
                f"Early stopping after {epoch} epochs; best validation loss was {best_val_loss:.6g}."
            )
            break

    if best_checkpoint.exists():
        checkpoint = torch.load(best_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = run_epoch(model, test_loader, config=config, device=device)
    elapsed_seconds = time.time() - start_time

    test_predictions = predict_dataframe(
        model,
        test_df,
        config=config,
        device=device,
    )
    predictions_path = output_dir / "test_predictions.csv"
    test_predictions.to_csv(predictions_path, index=False)

    error_summary = make_test_error_summary(test_predictions)
    error_summary_path = output_dir / "test_error_summary.csv"
    error_summary.to_csv(error_summary_path, index=False)

    plot_paths = (
        save_prediction_vs_truth_plots(
            test_predictions,
            output_dir / "prediction_vs_truth",
            max_points=config.n_prediction_plot_points,
        )
        if config.save_prediction_plots
        else []
    )

    metrics_path = output_dir / "history.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "history": history,
                "test_metrics": test_metrics,
                "elapsed_seconds": elapsed_seconds,
                "artifacts": {
                    "test_predictions": str(predictions_path),
                    "test_error_summary": str(error_summary_path),
                    "prediction_vs_truth_plots": [str(path) for path in plot_paths],
                    "afc_confidence_training_statistics": str(
                        confidence_statistics_path
                    ),
                },
            },
            f,
            indent=2,
        )

    exported_npz = None
    if config.export_best_npz and best_checkpoint.exists():
        from aerosandbox.aerodynamics.aero_2D.afc_neuralfoil_training.export_weights import (
            export_checkpoint_to_npz,
        )

        exported_npz = export_checkpoint_to_npz(
            checkpoint_path=best_checkpoint,
            output_directory=output_dir / "exported_weights",
            model_size=config.model_size,
            overwrite=True,
        )

    model_card_path = write_model_card(
        output_dir / "model_card.md",
        config=config,
        dataset_manifest=dataset_manifest,
        full_df=df,
        split_dfs=split_dfs,
        test_metrics=test_metrics,
        error_summary=error_summary,
        predictions_path=predictions_path,
        error_summary_path=error_summary_path,
        plot_paths=plot_paths,
        exported_npz=exported_npz,
        elapsed_seconds=elapsed_seconds,
    )

    return {
        "best_checkpoint": best_checkpoint,
        "latest_checkpoint": latest_checkpoint,
        "history_path": metrics_path,
        "model_card_path": model_card_path,
        "test_predictions_path": predictions_path,
        "test_error_summary_path": error_summary_path,
        "confidence_statistics_path": confidence_statistics_path,
        "prediction_plot_paths": plot_paths,
        "exported_npz": exported_npz,
        "test_metrics": test_metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an AFC-NeuralFoil MLP.")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config.")
    parser.add_argument("--data", type=str, default=None, help="Override dataset path.")
    parser.add_argument("--output-directory", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--model-size", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.data is not None:
        config.data_path = args.data
    if args.output_directory is not None:
        config.output_directory = args.output_directory
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.model_size is not None:
        config.model_size = args.model_size

    result = train(config)
    print("Training complete.")
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
