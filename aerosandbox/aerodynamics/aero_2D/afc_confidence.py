from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Union

import numpy as onp
import pandas as pd

import aerosandbox.numpy as np

from aerosandbox.aerodynamics.aero_2D.afc_dataset import (
    KULFAN_LOWER_COLUMNS,
    KULFAN_UPPER_COLUMNS,
    read_afc_dataset,
    validate_afc_dataframe,
)


AFC_CONFIDENCE_FEATURE_COLUMNS = [
    *KULFAN_UPPER_COLUMNS,
    *KULFAN_LOWER_COLUMNS,
    "leading_edge_weight",
    "TE_thickness",
    "alpha",
    "log_Re",
    "Mach",
    "C_mu",
    "x_jet",
    "theta_jet",
]


DEFAULT_AFC_INPUT_RANGES = {
    **{column: (-1.5, 1.5) for column in [*KULFAN_UPPER_COLUMNS, *KULFAN_LOWER_COLUMNS]},
    "leading_edge_weight": (-1.0, 1.0),
    "TE_thickness": (0.0, 0.12),
    "alpha": (-20.0, 30.0),  # deg; AFC training-quality envelope, not NeuralFoil's full 360 deg envelope.
    "Re": (5e4, 5e7),
    "Mach": (0.0, 0.70),
    "C_mu": (0.0, 0.12),
    "x_jet": (0.0, 0.60),
    "theta_jet": (-10.0, 100.0),
}


def _safe_sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -60, 60)))


def _case_count(values: Iterable[object]) -> int:
    n_cases = 1
    for value in values:
        if np.length(value) > 1:
            if n_cases == 1:
                n_cases = np.length(value)
            elif np.length(value) != n_cases:
                raise ValueError(
                    "All AFC confidence vectorized inputs must have the same length."
                )
    return n_cases


def _broadcast(value, n_cases: int):
    return np.reshape(np.ones(n_cases) * value, -1)


def _as_numeric_array(value):
    return onp.asarray(value, dtype=float)


def _is_numeric(value) -> bool:
    try:
        _as_numeric_array(value)
        return True
    except Exception:
        return False


def _range_violation(value, lower: float, upper: float):
    scale = max(upper - lower, 1e-12)
    return np.maximum((lower - value) / scale, 0) + np.maximum((value - upper) / scale, 0)


def make_afc_confidence_feature_matrix(
    kulfan_parameters: Mapping[str, object],
    *,
    alpha,
    Re,
    Mach=0.0,
    C_mu=0.0,
    x_jet=0.1,
    theta_jet=30.0,
):
    """
    Builds the raw feature matrix used by AFC training-distance/OOD checks.

    This is intentionally close to the dataset schema, with `Re` represented as `log_Re`.
    """
    rows = [
        *[kulfan_parameters["upper_weights"][i] for i in range(8)],
        *[kulfan_parameters["lower_weights"][i] for i in range(8)],
        kulfan_parameters["leading_edge_weight"],
        kulfan_parameters["TE_thickness"],
        alpha,
        np.log(np.maximum(Re, 1e-300)),
        Mach,
        C_mu,
        x_jet,
        theta_jet,
    ]
    n_cases = _case_count(rows)
    rows = [_broadcast(row, n_cases) for row in rows]
    return np.stack(rows, axis=1)


def input_domain_ood_score(
    kulfan_parameters: Mapping[str, object],
    *,
    alpha,
    Re,
    Mach=0.0,
    C_mu=0.0,
    x_jet=0.1,
    theta_jet=30.0,
    ranges: Optional[Mapping[str, Sequence[float]]] = None,
):
    """
    Differentiable input-domain risk score in `[0, 1]`.

    `0` means all checked values are within the nominal AFC surrogate range. Values approach `1` as inputs move farther
    outside the range. This is a heuristic applicability score, not calibrated epistemic uncertainty.
    """
    ranges = dict(DEFAULT_AFC_INPUT_RANGES if ranges is None else ranges)

    checked_values = {
        **{KULFAN_UPPER_COLUMNS[i]: kulfan_parameters["upper_weights"][i] for i in range(8)},
        **{KULFAN_LOWER_COLUMNS[i]: kulfan_parameters["lower_weights"][i] for i in range(8)},
        "leading_edge_weight": kulfan_parameters["leading_edge_weight"],
        "TE_thickness": kulfan_parameters["TE_thickness"],
        "alpha": alpha,
        "Re": Re,
        "Mach": Mach,
        "C_mu": C_mu,
        "x_jet": x_jet,
        "theta_jet": theta_jet,
    }

    n_cases = _case_count(checked_values.values())
    total = np.zeros(n_cases)
    for column, value in checked_values.items():
        lower, upper = ranges[column]
        violation = _range_violation(_broadcast(value, n_cases), lower, upper)
        weight = 0.08 if column in [*KULFAN_UPPER_COLUMNS, *KULFAN_LOWER_COLUMNS] else 1.0
        total = total + weight * violation**2

    return np.clip(1 - np.exp(-3.0 * total), 0, 1)


def _training_dataframe_to_feature_array(df: pd.DataFrame) -> onp.ndarray:
    values = []
    for column in AFC_CONFIDENCE_FEATURE_COLUMNS:
        if column == "log_Re":
            values.append(onp.log(onp.clip(df["Re"].to_numpy(float), 1e-300, None)))
        else:
            values.append(df[column].to_numpy(float))
    return onp.stack(values, axis=1)


def make_afc_training_statistics(
    df: pd.DataFrame,
    *,
    dataset_id: str = "afc_training_dataset",
    validate: bool = True,
) -> Dict[str, object]:
    """
    Makes compact feature statistics for AFC-NeuralFoil OOD distance checks.

    These statistics are intentionally lightweight JSON data: feature means/stds/mins/maxs and case count. They can be
    generated from synthetic, CFD, RANS, or experimental training data.
    """
    if validate:
        validate_afc_dataframe(df)

    x = _training_dataframe_to_feature_array(df)
    std = onp.nanstd(x, axis=0)
    std = onp.where(std < 1e-10, 1.0, std)

    return {
        "schema_version": "afc_confidence_stats_v1",
        "dataset_id": dataset_id,
        "n_cases": int(len(df)),
        "feature_columns": AFC_CONFIDENCE_FEATURE_COLUMNS,
        "mean": onp.nanmean(x, axis=0).tolist(),
        "std": std.tolist(),
        "min": onp.nanmin(x, axis=0).tolist(),
        "max": onp.nanmax(x, axis=0).tolist(),
        "distance_threshold": 3.0,
        "note": (
            "Training-distance OOD score is a heuristic normalized feature-space distance, "
            "not rigorous uncertainty quantification."
        ),
    }


def make_afc_training_statistics_from_file(
    filepath: Union[str, Path],
    *,
    dataset_id: Optional[str] = None,
) -> Dict[str, object]:
    df = read_afc_dataset(filepath)
    return make_afc_training_statistics(
        df,
        dataset_id=dataset_id or Path(filepath).stem,
        validate=False,
    )


def write_afc_training_statistics(
    statistics: Mapping[str, object],
    filepath: Union[str, Path],
    *,
    indent: int = 2,
) -> Path:
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(dict(statistics), f, indent=indent)
    return filepath


def read_afc_training_statistics(filepath: Union[str, Path]) -> Dict[str, object]:
    with open(filepath, "r", encoding="utf-8") as f:
        statistics = json.load(f)

    required_fields = ["feature_columns", "mean", "std", "distance_threshold"]
    missing = [field for field in required_fields if field not in statistics]
    if missing:
        raise ValueError(f"AFC confidence statistics are missing required fields: {missing}")
    return statistics


def resolve_afc_training_statistics(
    training_statistics: Optional[Union[str, Path, Mapping[str, object]]] = None,
    training_statistics_path: Optional[Union[str, Path]] = None,
) -> Optional[Dict[str, object]]:
    if training_statistics is not None:
        if isinstance(training_statistics, (str, Path)):
            return read_afc_training_statistics(training_statistics)
        return dict(training_statistics)
    if training_statistics_path is not None:
        return read_afc_training_statistics(training_statistics_path)
    return None


def training_distance_ood_score(
    kulfan_parameters: Mapping[str, object],
    *,
    alpha,
    Re,
    Mach=0.0,
    C_mu=0.0,
    x_jet=0.1,
    theta_jet=30.0,
    training_statistics: Optional[Union[str, Path, Mapping[str, object]]] = None,
    training_statistics_path: Optional[Union[str, Path]] = None,
):
    """
    Differentiable normalized feature-space OOD score from saved training statistics.

    If no statistics are supplied, returns zero risk so callers can still use domain and stall-risk checks.
    """
    training_statistics = resolve_afc_training_statistics(
        training_statistics=training_statistics,
        training_statistics_path=training_statistics_path,
    )
    if training_statistics is None:
        n_cases = _case_count([alpha, Re, Mach, C_mu, x_jet, theta_jet])
        return np.zeros(n_cases), np.zeros(n_cases)

    feature_columns = training_statistics.get("feature_columns")
    if feature_columns is not None and list(feature_columns) != AFC_CONFIDENCE_FEATURE_COLUMNS:
        raise ValueError(
            "AFC confidence statistics feature columns do not match the current AFC confidence schema."
        )

    x = make_afc_confidence_feature_matrix(
        kulfan_parameters=kulfan_parameters,
        alpha=alpha,
        Re=Re,
        Mach=Mach,
        C_mu=C_mu,
        x_jet=x_jet,
        theta_jet=theta_jet,
    )
    mean = np.array(training_statistics["mean"])
    std = np.maximum(np.array(training_statistics["std"]), 1e-10)
    distance = np.sqrt(np.mean(((x - mean) / std) ** 2, axis=1))
    threshold = float(training_statistics.get("distance_threshold", 3.0))
    score = 1 - np.exp(-np.maximum(distance - threshold, 0) ** 2 / 4)
    return np.clip(score, 0, 1), distance


def stall_risk_proxy(
    *,
    alpha,
    section_CL,
    analysis_confidence,
    Top_Xtr=1.0,
    Bot_Xtr=1.0,
):
    """
    Smooth placeholder stall-risk proxy in `[0, 1]`.

    This uses alpha, section lift level, confidence, and transition placeholders. It is intended for ranking and
    warning labels, not for certifying stall margin.
    """
    alpha_term = _safe_sigmoid((alpha - 14.0) / 3.5)
    cl_term = _safe_sigmoid((np.fabs(section_CL) - 1.35) / 0.25)
    confidence_term = np.clip(1 - analysis_confidence, 0, 1)
    transition_term = np.clip(1 - 0.5 * (Top_Xtr + Bot_Xtr), 0, 1)
    return np.clip(
        0.45 * alpha_term
        + 0.35 * cl_term
        + 0.15 * confidence_term
        + 0.05 * transition_term,
        0,
        1,
    )


def _numeric_reasons(
    *,
    domain_score,
    distance_score,
    distance,
    stall_risk,
    ranges: Mapping[str, Sequence[float]],
    values: Mapping[str, object],
) -> Union[str, list[str]]:
    try:
        n_cases = max(
            len(onp.ravel(_as_numeric_array(domain_score))),
            len(onp.ravel(_as_numeric_array(stall_risk))),
        )
    except Exception:
        return "symbolic_confidence_evaluation"

    reasons = []
    for i in range(n_cases):
        row_reasons = []
        try:
            if onp.ravel(_as_numeric_array(domain_score))[i] > 0.15:
                bad = []
                for key, value in values.items():
                    if key not in ranges or not _is_numeric(value):
                        continue
                    arr = onp.ravel(_as_numeric_array(value))
                    v = arr[i] if len(arr) > 1 else arr[0]
                    lower, upper = ranges[key]
                    if v < lower or v > upper:
                        bad.append(key)
                row_reasons.append("input_range:" + ",".join(bad or ["mixed"]))
            if onp.ravel(_as_numeric_array(distance_score))[i] > 0.15:
                d = onp.ravel(_as_numeric_array(distance))[i]
                row_reasons.append(f"training_distance_high:{d:.3g}")
            if onp.ravel(_as_numeric_array(stall_risk))[i] > 0.55:
                row_reasons.append("stall_proxy_high")
        except Exception:
            return "symbolic_confidence_evaluation"
        reasons.append("; ".join(row_reasons) if row_reasons else "in_distribution")

    return reasons[0] if n_cases == 1 else reasons


def evaluate_afc_confidence(
    kulfan_parameters: Mapping[str, object],
    *,
    alpha,
    Re,
    Mach=0.0,
    C_mu=0.0,
    x_jet=0.1,
    theta_jet=30.0,
    section_CL=0.0,
    Top_Xtr=1.0,
    Bot_Xtr=1.0,
    base_analysis_confidence=1.0,
    training_statistics: Optional[Union[str, Path, Mapping[str, object]]] = None,
    training_statistics_path: Optional[Union[str, Path]] = None,
    ranges: Optional[Mapping[str, Sequence[float]]] = None,
) -> Dict[str, object]:
    """
    Combines AFC confidence factors into model-facing outputs.

    `analysis_confidence` remains a heuristic applicability score. It is not a calibrated uncertainty interval or a
    formal probability of correctness.
    """
    ranges = dict(DEFAULT_AFC_INPUT_RANGES if ranges is None else ranges)
    stats = resolve_afc_training_statistics(
        training_statistics=training_statistics,
        training_statistics_path=training_statistics_path,
    )

    domain_score = input_domain_ood_score(
        kulfan_parameters=kulfan_parameters,
        alpha=alpha,
        Re=Re,
        Mach=Mach,
        C_mu=C_mu,
        x_jet=x_jet,
        theta_jet=theta_jet,
        ranges=ranges,
    )
    distance_score, distance = training_distance_ood_score(
        kulfan_parameters=kulfan_parameters,
        alpha=alpha,
        Re=Re,
        Mach=Mach,
        C_mu=C_mu,
        x_jet=x_jet,
        theta_jet=theta_jet,
        training_statistics=stats,
    )
    ood_score = 1 - (1 - domain_score) * (1 - distance_score)

    stall_risk = stall_risk_proxy(
        alpha=alpha,
        section_CL=section_CL,
        analysis_confidence=base_analysis_confidence,
        Top_Xtr=Top_Xtr,
        Bot_Xtr=Bot_Xtr,
    )

    # Preserve the no-AFC aerodynamic-confidence limit exactly while still reporting OOD/stall fields.
    afc_gate = 1 - np.exp(-60 * np.fabs(C_mu))
    confidence_penalty = 0.80 * ood_score + 0.35 * stall_risk
    adjusted_confidence = base_analysis_confidence * np.exp(
        -2.2 * afc_gate * confidence_penalty
    )
    adjusted_confidence = np.clip(adjusted_confidence, 0, 1)

    reason_values = {
        **{
            KULFAN_UPPER_COLUMNS[i]: kulfan_parameters["upper_weights"][i]
            for i in range(8)
        },
        **{
            KULFAN_LOWER_COLUMNS[i]: kulfan_parameters["lower_weights"][i]
            for i in range(8)
        },
        "alpha": alpha,
        "Re": Re,
        "Mach": Mach,
        "C_mu": C_mu,
        "x_jet": x_jet,
        "theta_jet": theta_jet,
        "leading_edge_weight": kulfan_parameters["leading_edge_weight"],
        "TE_thickness": kulfan_parameters["TE_thickness"],
    }
    reason = _numeric_reasons(
        domain_score=domain_score,
        distance_score=distance_score,
        distance=distance,
        stall_risk=stall_risk,
        ranges=ranges,
        values=reason_values,
    )

    return {
        "analysis_confidence": np.reshape(adjusted_confidence, -1),
        "ood_score": np.reshape(ood_score, -1),
        "input_domain_ood_score": np.reshape(domain_score, -1),
        "training_distance_ood_score": np.reshape(distance_score, -1),
        "training_distance": np.reshape(distance, -1),
        "stall_risk": np.reshape(stall_risk, -1),
        "confidence_reason": reason,
    }
