from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Set, Union
import re

import numpy as onp

import aerosandbox.numpy as np

from aerosandbox.aerodynamics.aero_2D.afc_confidence import evaluate_afc_confidence


AFC_N_INPUTS = 30
AFC_N_OUTPUTS = 6
AFC_OUTPUT_KEYS = [
    "analysis_confidence",
    "CL",
    "CD",
    "CM",
    "Top_Xtr",
    "Bot_Xtr",
]

DEFAULT_AFC_NEURALFOIL_WEIGHTS_DIR = (
    Path(__file__).parent / "afc_neuralfoil_weights"
)

_nn_parameters_cache: Dict[tuple[str, Path], Dict[str, np.ndarray]] = {}

_eps: float = 10 / np.finfo(np.array(1.0).dtype).max
_ln_eps: float = np.log(_eps)


def _sigmoid(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    x = np.clip(x, _ln_eps, -_ln_eps)
    return 1 / (1 + np.exp(-x))


def get_available_afc_model_sizes(
    weights_directory: Union[str, Path] = DEFAULT_AFC_NEURALFOIL_WEIGHTS_DIR,
) -> Set[str]:
    """
    Returns AFC-NeuralFoil model sizes with available `.npz` weight files.
    """
    weights_directory = Path(weights_directory)
    if not weights_directory.exists():
        return set()

    return {
        re.search(r"nn-(.*).npz", path.name).group(1)
        for path in weights_directory.glob("nn-*.npz")
        if re.search(r"nn-(.*).npz", path.name) is not None
    }


def load_afc_neuralfoil_weights(
    model_size: str = "small",
    weights_directory: Union[str, Path] = DEFAULT_AFC_NEURALFOIL_WEIGHTS_DIR,
) -> Dict[str, np.ndarray]:
    """
    Loads AFC-NeuralFoil-style MLP weights from `nn-{model_size}.npz`.

    The expected key format matches NeuralFoil's lightweight NumPy network files:
    `net.0.weight`, `net.0.bias`, `net.2.weight`, `net.2.bias`, etc.
    """
    weights_directory = Path(weights_directory)
    cache_key = (model_size, weights_directory.resolve())
    if cache_key in _nn_parameters_cache:
        return _nn_parameters_cache[cache_key]

    filepath = weights_directory / f"nn-{model_size}.npz"
    if not filepath.exists():
        available = sorted(get_available_afc_model_sizes(weights_directory))
        raise FileNotFoundError(
            f'AFC-NeuralFoil weights not found at "{filepath}". '
            f"Available model sizes in this directory are {available}. "
            "Run `python aerosandbox/aerodynamics/aero_2D/generate_dummy_afc_neuralfoil_weights.py` "
            "to generate placeholder weights."
        )

    params = dict(onp.load(filepath))
    _validate_weight_shapes(params=params)
    _nn_parameters_cache[cache_key] = params
    return params


def _validate_weight_shapes(params: Dict[str, np.ndarray]) -> None:
    layer_indices = _get_layer_indices(params)
    first_weight = params[f"net.{layer_indices[0]}.weight"]
    last_weight = params[f"net.{layer_indices[-1]}.weight"]

    if first_weight.shape[1] != AFC_N_INPUTS:
        raise ValueError(
            f"AFC-NeuralFoil first layer expects {first_weight.shape[1]} inputs, "
            f"but this implementation uses {AFC_N_INPUTS}."
        )
    if last_weight.shape[0] != AFC_N_OUTPUTS:
        raise ValueError(
            f"AFC-NeuralFoil last layer returns {last_weight.shape[0]} outputs, "
            f"but this implementation uses {AFC_N_OUTPUTS}."
        )


def _get_layer_indices(params: Dict[str, np.ndarray]) -> List[int]:
    try:
        layer_indices = sorted(
            {
                int(match.group(1))
                for key in params.keys()
                if (match := re.fullmatch(r"net\.(\d+)\.(weight|bias)", key))
                is not None
            }
        )
        if len(layer_indices) == 0:
            raise ValueError
        return layer_indices
    except Exception as e:
        raise ValueError(
            "Unexpected AFC-NeuralFoil weight file format. Expected keys such as "
            '`"net.0.weight"`, `"net.0.bias"`, `"net.2.weight"`, etc.'
        ) from e


def make_afc_neuralfoil_feature_matrix(
    kulfan_parameters: Dict[str, Union[float, np.ndarray]],
    alpha: Union[float, np.ndarray],
    Re: Union[float, np.ndarray],
    Mach: Union[float, np.ndarray] = 0.0,
    n_crit: Union[float, np.ndarray] = 9.0,
    xtr_upper: Union[float, np.ndarray] = 1.0,
    xtr_lower: Union[float, np.ndarray] = 1.0,
    C_mu: Union[float, np.ndarray] = 0.0,
    x_jet: Union[float, np.ndarray] = 0.1,
    theta_jet: Union[float, np.ndarray] = 30.0,
) -> np.ndarray:
    """
    Builds the AFC-NeuralFoil MLP input feature matrix.

    Features are flat and vectorized. Airfoil geometry is represented by 8+8 Kulfan weights, leading-edge weight, and
    trailing-edge thickness. Angle-like variables use sine/cosine encodings, and Reynolds number uses a log encoding.
    """
    input_rows: List[Union[float, np.ndarray]] = [
        *[kulfan_parameters["upper_weights"][i] for i in range(8)],
        *[kulfan_parameters["lower_weights"][i] for i in range(8)],
        kulfan_parameters["leading_edge_weight"],
        kulfan_parameters["TE_thickness"] * 50,
        np.sind(alpha),
        np.cosd(alpha),
        np.sind(2 * alpha),
        (np.log(Re) - 12.5) / 3.5,
        Mach,
        (n_crit - 9) / 4.5,
        xtr_upper,
        xtr_lower,
        C_mu,
        x_jet,
        np.sind(theta_jet),
        np.cosd(theta_jet),
    ]

    N_cases = 1
    for row in input_rows:
        if np.length(row) > 1:
            if N_cases == 1:
                N_cases = np.length(row)
            elif np.length(row) != N_cases:
                raise ValueError(
                    "All AFC-NeuralFoil vectorized inputs must have the same length. "
                    f"Got conflicting lengths {N_cases} and {np.length(row)}."
                )

    for i, row in enumerate(input_rows):
        input_rows[i] = np.ones(N_cases) * row

    return np.stack(input_rows, axis=1)


def _evaluate_mlp(x: np.ndarray, params: Dict[str, np.ndarray]) -> np.ndarray:
    layer_indices = _get_layer_indices(params)
    x = np.transpose(x)

    layer_indices_to_iterate = layer_indices.copy()
    while len(layer_indices_to_iterate) != 0:
        i = layer_indices_to_iterate.pop(0)
        w = params[f"net.{i}.weight"]
        b = params[f"net.{i}.bias"]
        x = w @ x + np.reshape(b, (-1, 1))

        if len(layer_indices_to_iterate) != 0:
            x = np.swish(x)

    return np.transpose(x)


def get_aero_from_afc_kulfan_parameters(
    kulfan_parameters: Dict[str, Union[float, np.ndarray]],
    alpha: Union[float, np.ndarray],
    Re: Union[float, np.ndarray],
    Mach: Union[float, np.ndarray] = 0.0,
    n_crit: Union[float, np.ndarray] = 9.0,
    xtr_upper: Union[float, np.ndarray] = 1.0,
    xtr_lower: Union[float, np.ndarray] = 1.0,
    C_mu: Union[float, np.ndarray] = 0.0,
    x_jet: Union[float, np.ndarray] = 0.1,
    theta_jet: Union[float, np.ndarray] = 30.0,
    model_size: str = "small",
    weights_directory: Union[str, Path] = DEFAULT_AFC_NEURALFOIL_WEIGHTS_DIR,
    zero_afc_behavior: str = "baseline_correction",
    baseline_model_size: str = "large",
    training_statistics: Optional[Union[str, Path, Mapping[str, object]]] = None,
    training_statistics_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Evaluates an AFC-NeuralFoil-style MLP from Kulfan parameters and flow/AFC features.

    The current network convention predicts smooth corrections to a NeuralFoil baseline. This makes the no-AFC limit
    robust: corrections are multiplied by a smooth gate that is exactly zero when `C_mu = 0`.

    Args:
        kulfan_parameters: Dictionary with `upper_weights`, `lower_weights`, `leading_edge_weight`, and `TE_thickness`.
        alpha: Angle of attack [deg].
        Re: Reynolds number [-].
        Mach: Mach number [-].
        n_crit: Boundary-layer critical amplification factor.
        xtr_upper: Forced upper transition location, x/c.
        xtr_lower: Forced lower transition location, x/c.
        C_mu: AFC jet momentum coefficient [-].
        x_jet: AFC jet location, x/c.
        theta_jet: AFC jet angle [deg].
        model_size: Weight file size label, e.g. `"small"`, `"medium"`, or `"large"`.
        weights_directory: Directory containing `nn-{model_size}.npz`.
        zero_afc_behavior: Either `"baseline_correction"` or `"neuralfoil"`.
            `"baseline_correction"` always evaluates the MLP correction, with a zero correction at `C_mu = 0`.
            `"neuralfoil"` returns the baseline directly when `C_mu` is numerically all zero; symbolic inputs still use
            the smooth baseline-correction path.
        baseline_model_size: NeuralFoil model size used for the baseline.
        training_statistics: Optional AFC confidence statistics dict or JSON path for training-distance OOD checks.
        training_statistics_path: Optional AFC confidence statistics JSON path.

    Returns:
        A dict with `CL`, `CD`, `CM`, `Top_Xtr`, `Bot_Xtr`, `analysis_confidence`, `ood_score`, `stall_risk`, and
        `confidence_reason`, plus baseline/correction fields.
    """
    if zero_afc_behavior not in {"baseline_correction", "neuralfoil"}:
        raise ValueError(
            '`zero_afc_behavior` must be either "baseline_correction" or "neuralfoil".'
        )

    baseline = _get_neuralfoil_baseline_from_kulfan_parameters(
        kulfan_parameters=kulfan_parameters,
        alpha=alpha,
        Re=Re,
        Mach=Mach,
        n_crit=n_crit,
        xtr_upper=xtr_upper,
        xtr_lower=xtr_lower,
        model_size=baseline_model_size,
    )

    if zero_afc_behavior == "neuralfoil" and _is_numerically_all_zero(C_mu):
        confidence = evaluate_afc_confidence(
            kulfan_parameters=kulfan_parameters,
            alpha=alpha,
            Re=Re,
            Mach=Mach,
            C_mu=C_mu,
            x_jet=x_jet,
            theta_jet=theta_jet,
            section_CL=baseline["CL"],
            Top_Xtr=baseline["Top_Xtr"],
            Bot_Xtr=baseline["Bot_Xtr"],
            base_analysis_confidence=baseline["analysis_confidence"],
            training_statistics=training_statistics,
            training_statistics_path=training_statistics_path,
        )
        baseline = dict(baseline)
        baseline.update(confidence)
        return baseline

    params = load_afc_neuralfoil_weights(
        model_size=model_size,
        weights_directory=weights_directory,
    )
    x = make_afc_neuralfoil_feature_matrix(
        kulfan_parameters=kulfan_parameters,
        alpha=alpha,
        Re=Re,
        Mach=Mach,
        n_crit=n_crit,
        xtr_upper=xtr_upper,
        xtr_lower=xtr_lower,
        C_mu=C_mu,
        x_jet=x_jet,
        theta_jet=theta_jet,
    )
    y = _evaluate_mlp(x=x, params=params)

    afc_gate = 1 - np.exp(-60 * C_mu)

    model_confidence = _sigmoid(y[:, 0])
    dCL = afc_gate * y[:, 1] / 2
    dlogCD = afc_gate * y[:, 2] / 5
    dCM = afc_gate * y[:, 3] / 20
    dTop_Xtr = afc_gate * y[:, 4] / 5
    dBot_Xtr = afc_gate * y[:, 5] / 5

    CL = baseline["CL"] + dCL
    CD = baseline["CD"] * np.exp(dlogCD)
    CM = baseline["CM"] + dCM
    Top_Xtr = np.clip(baseline["Top_Xtr"] + dTop_Xtr, 0, 1)
    Bot_Xtr = np.clip(baseline["Bot_Xtr"] + dBot_Xtr, 0, 1)
    raw_analysis_confidence = baseline["analysis_confidence"] * (
        1 - afc_gate * (1 - model_confidence)
    )
    confidence = evaluate_afc_confidence(
        kulfan_parameters=kulfan_parameters,
        alpha=alpha,
        Re=Re,
        Mach=Mach,
        C_mu=C_mu,
        x_jet=x_jet,
        theta_jet=theta_jet,
        section_CL=CL,
        Top_Xtr=Top_Xtr,
        Bot_Xtr=Bot_Xtr,
        base_analysis_confidence=raw_analysis_confidence,
        training_statistics=training_statistics,
        training_statistics_path=training_statistics_path,
    )

    output = {
        "analysis_confidence": confidence["analysis_confidence"],
        "CL": np.reshape(CL, -1),
        "CD": np.reshape(CD, -1),
        "CM": np.reshape(CM, -1),
        "Top_Xtr": np.reshape(Top_Xtr, -1),
        "Bot_Xtr": np.reshape(Bot_Xtr, -1),
        "CL_baseline": baseline["CL"],
        "CD_baseline": baseline["CD"],
        "CM_baseline": baseline["CM"],
        "Top_Xtr_baseline": baseline["Top_Xtr"],
        "Bot_Xtr_baseline": baseline["Bot_Xtr"],
        "analysis_confidence_baseline": baseline["analysis_confidence"],
        "dCL_afc": np.reshape(dCL, -1),
        "dCD_afc": np.reshape(CD - baseline["CD"], -1),
        "dCM_afc": np.reshape(dCM, -1),
        "afc_gate": np.reshape(afc_gate, -1),
        "afc_model_confidence": np.reshape(model_confidence, -1),
        "analysis_confidence_before_ood": np.reshape(raw_analysis_confidence, -1),
    }
    output.update(
        {
            key: value
            for key, value in confidence.items()
            if key != "analysis_confidence"
        }
    )
    return output


def get_aero_from_afc_airfoil(
    airfoil,
    alpha: Union[float, np.ndarray],
    Re: Union[float, np.ndarray],
    Mach: Union[float, np.ndarray] = 0.0,
    n_crit: Union[float, np.ndarray] = 9.0,
    xtr_upper: Union[float, np.ndarray] = 1.0,
    xtr_lower: Union[float, np.ndarray] = 1.0,
    C_mu: Union[float, np.ndarray] = 0.0,
    x_jet: Union[float, np.ndarray] = 0.1,
    theta_jet: Union[float, np.ndarray] = 30.0,
    model_size: str = "small",
    weights_directory: Union[str, Path] = DEFAULT_AFC_NEURALFOIL_WEIGHTS_DIR,
    zero_afc_behavior: str = "baseline_correction",
    baseline_model_size: str = "large",
    training_statistics: Optional[Union[str, Path, Mapping[str, object]]] = None,
    training_statistics_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Evaluates the AFC-NeuralFoil MLP for an AeroSandbox Airfoil or KulfanAirfoil object.
    """
    normalization_outputs = airfoil.normalize(return_dict=True)
    normalized_airfoil = normalization_outputs["airfoil"].to_kulfan_airfoil(
        n_weights_per_side=8,
        normalize_coordinates=False,
    )
    delta_alpha = normalization_outputs["rotation_angle"]
    x_translation_LE = normalization_outputs["x_translation"]
    y_translation_LE = normalization_outputs["y_translation"]
    scale = normalization_outputs["scale_factor"]

    x_translation_qc = (
        -x_translation_LE + 0.25 * (1 / scale * np.cosd(delta_alpha)) - 0.25
    )
    y_translation_qc = -y_translation_LE + 0.25 * (1 / scale * np.sind(-delta_alpha))

    raw_aero = get_aero_from_afc_kulfan_parameters(
        kulfan_parameters=normalized_airfoil.kulfan_parameters,
        alpha=alpha + delta_alpha,
        Re=Re / scale,
        Mach=Mach,
        n_crit=n_crit,
        xtr_upper=xtr_upper,
        xtr_lower=xtr_lower,
        C_mu=C_mu,
        x_jet=x_jet,
        theta_jet=theta_jet,
        model_size=model_size,
        weights_directory=weights_directory,
        zero_afc_behavior=zero_afc_behavior,
        baseline_model_size=baseline_model_size,
        training_statistics=training_statistics,
        training_statistics_path=training_statistics_path,
    )

    raw_aero["CM"] = raw_aero["CM"] + (
        -raw_aero["CL"] * x_translation_qc + raw_aero["CD"] * y_translation_qc
    )

    return raw_aero


def _get_neuralfoil_baseline_from_kulfan_parameters(
    kulfan_parameters: Dict[str, Union[float, np.ndarray]],
    alpha: Union[float, np.ndarray],
    Re: Union[float, np.ndarray],
    Mach: Union[float, np.ndarray],
    n_crit: Union[float, np.ndarray],
    xtr_upper: Union[float, np.ndarray],
    xtr_lower: Union[float, np.ndarray],
    model_size: str,
) -> Dict[str, Union[float, np.ndarray]]:
    from aerosandbox.geometry.airfoil.kulfan_airfoil import KulfanAirfoil

    airfoil = KulfanAirfoil(
        lower_weights=kulfan_parameters["lower_weights"],
        upper_weights=kulfan_parameters["upper_weights"],
        leading_edge_weight=kulfan_parameters["leading_edge_weight"],
        TE_thickness=kulfan_parameters["TE_thickness"],
    )
    return airfoil.get_aero_from_neuralfoil(
        alpha=alpha,
        Re=Re,
        mach=Mach,
        n_crit=n_crit,
        xtr_upper=xtr_upper,
        xtr_lower=xtr_lower,
        model_size=model_size,
    )


def _is_numerically_all_zero(value) -> bool:
    try:
        return bool(onp.all(onp.asarray(value, dtype=float) == 0))
    except Exception:
        return False


def generate_dummy_afc_neuralfoil_weights(
    output_directory: Union[str, Path] = DEFAULT_AFC_NEURALFOIL_WEIGHTS_DIR,
    model_sizes: Iterable[str] = ("small", "medium", "large"),
    *,
    overwrite: bool = True,
    random_seed: int = 0,
) -> List[Path]:
    """
    Generates deterministic dummy AFC-NeuralFoil `.npz` MLP weights.

    These weights are only for interface testing. They are smooth, small-amplitude random networks and are not trained
    on CFD, XFoil, or experimental data.
    """
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    hidden_widths = {
        "small": 16,
        "medium": 32,
        "large": 64,
    }

    written = []
    for model_size in model_sizes:
        if model_size not in hidden_widths:
            raise ValueError(
                f'No dummy architecture defined for model_size="{model_size}".'
            )

        filepath = output_directory / f"nn-{model_size}.npz"
        if filepath.exists() and not overwrite:
            written.append(filepath)
            continue

        rng = onp.random.default_rng(
            random_seed + sum(ord(char) for char in model_size)
        )
        width = hidden_widths[model_size]
        layer_sizes = [AFC_N_INPUTS, width, width, AFC_N_OUTPUTS]

        params = {}
        for layer_number, (n_in, n_out) in enumerate(
            zip(layer_sizes[:-1], layer_sizes[1:])
        ):
            net_index = 2 * layer_number
            scale = 0.18 / max(n_in, 1) ** 0.5
            params[f"net.{net_index}.weight"] = rng.normal(
                loc=0.0, scale=scale, size=(n_out, n_in)
            ).astype(onp.float64)
            params[f"net.{net_index}.bias"] = rng.normal(
                loc=0.0, scale=0.02, size=n_out
            ).astype(onp.float64)

        params[f"net.{2 * (len(layer_sizes) - 2)}.bias"][0] = 2.0
        onp.savez(filepath, **params)
        written.append(filepath)

    _nn_parameters_cache.clear()
    return written
