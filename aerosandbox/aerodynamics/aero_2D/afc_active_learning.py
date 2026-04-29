from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple, Union

import json
import numpy as onp
import pandas as pd

from aerosandbox.aerodynamics.aero_2D.afc_dataset import (
    AFC_COLUMNS,
    CONDITION_COLUMNS,
    GEOMETRY_COLUMNS,
    KULFAN_LOWER_COLUMNS,
    KULFAN_UPPER_COLUMNS,
)


AFC_TASK_INPUT_COLUMNS = [
    *GEOMETRY_COLUMNS,
    *CONDITION_COLUMNS,
    *AFC_COLUMNS,
]

AFC_ACTIVE_LEARNING_SCORE_COLUMNS = [
    "predicted_CL",
    "predicted_CD",
    "predicted_CM",
    "predicted_Top_Xtr",
    "predicted_Bot_Xtr",
    "predicted_analysis_confidence",
    "nearest_training_distance",
    "score_uncertainty",
    "score_gradient",
    "score_optimization",
    "score_stall",
    "score_novelty",
    "priority_score",
]

AFC_ACTIVE_LEARNING_METADATA_COLUMNS = [
    "sample_id",
    "reason",
    "source",
]

AFC_ACTIVE_LEARNING_TASK_COLUMNS = [
    *AFC_ACTIVE_LEARNING_METADATA_COLUMNS,
    *AFC_TASK_INPUT_COLUMNS,
    *AFC_ACTIVE_LEARNING_SCORE_COLUMNS,
]

AFC_CFD_TASK_CONDITION_COLUMNS = [
    "alpha",
    "Re",
    "Mach",
    "C_mu",
    "x_jet",
    "theta_jet",
]

AFC_CFD_TASK_CONFIG_COLUMNS = [
    "mesh_config",
    "boundary_condition_config",
]

AFC_CFD_TASK_METADATA_COLUMNS = [
    "task_id",
    "airfoil",
    "solver_type",
    *AFC_CFD_TASK_CONFIG_COLUMNS,
    "priority",
    "priority_score",
    "reason",
    "selection_source",
]

AFC_CFD_TASK_COLUMNS = [
    *AFC_CFD_TASK_METADATA_COLUMNS,
    *GEOMETRY_COLUMNS,
    *AFC_CFD_TASK_CONDITION_COLUMNS,
]

DEFAULT_AFC_TASK_VALUES = {
    **{
        KULFAN_UPPER_COLUMNS[i]: value
        for i, value in enumerate([0.18, 0.20, 0.18, 0.14, 0.10, 0.07, 0.04, 0.02])
    },
    **{
        KULFAN_LOWER_COLUMNS[i]: value
        for i, value in enumerate(
            [-0.10, -0.09, -0.08, -0.06, -0.04, -0.025, -0.015, -0.005]
        )
    },
    "leading_edge_weight": 0.0,
    "TE_thickness": 0.01,
    "alpha": 4.0,
    "Re": 1e6,
    "Mach": 0.1,
    "n_crit": 9.0,
    "xtr_upper": 1.0,
    "xtr_lower": 1.0,
    "C_mu": 0.02,
    "x_jet": 0.1,
    "theta_jet": 30.0,
}

DEFAULT_AFC_CFD_TASK_CONFIG = {
    "solver_type": "SU2_RANS",
    "mesh_config": {
        "family": "placeholder_rans_2d",
        "chordwise_points": 257,
        "normal_growth_rate": 1.15,
        "y_plus_target": 1.0,
    },
    "boundary_condition_config": {
        "freestream_spec": "alpha_Re_Mach",
        "actuation_spec": "C_mu_x_jet_theta_jet",
        "transition_model": "placeholder_or_user_defined",
    },
}

AFC_ACTIVE_LEARNING_DEFAULT_WEIGHTS = {
    "uncertainty": 0.38,
    "gradient": 0.20,
    "optimization": 0.18,
    "stall": 0.16,
    "novelty": 0.08,
}


def _validate_design_space(
    design_space: Dict[str, Tuple[float, float]],
) -> Dict[str, Tuple[float, float]]:
    if not isinstance(design_space, dict):
        raise TypeError("`design_space` must be a dictionary of column -> (lower, upper).")

    normalized = {}
    for column, bounds in design_space.items():
        if column not in AFC_TASK_INPUT_COLUMNS:
            raise ValueError(
                f'Unknown AFC design-space column "{column}". Expected one of {AFC_TASK_INPUT_COLUMNS}.'
            )

        if len(bounds) != 2:
            raise ValueError(
                f'Design-space bounds for "{column}" must be a two-item tuple.'
            )

        lower, upper = float(bounds[0]), float(bounds[1])
        if not (onp.isfinite(lower) and onp.isfinite(upper)):
            raise ValueError(f'Design-space bounds for "{column}" must be finite.')
        if upper < lower:
            raise ValueError(
                f'Design-space bounds for "{column}" must satisfy upper >= lower.'
            )

        normalized[column] = (lower, upper)

    return normalized


def fill_missing_afc_task_inputs(
    df: pd.DataFrame,
    *,
    fixed_values: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Adds missing AFC task input columns using canonical defaults or user-supplied fixed values.

    This is intentionally separate from `validate_afc_dataframe()`: a CFD task list has inputs and active-learning
    metadata, but does not yet have CFD output labels such as `CL`, `CD`, or `CM`.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("`df` must be a pandas DataFrame.")

    fixed_values = {} if fixed_values is None else dict(fixed_values)
    unknown_fixed = [column for column in fixed_values if column not in AFC_TASK_INPUT_COLUMNS]
    if unknown_fixed:
        raise ValueError(f"Unknown fixed AFC task input columns: {unknown_fixed}")

    out = df.copy()
    for column in AFC_TASK_INPUT_COLUMNS:
        if column in fixed_values:
            out[column] = fixed_values[column]
        elif column not in out.columns:
            out[column] = DEFAULT_AFC_TASK_VALUES[column]

    for column in AFC_TASK_INPUT_COLUMNS:
        values = pd.to_numeric(out[column], errors="coerce")
        if values.isna().any():
            raise ValueError(f'Column "{column}" contains non-numeric or NaN values.')
        if not onp.all(onp.isfinite(values.to_numpy(dtype=float))):
            raise ValueError(f'Column "{column}" contains non-finite values.')
        out[column] = values

    return out[AFC_TASK_INPUT_COLUMNS]


def latin_hypercube_sample_afc_design_space(
    design_space: Dict[str, Tuple[float, float]],
    n_samples: int,
    *,
    fixed_values: Optional[Dict[str, float]] = None,
    random_seed: int = 0,
) -> pd.DataFrame:
    """
    Generates AFC CFD candidate inputs using Latin hypercube sampling.

    Args:
        design_space: Dictionary mapping AFC task input columns to `(lower, upper)` ranges.
        n_samples: Number of samples to generate.
        fixed_values: Optional constants for columns that should not vary.
        random_seed: Reproducible random seed.

    Returns:
        DataFrame containing all canonical AFC task input columns.
    """
    design_space = _validate_design_space(design_space)
    if n_samples < 0:
        raise ValueError("`n_samples` must be non-negative.")

    rng = onp.random.default_rng(random_seed)
    data = {}
    for column, (lower, upper) in design_space.items():
        if n_samples == 0:
            values = onp.array([])
        elif upper == lower:
            values = onp.full(n_samples, lower)
        else:
            values = (onp.arange(n_samples) + rng.random(n_samples)) / n_samples
            rng.shuffle(values)
            values = lower + values * (upper - lower)
        data[column] = values

    return fill_missing_afc_task_inputs(
        pd.DataFrame(data),
        fixed_values=fixed_values,
    )


def local_refinement_sample_afc_design_space(
    optimum: Dict[str, float],
    design_space: Dict[str, Tuple[float, float]],
    n_samples: int,
    *,
    radius_fraction: float = 0.15,
    fixed_values: Optional[Dict[str, float]] = None,
    random_seed: int = 0,
) -> pd.DataFrame:
    """
    Generates a local LHS cloud around a current optimizer solution.

    The local box is clipped to the global `design_space`, so the resulting task list remains feasible.
    """
    design_space = _validate_design_space(design_space)
    if radius_fraction < 0:
        raise ValueError("`radius_fraction` must be non-negative.")

    local_space = {}
    for column, (lower, upper) in design_space.items():
        center = float(optimum.get(column, DEFAULT_AFC_TASK_VALUES[column]))
        span = upper - lower
        local_lower = max(lower, center - radius_fraction * span)
        local_upper = min(upper, center + radius_fraction * span)
        local_space[column] = (local_lower, local_upper)

    return latin_hypercube_sample_afc_design_space(
        local_space,
        n_samples=n_samples,
        fixed_values=fixed_values,
        random_seed=random_seed,
    )


def _kulfan_parameters_from_row(row: pd.Series) -> Dict[str, onp.ndarray]:
    return {
        "upper_weights": onp.array([row[column] for column in KULFAN_UPPER_COLUMNS]),
        "lower_weights": onp.array([row[column] for column in KULFAN_LOWER_COLUMNS]),
        "leading_edge_weight": float(row["leading_edge_weight"]),
        "TE_thickness": float(row["TE_thickness"]),
    }


def _default_afc_model_callable(
    candidates: pd.DataFrame,
    *,
    model_size: str,
    include_360_deg_effects: bool,
) -> pd.DataFrame:
    import aerosandbox as asb

    rows = []
    for _, row in candidates.iterrows():
        airfoil = asb.KulfanAirfoil(
            name="AFC active-learning candidate",
            **_kulfan_parameters_from_row(row),
        )
        aero = airfoil.get_aero_from_afc_neuralfoil(
            alpha=float(row["alpha"]),
            Re=float(row["Re"]),
            mach=float(row["Mach"]),
            C_mu=float(row["C_mu"]),
            x_jet=float(row["x_jet"]),
            theta_jet=float(row["theta_jet"]),
            model_size=model_size,
            include_360_deg_effects=include_360_deg_effects,
        )
        rows.append(
            {
                "CL": float(aero["CL"]),
                "CD": float(aero["CD"]),
                "CM": float(aero["CM"]),
                "Top_Xtr": float(aero["Top_Xtr"]),
                "Bot_Xtr": float(aero["Bot_Xtr"]),
                "analysis_confidence": float(aero["analysis_confidence"]),
            }
        )

    return pd.DataFrame(rows)


def evaluate_afc_active_learning_candidates(
    candidates: pd.DataFrame,
    *,
    model_callable: Optional[Callable[[pd.DataFrame], Union[pd.DataFrame, Dict]]] = None,
    model_size: str = "large",
    include_360_deg_effects: bool = True,
) -> pd.DataFrame:
    """
    Evaluates candidate task points with the current AFC surrogate.

    `model_callable` may be supplied to use a trained AFC-NeuralFoil model, an ensemble, or an external error model.
    It receives a DataFrame with canonical task input columns and should return a DataFrame or dict containing any of:
    `CL`, `CD`, `CM`, `Top_Xtr`, `Bot_Xtr`, and `analysis_confidence`.
    """
    candidates = fill_missing_afc_task_inputs(candidates)

    if model_callable is None:
        outputs = _default_afc_model_callable(
            candidates,
            model_size=model_size,
            include_360_deg_effects=include_360_deg_effects,
        )
    else:
        outputs = model_callable(candidates.copy())
        if isinstance(outputs, dict):
            outputs = pd.DataFrame(outputs)
        elif not isinstance(outputs, pd.DataFrame):
            raise TypeError("`model_callable` must return a pandas DataFrame or dict.")

    outputs = outputs.reset_index(drop=True).copy()
    if len(outputs) != len(candidates):
        raise ValueError(
            "`model_callable` returned a different number of rows than the candidate table."
        )

    defaults = {
        "CL": 0.0,
        "CD": onp.nan,
        "CM": 0.0,
        "Top_Xtr": onp.nan,
        "Bot_Xtr": onp.nan,
        "analysis_confidence": 0.5,
    }
    for column, default in defaults.items():
        if column not in outputs.columns:
            outputs[column] = default
        outputs[column] = pd.to_numeric(outputs[column], errors="coerce")

    if outputs["CD"].isna().any():
        outputs["CD"] = outputs["CD"].fillna(0.02)

    outputs["analysis_confidence"] = outputs["analysis_confidence"].fillna(0.5)
    outputs["analysis_confidence"] = outputs["analysis_confidence"].clip(0.0, 1.0)

    return pd.DataFrame(
        {
            "predicted_CL": outputs["CL"],
            "predicted_CD": outputs["CD"],
            "predicted_CM": outputs["CM"],
            "predicted_Top_Xtr": outputs["Top_Xtr"],
            "predicted_Bot_Xtr": outputs["Bot_Xtr"],
            "predicted_analysis_confidence": outputs["analysis_confidence"],
        }
    )


def _normalize_score(values: Union[pd.Series, onp.ndarray]) -> onp.ndarray:
    values = onp.asarray(values, dtype=float)
    values = onp.where(onp.isfinite(values), values, 0.0)
    lower = onp.nanmin(values) if len(values) > 0 else 0.0
    upper = onp.nanmax(values) if len(values) > 0 else 0.0
    if upper - lower < 1e-12:
        return onp.zeros_like(values)
    return onp.clip((values - lower) / (upper - lower), 0.0, 1.0)


def _training_novelty_score(
    candidates: pd.DataFrame,
    training_data: Optional[pd.DataFrame],
    design_space: Dict[str, Tuple[float, float]],
) -> Tuple[onp.ndarray, onp.ndarray]:
    if training_data is None or len(training_data) == 0:
        return onp.ones(len(candidates)), onp.full(len(candidates), onp.inf)

    columns = [column for column in AFC_TASK_INPUT_COLUMNS if column in training_data.columns]
    if len(columns) == 0:
        return onp.ones(len(candidates)), onp.full(len(candidates), onp.inf)

    training_inputs = fill_missing_afc_task_inputs(training_data[columns])
    candidate_inputs = fill_missing_afc_task_inputs(candidates[columns])

    center = {}
    scale = {}
    combined = pd.concat([training_inputs[columns], candidate_inputs[columns]], axis=0)
    for column in columns:
        if column in design_space:
            lower, upper = design_space[column]
        else:
            lower = float(combined[column].min())
            upper = float(combined[column].max())
        center[column] = 0.5 * (lower + upper)
        scale[column] = max(upper - lower, 1e-12)

    train_matrix = onp.column_stack(
        [(training_inputs[column].to_numpy() - center[column]) / scale[column] for column in columns]
    )
    candidate_matrix = onp.column_stack(
        [(candidate_inputs[column].to_numpy() - center[column]) / scale[column] for column in columns]
    )

    nearest = onp.empty(len(candidate_matrix))
    chunk_size = 256
    for start in range(0, len(candidate_matrix), chunk_size):
        chunk = candidate_matrix[start : start + chunk_size]
        distances = onp.sqrt(
            onp.sum((chunk[:, None, :] - train_matrix[None, :, :]) ** 2, axis=2)
        )
        nearest[start : start + chunk_size] = onp.min(distances, axis=1)

    novelty = onp.clip(nearest / 0.35, 0.0, 1.0)
    return novelty, nearest


def estimate_afc_candidate_gradient_scores(
    candidates: pd.DataFrame,
    *,
    design_space: Dict[str, Tuple[float, float]],
    model_callable: Optional[Callable[[pd.DataFrame], Union[pd.DataFrame, Dict]]] = None,
    model_size: str = "large",
    include_360_deg_effects: bool = True,
    gradient_variables: Sequence[str] = ("alpha", "C_mu"),
) -> onp.ndarray:
    """
    Estimates finite-difference sensitivity of predicted `CL`, `CD`, and `CM`.

    High-gradient regions are good active-learning targets because small input errors can produce large aerodynamic
    errors or optimizer sensitivity.
    """
    candidates = fill_missing_afc_task_inputs(candidates)
    design_space = _validate_design_space(design_space)
    base = evaluate_afc_active_learning_candidates(
        candidates,
        model_callable=model_callable,
        model_size=model_size,
        include_360_deg_effects=include_360_deg_effects,
    )

    gradient_score = onp.zeros(len(candidates))
    for variable in gradient_variables:
        if variable not in design_space:
            continue

        lower, upper = design_space[variable]
        span = upper - lower
        if span <= 0:
            continue

        step = max(1e-4 * span, 1e-6)
        perturbed = candidates.copy()
        perturbed[variable] = onp.clip(
            perturbed[variable].to_numpy(dtype=float) + step,
            lower,
            upper,
        )

        actual_step = perturbed[variable].to_numpy(dtype=float) - candidates[
            variable
        ].to_numpy(dtype=float)
        valid = onp.abs(actual_step) > 1e-14
        if not onp.any(valid):
            continue

        perturbed_outputs = evaluate_afc_active_learning_candidates(
            perturbed,
            model_callable=model_callable,
            model_size=model_size,
            include_360_deg_effects=include_360_deg_effects,
        )

        dCL = onp.zeros(len(candidates))
        dCD = onp.zeros(len(candidates))
        dCM = onp.zeros(len(candidates))
        dCL[valid] = (
            perturbed_outputs["predicted_CL"].to_numpy()[valid]
            - base["predicted_CL"].to_numpy()[valid]
        ) / (actual_step[valid] / span)
        dCD[valid] = (
            perturbed_outputs["predicted_CD"].to_numpy()[valid]
            - base["predicted_CD"].to_numpy()[valid]
        ) / (actual_step[valid] / span)
        dCM[valid] = (
            perturbed_outputs["predicted_CM"].to_numpy()[valid]
            - base["predicted_CM"].to_numpy()[valid]
        ) / (actual_step[valid] / span)

        gradient_score += dCL**2 + (20 * dCD) ** 2 + dCM**2

    return _normalize_score(onp.sqrt(gradient_score))


def _optimization_score(
    candidates: pd.DataFrame,
    optimum: Optional[Dict[str, float]],
    design_space: Dict[str, Tuple[float, float]],
) -> onp.ndarray:
    if optimum is None:
        return onp.zeros(len(candidates))

    columns = [column for column in design_space if column in optimum]
    if len(columns) == 0:
        return onp.zeros(len(candidates))

    distance_squared = onp.zeros(len(candidates))
    for column in columns:
        lower, upper = design_space[column]
        scale = max(upper - lower, 1e-12)
        distance_squared += (
            (candidates[column].to_numpy(dtype=float) - float(optimum[column])) / scale
        ) ** 2

    distance = onp.sqrt(distance_squared / max(len(columns), 1))
    return onp.exp(-(distance / 0.18) ** 2)


def _stall_score(candidates: pd.DataFrame, predictions: pd.DataFrame) -> onp.ndarray:
    alpha = candidates["alpha"].to_numpy(dtype=float)
    CL = predictions["predicted_CL"].to_numpy(dtype=float)
    xtr_top = predictions["predicted_Top_Xtr"].to_numpy(dtype=float)

    alpha_score = 1 / (1 + onp.exp(-(onp.abs(alpha) - 11.0) / 1.8))
    cl_score = 1 / (1 + onp.exp(-(onp.abs(CL) - 1.15) / 0.15))
    transition_score = onp.where(
        onp.isfinite(xtr_top),
        1 / (1 + onp.exp((xtr_top - 0.08) / 0.04)),
        0.0,
    )

    return onp.clip(0.55 * alpha_score + 0.35 * cl_score + 0.10 * transition_score, 0.0, 1.0)


def _compose_reason(row: pd.Series) -> str:
    scores = {
        "低confidence": row["score_uncertainty"],
        "高梯度": row["score_gradient"],
        "优化局部加密": row["score_optimization"],
        "失速附近": row["score_stall"],
        "OOD稀疏区域": row["score_novelty"],
    }

    reasons = [
        reason
        for reason, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)
        if score >= 0.55
    ]
    if len(reasons) == 0:
        reasons = [max(scores.items(), key=lambda item: item[1])[0]]

    return " / ".join(reasons[:3])


def _assign_task_metadata(
    tasks: pd.DataFrame,
    *,
    source: str,
    sample_id_prefix: str,
) -> pd.DataFrame:
    tasks = tasks.copy().reset_index(drop=True)
    tasks["sample_id"] = [
        f"{sample_id_prefix}_{i:05d}" for i in range(len(tasks))
    ]
    tasks["source"] = source
    tasks["reason"] = tasks.apply(_compose_reason, axis=1)
    return tasks


def recommend_afc_cfd_samples(
    training_data: Optional[pd.DataFrame],
    design_space: Dict[str, Tuple[float, float]],
    *,
    n_candidates: int = 256,
    n_recommend: int = 24,
    n_initial_lhs: int = 0,
    optimum: Optional[Dict[str, float]] = None,
    local_refinement_fraction: float = 0.35,
    local_radius_fraction: float = 0.15,
    fixed_values: Optional[Dict[str, float]] = None,
    model_callable: Optional[Callable[[pd.DataFrame], Union[pd.DataFrame, Dict]]] = None,
    model_size: str = "large",
    include_360_deg_effects: bool = True,
    gradient_variables: Sequence[str] = ("alpha", "C_mu"),
    score_weights: Optional[Dict[str, float]] = None,
    random_seed: int = 0,
    output_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Recommends the next AFC CFD samples for an active-learning loop.

    The function does not run CFD. It only creates a ranked task list from the current training data, surrogate
    confidence, finite-difference model gradients, optional optimizer-local refinement, and a simple stall proxy.

    Args:
        training_data: Current labeled AFC dataset, or `None` for an initial campaign.
        design_space: Candidate ranges for AFC task input columns.
        n_candidates: Total candidate-pool size before ranking.
        n_recommend: Number of task-list rows to return.
        n_initial_lhs: Number of pure LHS initial samples to reserve at the top of the returned batch.
        optimum: Optional current optimization solution for local refinement.
        local_refinement_fraction: Fraction of active candidates sampled near `optimum`.
        local_radius_fraction: Local refinement box half-width as a fraction of each global design-space range.
        fixed_values: Constants for any task input columns that should not vary.
        model_callable: Optional model/error callable. If omitted, uses `get_aero_from_afc_neuralfoil()` row by row.
        model_size: Baseline AFC/NeuralFoil model size used by the default model callable.
        include_360_deg_effects: Passed to the default model callable.
        gradient_variables: Input columns used for finite-difference high-gradient scoring.
        score_weights: Optional weights for uncertainty, gradient, optimization, stall, and novelty scores.
        random_seed: Reproducible random seed.
        output_path: Optional `.csv` or `.parquet` path to write the recommended task list.

    Returns:
        Ranked task list containing canonical inputs, prediction metadata, scores, and recommendation reasons.
    """
    design_space = _validate_design_space(design_space)
    if n_candidates < 0 or n_recommend < 0 or n_initial_lhs < 0:
        raise ValueError("Candidate and recommendation counts must be non-negative.")
    if not 0 <= local_refinement_fraction <= 1:
        raise ValueError("`local_refinement_fraction` must be in [0, 1].")

    score_weights = {
        **AFC_ACTIVE_LEARNING_DEFAULT_WEIGHTS,
        **({} if score_weights is None else score_weights),
    }
    weight_sum = sum(max(float(v), 0.0) for v in score_weights.values())
    if weight_sum <= 0:
        raise ValueError("At least one active-learning score weight must be positive.")
    score_weights = {key: max(float(value), 0.0) / weight_sum for key, value in score_weights.items()}

    n_initial_lhs = min(n_initial_lhs, n_recommend)
    n_active_recommend = n_recommend - n_initial_lhs

    initial_tasks = pd.DataFrame()
    if n_initial_lhs > 0:
        initial_inputs = latin_hypercube_sample_afc_design_space(
            design_space,
            n_samples=n_initial_lhs,
            fixed_values=fixed_values,
            random_seed=random_seed + 1001,
        )
        initial_predictions = evaluate_afc_active_learning_candidates(
            initial_inputs,
            model_callable=model_callable,
            model_size=model_size,
            include_360_deg_effects=include_360_deg_effects,
        )
        novelty, nearest = _training_novelty_score(
            initial_inputs,
            training_data,
            design_space,
        )
        initial_tasks = pd.concat([initial_inputs, initial_predictions], axis=1)
        initial_tasks["nearest_training_distance"] = nearest
        initial_tasks["score_uncertainty"] = 1 - initial_tasks[
            "predicted_analysis_confidence"
        ]
        initial_tasks["score_gradient"] = 0.0
        initial_tasks["score_optimization"] = 0.0
        initial_tasks["score_stall"] = _stall_score(initial_inputs, initial_predictions)
        initial_tasks["score_novelty"] = novelty
        initial_tasks["priority_score"] = 1.0
        initial_tasks = _assign_task_metadata(
            initial_tasks,
            source="lhs_initial",
            sample_id_prefix="afc_lhs",
        )
        initial_tasks["reason"] = "LHS初始采样"

    if n_active_recommend > 0 and n_candidates > 0:
        n_local_candidates = (
            int(onp.round(n_candidates * local_refinement_fraction))
            if optimum is not None
            else 0
        )
        n_global_candidates = n_candidates - n_local_candidates

        global_candidates = latin_hypercube_sample_afc_design_space(
            design_space,
            n_samples=n_global_candidates,
            fixed_values=fixed_values,
            random_seed=random_seed,
        )
        if n_local_candidates > 0:
            local_candidates = local_refinement_sample_afc_design_space(
                optimum=optimum,
                design_space=design_space,
                n_samples=n_local_candidates,
                radius_fraction=local_radius_fraction,
                fixed_values=fixed_values,
                random_seed=random_seed + 37,
            )
            candidates = pd.concat([global_candidates, local_candidates], ignore_index=True)
        else:
            candidates = global_candidates

        predictions = evaluate_afc_active_learning_candidates(
            candidates,
            model_callable=model_callable,
            model_size=model_size,
            include_360_deg_effects=include_360_deg_effects,
        )
        novelty, nearest = _training_novelty_score(
            candidates,
            training_data,
            design_space,
        )

        scored = pd.concat([candidates, predictions], axis=1)
        scored["nearest_training_distance"] = nearest
        scored["score_uncertainty"] = onp.clip(
            1 - scored["predicted_analysis_confidence"].to_numpy(dtype=float),
            0.0,
            1.0,
        )
        scored["score_gradient"] = estimate_afc_candidate_gradient_scores(
            candidates,
            design_space=design_space,
            model_callable=model_callable,
            model_size=model_size,
            include_360_deg_effects=include_360_deg_effects,
            gradient_variables=gradient_variables,
        )
        scored["score_optimization"] = _optimization_score(
            candidates,
            optimum=optimum,
            design_space=design_space,
        )
        scored["score_stall"] = _stall_score(candidates, predictions)
        scored["score_novelty"] = novelty

        scored["priority_score"] = (
            score_weights["uncertainty"] * scored["score_uncertainty"]
            + score_weights["gradient"] * scored["score_gradient"]
            + score_weights["optimization"] * scored["score_optimization"]
            + score_weights["stall"] * scored["score_stall"]
            + score_weights["novelty"] * scored["score_novelty"]
        )

        active_tasks = scored.sort_values(
            "priority_score",
            ascending=False,
        ).head(n_active_recommend)
        active_tasks = _assign_task_metadata(
            active_tasks,
            source="active_learning",
            sample_id_prefix="afc_al",
        )
    else:
        active_tasks = pd.DataFrame()

    tasks = pd.concat([initial_tasks, active_tasks], ignore_index=True)
    tasks = tasks[AFC_ACTIVE_LEARNING_TASK_COLUMNS]

    if output_path is not None:
        write_afc_active_learning_tasks(tasks, output_path)

    return tasks


def _json_default(value):
    if isinstance(value, onp.ndarray):
        return value.tolist()
    if isinstance(value, (onp.integer,)):
        return int(value)
    if isinstance(value, (onp.floating,)):
        return float(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def _ensure_json_string(value: Union[str, Dict, Sequence, None]) -> str:
    if value is None:
        return "{}"
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else "{}"
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=_json_default)


def build_afc_cfd_task_list(
    recommendations: pd.DataFrame,
    *,
    solver_type: str = DEFAULT_AFC_CFD_TASK_CONFIG["solver_type"],
    mesh_config: Optional[Union[str, Dict, Sequence]] = None,
    boundary_condition_config: Optional[Union[str, Dict, Sequence]] = None,
    task_id_prefix: str = "afc_cfd",
) -> pd.DataFrame:
    """
    Converts an active-learning recommendation table into a CFD execution task list.

    This schema is intentionally distinct from the AFC training-data schema:
    - Training data stores inputs plus CFD/experiment labels such as `CL`, `CD`, `CM`.
    - CFD task lists store only the inputs and execution metadata needed by a future SU2/RANS pipeline.
    """
    if not isinstance(recommendations, pd.DataFrame):
        raise TypeError("`recommendations` must be a pandas DataFrame.")

    missing = [
        column for column in AFC_ACTIVE_LEARNING_TASK_COLUMNS if column not in recommendations.columns
    ]
    if missing:
        raise ValueError(
            "Active-learning recommendation table is missing required columns: "
            f"{missing}"
        )

    recommendations = recommendations.copy().reset_index(drop=True)
    recommendations = recommendations.sort_values(
        "priority_score",
        ascending=False,
        kind="mergesort",
    ).reset_index(drop=True)

    task_ids = []
    for i, row in recommendations.iterrows():
        sample_id = str(row.get("sample_id", "")).strip()
        task_ids.append(sample_id if sample_id else f"{task_id_prefix}_{i:05d}")

    mesh_config_string = _ensure_json_string(
        mesh_config if mesh_config is not None else DEFAULT_AFC_CFD_TASK_CONFIG["mesh_config"]
    )
    boundary_condition_string = _ensure_json_string(
        boundary_condition_config
        if boundary_condition_config is not None
        else DEFAULT_AFC_CFD_TASK_CONFIG["boundary_condition_config"]
    )

    tasks = pd.DataFrame(
        {
            "task_id": task_ids,
            "airfoil": [f"{task_id}_kulfan" for task_id in task_ids],
            "solver_type": str(solver_type),
            "mesh_config": mesh_config_string,
            "boundary_condition_config": boundary_condition_string,
            "priority": onp.arange(1, len(recommendations) + 1, dtype=int),
            "priority_score": recommendations["priority_score"].to_numpy(dtype=float),
            "reason": recommendations["reason"].astype(str).to_numpy(),
            "selection_source": recommendations["source"].astype(str).to_numpy(),
        }
    )
    for column in GEOMETRY_COLUMNS:
        tasks[column] = recommendations[column].to_numpy(dtype=float)
    for column in AFC_CFD_TASK_CONDITION_COLUMNS:
        tasks[column] = recommendations[column].to_numpy(dtype=float)

    return tasks[AFC_CFD_TASK_COLUMNS]


def summarize_afc_cfd_task_reasons(tasks: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a simple reason-count table for a CFD task batch.
    """
    if not isinstance(tasks, pd.DataFrame):
        raise TypeError("`tasks` must be a pandas DataFrame.")
    if "reason" not in tasks.columns:
        raise ValueError('CFD task table must contain a "reason" column.')

    counts = {}
    for reason_text in tasks["reason"].astype(str):
        for reason in [part.strip() for part in reason_text.split("/")]:
            if reason:
                counts[reason] = counts.get(reason, 0) + 1

    summary = (
        pd.DataFrame(
            {"reason": list(counts.keys()), "count": list(counts.values())}
        )
        .sort_values(["count", "reason"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return summary


def recommend_afc_cfd_tasks(
    training_data: Optional[pd.DataFrame],
    design_space: Dict[str, Tuple[float, float]],
    *,
    n_candidates: int = 256,
    n_recommend: int = 24,
    n_initial_lhs: int = 0,
    optimum: Optional[Dict[str, float]] = None,
    local_refinement_fraction: float = 0.35,
    local_radius_fraction: float = 0.15,
    fixed_values: Optional[Dict[str, float]] = None,
    model_callable: Optional[Callable[[pd.DataFrame], Union[pd.DataFrame, Dict]]] = None,
    model_size: str = "large",
    include_360_deg_effects: bool = True,
    gradient_variables: Sequence[str] = ("alpha", "C_mu"),
    score_weights: Optional[Dict[str, float]] = None,
    random_seed: int = 0,
    solver_type: str = DEFAULT_AFC_CFD_TASK_CONFIG["solver_type"],
    mesh_config: Optional[Union[str, Dict, Sequence]] = None,
    boundary_condition_config: Optional[Union[str, Dict, Sequence]] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    High-level active-learning entry point that returns a CFD-ready task list.

    Unlike `recommend_afc_cfd_samples()`, this function returns the operational CFD task schema instead of the
    intermediate scored recommendation table.
    """
    recommendations = recommend_afc_cfd_samples(
        training_data,
        design_space,
        n_candidates=n_candidates,
        n_recommend=n_recommend,
        n_initial_lhs=n_initial_lhs,
        optimum=optimum,
        local_refinement_fraction=local_refinement_fraction,
        local_radius_fraction=local_radius_fraction,
        fixed_values=fixed_values,
        model_callable=model_callable,
        model_size=model_size,
        include_360_deg_effects=include_360_deg_effects,
        gradient_variables=gradient_variables,
        score_weights=score_weights,
        random_seed=random_seed,
        output_path=None,
    )
    tasks = build_afc_cfd_task_list(
        recommendations,
        solver_type=solver_type,
        mesh_config=mesh_config,
        boundary_condition_config=boundary_condition_config,
    )
    if output_path is not None:
        write_afc_cfd_tasks(tasks, output_path)
    return tasks


def write_afc_active_learning_tasks(
    tasks: pd.DataFrame,
    filepath: Union[str, Path],
    *,
    index: bool = False,
) -> Path:
    """
    Writes an AFC active-learning recommendation table to `.csv`, `.parquet`, or `.jsonl`.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    missing = [
        column for column in AFC_ACTIVE_LEARNING_TASK_COLUMNS if column not in tasks.columns
    ]
    if missing:
        raise ValueError(f"AFC active-learning task list is missing columns: {missing}")

    tasks = tasks[AFC_ACTIVE_LEARNING_TASK_COLUMNS]
    suffix = filepath.suffix.lower()
    if suffix == ".csv":
        tasks.to_csv(filepath, index=index)
    elif suffix == ".parquet":
        try:
            tasks.to_parquet(filepath, index=index)
        except ImportError as e:
            raise ImportError(
                "Writing Parquet AFC task lists requires a pandas Parquet engine such as `pyarrow` or `fastparquet`."
            ) from e
    elif suffix == ".jsonl":
        tasks.to_json(filepath, orient="records", lines=True, force_ascii=False)
    else:
        raise ValueError(
            f'Unsupported AFC task-list extension "{filepath.suffix}". Use `.csv`, `.parquet`, or `.jsonl`.'
        )

    return filepath


def read_afc_active_learning_tasks(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Reads an AFC active-learning recommendation table from `.csv`, `.parquet`, or `.jsonl`.
    """
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()
    if suffix == ".csv":
        tasks = pd.read_csv(filepath)
    elif suffix == ".parquet":
        try:
            tasks = pd.read_parquet(filepath)
        except ImportError as e:
            raise ImportError(
                "Reading Parquet AFC task lists requires a pandas Parquet engine such as `pyarrow` or `fastparquet`."
            ) from e
    elif suffix == ".jsonl":
        tasks = pd.read_json(filepath, orient="records", lines=True)
    else:
        raise ValueError(
            f'Unsupported AFC task-list extension "{filepath.suffix}". Use `.csv`, `.parquet`, or `.jsonl`.'
        )

    missing = [
        column for column in AFC_ACTIVE_LEARNING_TASK_COLUMNS if column not in tasks.columns
    ]
    if missing:
        raise ValueError(f"AFC active-learning task list is missing columns: {missing}")

    return tasks[AFC_ACTIVE_LEARNING_TASK_COLUMNS]


def write_afc_cfd_tasks(
    tasks: pd.DataFrame,
    filepath: Union[str, Path],
    *,
    index: bool = False,
) -> Path:
    """
    Writes a CFD execution task list to `.csv`, `.parquet`, or `.jsonl`.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    missing = [column for column in AFC_CFD_TASK_COLUMNS if column not in tasks.columns]
    if missing:
        raise ValueError(f"AFC CFD task list is missing columns: {missing}")

    tasks = tasks[AFC_CFD_TASK_COLUMNS].copy()
    for column in AFC_CFD_TASK_CONFIG_COLUMNS:
        tasks[column] = tasks[column].apply(_ensure_json_string)

    suffix = filepath.suffix.lower()
    if suffix == ".csv":
        tasks.to_csv(filepath, index=index)
    elif suffix == ".parquet":
        try:
            tasks.to_parquet(filepath, index=index)
        except ImportError as e:
            raise ImportError(
                "Writing Parquet AFC CFD task lists requires a pandas Parquet engine such as `pyarrow` or `fastparquet`."
            ) from e
    elif suffix == ".jsonl":
        tasks.to_json(filepath, orient="records", lines=True, force_ascii=False)
    else:
        raise ValueError(
            f'Unsupported AFC CFD task-list extension "{filepath.suffix}". Use `.csv`, `.parquet`, or `.jsonl`.'
        )

    return filepath


def read_afc_cfd_tasks(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Reads a CFD execution task list from `.csv`, `.parquet`, or `.jsonl`.
    """
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()
    if suffix == ".csv":
        tasks = pd.read_csv(filepath)
    elif suffix == ".parquet":
        try:
            tasks = pd.read_parquet(filepath)
        except ImportError as e:
            raise ImportError(
                "Reading Parquet AFC CFD task lists requires a pandas Parquet engine such as `pyarrow` or `fastparquet`."
            ) from e
    elif suffix == ".jsonl":
        tasks = pd.read_json(filepath, orient="records", lines=True)
    else:
        raise ValueError(
            f'Unsupported AFC CFD task-list extension "{filepath.suffix}". Use `.csv`, `.parquet`, or `.jsonl`.'
        )

    missing = [column for column in AFC_CFD_TASK_COLUMNS if column not in tasks.columns]
    if missing:
        raise ValueError(f"AFC CFD task list is missing columns: {missing}")

    tasks = tasks[AFC_CFD_TASK_COLUMNS].copy()
    tasks["priority"] = pd.to_numeric(tasks["priority"], errors="raise").astype(int)
    tasks["priority_score"] = pd.to_numeric(
        tasks["priority_score"], errors="raise"
    ).astype(float)
    for column in [*GEOMETRY_COLUMNS, *AFC_CFD_TASK_CONDITION_COLUMNS]:
        tasks[column] = pd.to_numeric(tasks[column], errors="raise").astype(float)
    for column in [
        "task_id",
        "airfoil",
        "solver_type",
        "reason",
        "selection_source",
        *AFC_CFD_TASK_CONFIG_COLUMNS,
    ]:
        tasks[column] = tasks[column].astype(str)

    return tasks
