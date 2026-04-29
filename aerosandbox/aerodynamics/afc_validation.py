from __future__ import annotations

import importlib.util
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Union

import numpy as onp
import pandas as pd

import aerosandbox.numpy as np
from aerosandbox.aerodynamics.aero_2D.afc_dataset import (
    KULFAN_LOWER_COLUMNS,
    KULFAN_UPPER_COLUMNS,
    make_fake_afc_dataset,
    read_afc_dataset,
    validate_afc_dataframe,
)


AFC_BENCHMARK_CASE = {
    "airfoil": "naca0012",
    "alpha_deg": 5.0,
    "velocity_mps": 25.0,
    "mach": 0.05,
    "reynolds": 1e6,
    "spanwise_resolution": 2,
    "model_size": "xsmall",
    "c_mu_sweep": [0.0, 0.01, 0.02],
    "x_jet": 0.1,
    "theta_jet": 30.0,
}

REPO_ROOT = Path(__file__).resolve().parents[2]


def _scalar(value) -> float:
    return float(onp.ravel(onp.asarray(value, dtype=float))[0])


def _array_list(value) -> list:
    return onp.ravel(onp.asarray(value, dtype=float)).tolist()


def _json_default(value):
    if isinstance(value, (onp.integer,)):
        return int(value)
    if isinstance(value, (onp.floating,)):
        return float(value)
    if isinstance(value, onp.ndarray):
        return value.tolist()
    return str(value)


def _resolve_benchmark_case(
    benchmark_case: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    return {
        **AFC_BENCHMARK_CASE,
        **({} if benchmark_case is None else dict(benchmark_case)),
    }


def _git_commit_hash() -> Optional[str]:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=REPO_ROOT,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            .strip()
        )
    except Exception:
        return None


def _runtime_summary(start_time: float) -> Dict[str, float]:
    elapsed = time.time() - start_time
    return {
        "elapsed_seconds": float(elapsed),
    }


def _infer_dataset_manifest_path(dataset_path: Union[str, Path]) -> Optional[Path]:
    dataset_path = Path(dataset_path)
    candidates = [
        dataset_path.with_suffix(".manifest.json"),
        dataset_path.parent / f"{dataset_path.stem}.manifest.json",
        dataset_path.parent / "manifest.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_json_if_exists(filepath: Optional[Union[str, Path]]) -> Optional[Dict[str, object]]:
    if filepath is None:
        return None
    path = Path(filepath)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _rmse(predicted, actual) -> float:
    predicted = onp.asarray(predicted, dtype=float)
    actual = onp.asarray(actual, dtype=float)
    return float(onp.sqrt(onp.mean((predicted - actual) ** 2)))


def _mae(predicted, actual) -> float:
    predicted = onp.asarray(predicted, dtype=float)
    actual = onp.asarray(actual, dtype=float)
    return float(onp.mean(onp.abs(predicted - actual)))


def _spanwise_plot_data(result: Dict[str, object]) -> Dict[str, list]:
    return {
        "spanwise_y": _array_list(result["spanwise_y"]),
        "spanwise_load": _array_list(result["spanwise_load"]),
        "local_C_mu": _array_list(result["local_C_mu"]),
        "section_CL": _array_list(result["section_CL"]),
        "section_CD": _array_list(result["section_CD"]),
        "residual_CL": _array_list(result.get("residual_CL", onp.zeros_like(onp.asarray(result["spanwise_y"], dtype=float)))),
        "residual_CM": _array_list(result.get("residual_CM", onp.zeros_like(onp.asarray(result["spanwise_y"], dtype=float)))),
    }


def make_afc_benchmark_airplane(
    benchmark_case: Optional[Dict[str, object]] = None,
):
    """
    Returns a small symmetric rectangular wing used by AFC regression and validation tests.
    """
    import aerosandbox as asb

    benchmark_case = _resolve_benchmark_case(benchmark_case)

    return asb.Airplane(
        name="AFC Validation Benchmark Wing",
        xyz_ref=[0.25, 0, 0],
        s_ref=4,
        c_ref=1,
        b_ref=4,
        wings=[
            asb.Wing(
                symmetric=True,
                xsecs=[
                    asb.WingXSec(
                        xyz_le=[0, 0, 0],
                        chord=1,
                        airfoil=asb.Airfoil(benchmark_case["airfoil"]),
                    ),
                    asb.WingXSec(
                        xyz_le=[0, 2, 0],
                        chord=1,
                        airfoil=asb.Airfoil(benchmark_case["airfoil"]),
                    ),
                ],
            )
        ],
    )


def run_2d_zero_afc_consistency_check(
    *,
    alphas: Iterable[float] = (-4.0, 0.0, 6.0, 12.0),
    Re: float = AFC_BENCHMARK_CASE["reynolds"],
    mach: float = AFC_BENCHMARK_CASE["mach"],
    model_size: str = AFC_BENCHMARK_CASE["model_size"],
    tolerance: float = 1e-12,
    benchmark_case: Optional[Dict[str, object]] = None,
    weights_directory: Optional[Union[str, Path]] = None,
    training_statistics_path: Optional[Union[str, Path]] = None,
) -> Dict[str, object]:
    """
    Validates that `C_mu = 0` AFC-NeuralFoil returns the same values as baseline NeuralFoil.
    """
    import aerosandbox as asb

    benchmark_case = _resolve_benchmark_case(benchmark_case)
    airfoil = asb.Airfoil(benchmark_case["airfoil"])
    alpha = np.array(list(alphas))
    baseline = airfoil.get_aero_from_neuralfoil(
        alpha=alpha,
        Re=Re,
        mach=mach,
        model_size=model_size,
    )
    afc = airfoil.get_aero_from_afc_neuralfoil(
        alpha=alpha,
        Re=Re,
        mach=mach,
        C_mu=0.0,
        model_size=model_size,
        weights_directory=weights_directory,
        training_statistics_path=training_statistics_path,
    )

    fields = ["CL", "CD", "CM", "Top_Xtr", "Bot_Xtr", "analysis_confidence"]
    max_abs_error = {
        field: float(onp.max(onp.abs(onp.asarray(afc[field]) - onp.asarray(baseline[field]))))
        for field in fields
    }
    return {
        "name": "2D C_mu=0 consistency",
        "passed": all(error <= tolerance for error in max_abs_error.values()),
        "tolerance": tolerance,
        "max_abs_error": max_abs_error,
    }


def load_external_afc_validation_dataset(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Loads an external AFC validation dataset using the canonical schema.

    This is the interface intended for large CFD/RANS/experimental validation data. The data itself should live outside
    the source tree and should not be committed to this repository.
    """
    df = read_afc_dataset(filepath)
    validate_afc_dataframe(df)
    return df


def compute_2d_surrogate_error(
    df: pd.DataFrame,
    *,
    max_cases: Optional[int] = 25,
    model_size: str = AFC_BENCHMARK_CASE["model_size"],
    weights_directory: Optional[Union[str, Path]] = None,
    training_statistics_path: Optional[Union[str, Path]] = None,
) -> Dict[str, object]:
    """
    Computes 2D AFC surrogate errors against a dataset with the canonical AFC schema.
    """
    import aerosandbox as asb

    validate_afc_dataframe(df)
    if max_cases is not None:
        df = df.head(max_cases).reset_index(drop=True)

    predictions = {field: [] for field in ["CL", "CD", "CM", "Top_Xtr", "Bot_Xtr"]}
    for _, row in df.iterrows():
        airfoil = asb.KulfanAirfoil(
            upper_weights=[row[column] for column in KULFAN_UPPER_COLUMNS],
            lower_weights=[row[column] for column in KULFAN_LOWER_COLUMNS],
            leading_edge_weight=row["leading_edge_weight"],
            TE_thickness=row["TE_thickness"],
        )
        aero = airfoil.get_aero_from_afc_neuralfoil(
            alpha=row["alpha"],
            Re=row["Re"],
            mach=row["Mach"],
            C_mu=row["C_mu"],
            x_jet=row["x_jet"],
            theta_jet=row["theta_jet"],
            model_size=model_size,
            include_360_deg_effects=True,
            weights_directory=weights_directory,
            training_statistics_path=training_statistics_path,
        )
        for field in predictions:
            predictions[field].append(_scalar(aero[field]))

    metrics = {}
    for field, predicted in predictions.items():
        actual = df[field].to_numpy(dtype=float)
        metrics[field] = {
            "rmse": _rmse(predicted, actual),
            "mae": _mae(predicted, actual),
            "bias": float(onp.mean(onp.asarray(predicted) - actual)),
        }

    return {
        "name": "2D surrogate dataset error",
        "n_cases": int(len(df)),
        "metrics": metrics,
        "passed": all(onp.isfinite(metric["rmse"]) for metric in metrics.values()),
    }


def compute_2d_polar_comparison(
    df: pd.DataFrame,
    *,
    max_cases: Optional[int] = 40,
    model_size: str = AFC_BENCHMARK_CASE["model_size"],
    weights_directory: Optional[Union[str, Path]] = None,
    training_statistics_path: Optional[Union[str, Path]] = None,
) -> Dict[str, object]:
    """
    Builds compact 2D polar comparison data for reporting.
    """
    import aerosandbox as asb

    validate_afc_dataframe(df)
    if max_cases is not None:
        df = df.head(max_cases).reset_index(drop=True)

    rows = []
    for _, row in df.iterrows():
        airfoil = asb.KulfanAirfoil(
            upper_weights=[row[column] for column in KULFAN_UPPER_COLUMNS],
            lower_weights=[row[column] for column in KULFAN_LOWER_COLUMNS],
            leading_edge_weight=row["leading_edge_weight"],
            TE_thickness=row["TE_thickness"],
        )
        aero = airfoil.get_aero_from_afc_neuralfoil(
            alpha=row["alpha"],
            Re=row["Re"],
            mach=row["Mach"],
            C_mu=row["C_mu"],
            x_jet=row["x_jet"],
            theta_jet=row["theta_jet"],
            model_size=model_size,
            include_360_deg_effects=True,
            weights_directory=weights_directory,
            training_statistics_path=training_statistics_path,
        )
        rows.append(
            {
                "alpha": float(row["alpha"]),
                "C_mu": float(row["C_mu"]),
                "actual_CL": float(row["CL"]),
                "actual_CD": float(row["CD"]),
                "predicted_CL": _scalar(aero["CL"]),
                "predicted_CD": _scalar(aero["CD"]),
            }
        )

    comparison = pd.DataFrame(rows)
    if len(comparison) == 0:
        slices = []
    else:
        c_mu_levels = (
            comparison["C_mu"]
            .drop_duplicates()
            .sort_values()
            .tolist()
        )
        if len(c_mu_levels) > 3:
            indices = onp.linspace(0, len(c_mu_levels) - 1, 3, dtype=int)
            c_mu_levels = [c_mu_levels[i] for i in indices]
        slices = []
        for c_mu in c_mu_levels:
            subset = comparison[onp.isclose(comparison["C_mu"], c_mu)].sort_values("alpha")
            if len(subset) == 0:
                continue
            slices.append(
                {
                    "C_mu": float(c_mu),
                    "alpha": subset["alpha"].tolist(),
                    "actual_CL": subset["actual_CL"].tolist(),
                    "predicted_CL": subset["predicted_CL"].tolist(),
                    "actual_CD": subset["actual_CD"].tolist(),
                    "predicted_CD": subset["predicted_CD"].tolist(),
                }
            )

    return {
        "name": "2D polar comparison",
        "n_cases": int(len(comparison)),
        "polar_slices": slices,
        "sample_rows": comparison.to_dict(orient="records"),
        "passed": len(slices) > 0 and all(len(slice_["alpha"]) > 0 for slice_ in slices),
    }


def run_3d_no_afc_baseline_check(
    *,
    spanwise_resolution: int = AFC_BENCHMARK_CASE["spanwise_resolution"],
    model_size: str = AFC_BENCHMARK_CASE["model_size"],
    include_avl: bool = False,
    benchmark_case: Optional[Dict[str, object]] = None,
    weights_directory: Optional[Union[str, Path]] = None,
    training_statistics_path: Optional[Union[str, Path]] = None,
) -> Dict[str, object]:
    """
    Compares no-AFC 3D results against AeroSandbox's baseline LiftingLine and VLM, optionally AVL.
    """
    import aerosandbox as asb

    benchmark_case = _resolve_benchmark_case(benchmark_case)
    airplane = make_afc_benchmark_airplane(benchmark_case)
    op_point = asb.OperatingPoint(
        velocity=benchmark_case["velocity_mps"],
        alpha=benchmark_case["alpha_deg"],
    )
    baseline = asb.LiftingLine(
        airplane=airplane,
        op_point=op_point,
        spanwise_resolution=spanwise_resolution,
        model_size=model_size,
    ).run()
    afc = asb.AFCNeuralFoilLiftingLine(
        airplane=airplane,
        op_point=op_point,
        C_mu=0.0,
        spanwise_resolution=spanwise_resolution,
        model_size=model_size,
        weights_directory=weights_directory,
        training_statistics_path=training_statistics_path,
    ).run()
    vlm = asb.VortexLatticeMethod(
        airplane=airplane,
        op_point=op_point,
        spanwise_resolution=spanwise_resolution,
        chordwise_resolution=2,
    ).run()

    fields = ["CL", "CD", "Cm"]
    errors = {
        field: abs(_scalar(afc[field]) - _scalar(baseline[field]))
        for field in fields
    }
    comparison_fields = ["CL", "CD", "CY", "Cl", "Cm", "Cn"]
    comparison_table = []
    for solver_name, result in [
        ("AeroSandbox LiftingLine", baseline),
        ("AFCNeuralFoilLiftingLine C_mu=0", afc),
        ("AeroSandbox VLM inviscid", vlm),
    ]:
        row = {"solver": solver_name}
        for field in comparison_fields:
            row[field] = _scalar(result[field]) if field in result else None
        comparison_table.append(row)

    report = {
        "name": "3D no-AFC baseline comparison with VLM",
        "baseline": {field: _scalar(baseline[field]) for field in fields},
        "afc_zero": {field: _scalar(afc[field]) for field in fields},
        "vlm": {
            field: _scalar(vlm[field])
            for field in comparison_fields
            if field in vlm
        },
        "comparison_table": comparison_table,
        "abs_error": errors,
        "passed": all(error < 1e-10 for error in errors.values())
        and all(onp.isfinite(row["CL"]) for row in comparison_table),
    }

    if include_avl:
        try:
            avl = asb.AVL(airplane=airplane, op_point=op_point).run()
            report["avl"] = {field: _scalar(avl[field]) for field in fields if field in avl}
        except Exception as e:
            report["avl_error"] = f"{type(e).__name__}: {e}"

    return report


def run_3d_afc_monotonicity_check(
    *,
    c_mu_values: Tuple[float, ...] = tuple(AFC_BENCHMARK_CASE["c_mu_sweep"]),
    spanwise_resolution: int = AFC_BENCHMARK_CASE["spanwise_resolution"],
    model_size: str = AFC_BENCHMARK_CASE["model_size"],
    benchmark_case: Optional[Dict[str, object]] = None,
    weights_directory: Optional[Union[str, Path]] = None,
    training_statistics_path: Optional[Union[str, Path]] = None,
) -> Dict[str, object]:
    """
    Checks that CL increases reasonably with C_mu and that CD/CM vary continuously for the benchmark wing.
    """
    import aerosandbox as asb

    benchmark_case = _resolve_benchmark_case(benchmark_case)
    airplane = make_afc_benchmark_airplane(benchmark_case)
    op_point = asb.OperatingPoint(
        velocity=benchmark_case["velocity_mps"],
        alpha=benchmark_case["alpha_deg"],
    )

    cases = []
    final_aero = None
    for C_mu in c_mu_values:
        aero = asb.AFCNeuralFoilLiftingLine(
            airplane=airplane,
            op_point=op_point,
            C_mu=C_mu,
            spanwise_resolution=spanwise_resolution,
            model_size=model_size,
            weights_directory=weights_directory,
            training_statistics_path=training_statistics_path,
        ).run()
        final_aero = aero
        cases.append(
            {
                "C_mu": float(C_mu),
                "CL": _scalar(aero["CL"]),
                "CD": _scalar(aero["CD"]),
                "CM": _scalar(aero["CM"]),
                "min_confidence": float(onp.min(onp.asarray(aero["analysis_confidence"], dtype=float))),
            }
        )

    CL = onp.array([case["CL"] for case in cases])
    CD = onp.array([case["CD"] for case in cases])
    CM = onp.array([case["CM"] for case in cases])
    dCL = onp.diff(CL)
    dCD = onp.diff(CD)
    dCM = onp.diff(CM)

    return {
        "name": "3D AFC monotonicity and continuity",
        "cases": cases,
        "dCL": dCL.tolist(),
        "dCD": dCD.tolist(),
        "dCM": dCM.tolist(),
        "spanwise_plot_data": _spanwise_plot_data(final_aero),
        "passed": bool(
            onp.all(dCL > -1e-6)
            and onp.all(onp.isfinite(dCD))
            and onp.all(onp.isfinite(dCM))
            and onp.max(onp.abs(dCD)) < 0.1
            and onp.max(onp.abs(dCM)) < 0.2
        ),
    }


def run_3d_cambered_vlm_diagnostic_check(
    *,
    spanwise_resolution: int = AFC_BENCHMARK_CASE["spanwise_resolution"],
    model_size: str = AFC_BENCHMARK_CASE["model_size"],
    benchmark_case: Optional[Dict[str, object]] = None,
    weights_directory: Optional[Union[str, Path]] = None,
    training_statistics_path: Optional[Union[str, Path]] = None,
) -> Dict[str, object]:
    """
    Runs a compact NeuralFoilCamberedVLM case and reports convergence diagnostics for validation plots.
    """
    import aerosandbox as asb

    benchmark_case = _resolve_benchmark_case(benchmark_case)
    airplane = make_afc_benchmark_airplane(benchmark_case)
    op_point = asb.OperatingPoint(
        velocity=benchmark_case["velocity_mps"],
        alpha=benchmark_case["alpha_deg"],
    )
    result = asb.NeuralFoilCamberedVLM(
        airplane=airplane,
        op_point=op_point,
        C_mu=0.01,
        spanwise_resolution=spanwise_resolution,
        chordwise_resolution=3,
        model_size=model_size,
        match_CM=False,
        max_iterations=8,
        weights_directory=weights_directory,
        training_statistics_path=training_statistics_path,
    ).run()

    required_fields = [
        "CL",
        "CD",
        "CDi",
        "CDp",
        "CM",
        "CY",
        "Cl",
        "Cn",
        "spanwise_y",
        "section_cl",
        "section_cd",
        "section_cm",
        "local_alpha",
        "local_Re",
        "local_C_mu",
        "residual_CL",
        "residual_CM",
        "converged",
        "failure_reason",
        "convergence_history",
        "iteration_count",
        "max_residual",
    ]
    missing_fields = [field for field in required_fields if field not in result]
    history = result["convergence_history"]
    initial_residual = history[0]["max_abs_residual_CL"]
    final_residual = history[-1]["max_abs_residual_CL"]

    return {
        "name": "3D NeuralFoilCamberedVLM convergence diagnostics",
        "missing_fields": missing_fields,
        "converged": bool(result["converged"]),
        "failure_reason": result["failure_reason"],
        "iteration_count": int(result["iteration_count"]),
        "max_residual": float(result["max_residual"]),
        "initial_max_abs_residual_CL": float(initial_residual),
        "final_max_abs_residual_CL": float(final_residual),
        "convergence_history": history,
        "spanwise_plot_data": _spanwise_plot_data(result),
        "passed": len(missing_fields) == 0
        and onp.isfinite(float(result["CL"]))
        and final_residual <= initial_residual,
    }


def run_opti_convergence_check(
    *,
    target_CL: float = 0.95,
    model_size: str = AFC_BENCHMARK_CASE["model_size"],
    benchmark_case: Optional[Dict[str, object]] = None,
    weights_directory: Optional[Union[str, Path]] = None,
    training_statistics_path: Optional[Union[str, Path]] = None,
) -> Dict[str, object]:
    """
    Solves a small differentiable AFC optimization problem.
    """
    import aerosandbox as asb

    opti = asb.Opti()
    benchmark_case = _resolve_benchmark_case(benchmark_case)
    C_mu = opti.variable(init_guess=0.02, lower_bound=0.0, upper_bound=0.1)
    aero = asb.Airfoil(benchmark_case["airfoil"]).get_aero_from_afc_neuralfoil(
        alpha=benchmark_case["alpha_deg"],
        Re=benchmark_case["reynolds"],
        mach=benchmark_case["mach"],
        C_mu=C_mu,
        x_jet=benchmark_case["x_jet"],
        theta_jet=benchmark_case["theta_jet"],
        model_size=model_size,
        weights_directory=weights_directory,
        training_statistics_path=training_statistics_path,
    )
    opti.subject_to(aero["CL"] >= target_CL)
    opti.minimize(aero["CD"] + 0.02 * C_mu**2)
    sol = opti.solve(verbose=False)

    return {
        "name": "Opti convergence",
        "target_CL": target_CL,
        "C_mu": _scalar(sol(C_mu)),
        "CL": _scalar(sol(aero["CL"])),
        "CD": _scalar(sol(aero["CD"])),
        "passed": bool(_scalar(sol(aero["CL"])) >= target_CL - 1e-6),
    }


def _make_validation_swap_network():
    import aerosandbox as asb

    return asb.AFCSystemNetwork(
        components=[
            asb.AFCInlet(name="inlet", area=0.02),
            asb.AFCCompressor(name="compressor", pressure_ratio=1.8),
            asb.AFCDuct(name="duct", length=1.5, hydraulic_diameter=0.05),
            asb.AFCValve(name="valve", hydraulic_diameter=0.05),
            asb.AFCPlenum(name="plenum", volume=0.01, reference_area=0.01),
            asb.AFCSteadyJetActuator(
                name="actuator",
                jet_area=0.002,
                jet_velocity_ratio=2.0,
            ),
        ]
    )


def run_swap_trend_validation(
    *,
    c_mu_values: Tuple[float, ...] = (0.01, 0.03, 0.05),
    freestream_velocity: float = 50.0,
    density_inf: float = 1.225,
    S_ref: float = 10.0,
) -> Dict[str, object]:
    """
    Checks that the placeholder SWaP model responds monotonically to increasing target C_mu.
    """
    network = _make_validation_swap_network()
    cases = []
    for c_mu in c_mu_values:
        result = network.design(
            freestream_velocity=freestream_velocity,
            density_inf=density_inf,
            S_ref=S_ref,
            target_C_mu=c_mu,
        )
        cases.append(
            {
                "C_mu": float(c_mu),
                "mass_flow": float(result["mass_flow"]),
                "required_power": float(result["required_power"]),
                "estimated_weight": float(result["estimated_weight"]),
                "estimated_volume": float(result["estimated_volume"]),
            }
        )

    mass_flow = onp.array([case["mass_flow"] for case in cases], dtype=float)
    power = onp.array([case["required_power"] for case in cases], dtype=float)

    return {
        "name": "SWaP trend validation",
        "model_kind": "placeholder",
        "cases": cases,
        "passed": bool(onp.all(onp.diff(mass_flow) > 0) and onp.all(onp.diff(power) > 0)),
        "notes": [
            "This section validates monotonic engineering trends only.",
            "The AFC system component correlations are placeholder models, not calibrated compressor-map or hardware data.",
        ],
    }


def _load_concept_design_example_module():
    module_path = REPO_ROOT / "examples" / "afc_concept_design_optimization.py"
    spec = importlib.util.spec_from_file_location("afc_concept_design_optimization", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load concept design example from {module_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_end_to_end_optimization_validation(
    *,
    output_directory: Union[str, Path],
) -> Dict[str, object]:
    """
    Runs the compact end-to-end concept design example and captures its before/after summary.
    """
    output_directory = Path(output_directory)
    module = _load_concept_design_example_module()
    summary = module.run_example(output_directory=output_directory)

    optimized = summary.get("optimized")
    initial = summary["initial"]
    objective_improved = (
        optimized is not None and float(optimized["objective"]) <= float(initial["objective"]) + 1e-9
    )
    return {
        "name": "End-to-end optimization before/after",
        "status": summary.get("status"),
        "failure_reason": summary.get("failure_reason"),
        "initial": summary.get("initial"),
        "optimized": optimized,
        "assumptions": summary.get("assumptions", []),
        "outputs": summary.get("outputs", {}),
        "passed": bool(summary.get("status") == "solved" and objective_improved),
    }


def run_afc_validation_suite(
    *,
    external_2d_dataset: Optional[Union[str, Path]] = None,
    synthetic_2d_cases: int = 12,
    include_avl: bool = False,
) -> Dict[str, object]:
    """
    Runs the lightweight AFC validation suite.
    """
    if external_2d_dataset is None:
        df_2d = make_fake_afc_dataset(n_cases=synthetic_2d_cases, random_seed=19)
        dataset_source = "synthetic"
    else:
        df_2d = load_external_afc_validation_dataset(external_2d_dataset)
        dataset_source = str(external_2d_dataset)

    checks = [
        run_2d_zero_afc_consistency_check(),
        compute_2d_surrogate_error(df_2d),
        run_3d_no_afc_baseline_check(include_avl=include_avl),
        run_3d_afc_monotonicity_check(),
        run_3d_cambered_vlm_diagnostic_check(),
        run_opti_convergence_check(),
    ]

    return {
        "benchmark_case": AFC_BENCHMARK_CASE,
        "dataset_source": dataset_source,
        "checks": checks,
        "passed": all(check.get("passed", False) for check in checks),
    }


def generate_afc_validation_acceptance_report(
    config: Dict[str, object],
    *,
    config_path: Optional[Union[str, Path]] = None,
    output_directory: Optional[Union[str, Path]] = None,
) -> Dict[str, object]:
    """
    Builds the formal AFC validation/acceptance report payload from a validation config.
    """
    start_time = time.time()

    benchmark_case = _resolve_benchmark_case(config.get("benchmark", {}))
    checks_config = dict(config.get("checks", {}))
    thresholds = dict(config.get("thresholds", {}))
    external_data = dict(config.get("external_data", {}))
    report_config = dict(config.get("report", {}))
    model_config = dict(config.get("model", {}))

    external_2d_dataset = external_data.get("external_2d_dataset")
    external_2d_manifest = external_data.get("external_2d_manifest")
    synthetic_2d_cases = int(external_data.get("synthetic_2d_cases", 12))
    dataset_truth_label = (
        "synthetic_placeholder"
        if external_2d_dataset in {None, "", "null"}
        else str(external_data.get("external_2d_truth_label", "external_unspecified"))
    )

    weights_directory = model_config.get("weights_directory")
    training_statistics_path = model_config.get("training_statistics_path")
    model_size = str(benchmark_case.get("model_size", AFC_BENCHMARK_CASE["model_size"]))

    if external_2d_dataset in {None, "", "null"}:
        df_2d = make_fake_afc_dataset(n_cases=synthetic_2d_cases, random_seed=19)
        dataset_source = "synthetic"
        dataset_manifest = {
            "dataset_id": "synthetic_validation_placeholder",
            "source": "synthetic_placeholder",
            "truth_label": "synthetic_placeholder",
            "n_cases": int(len(df_2d)),
            "policy": "Synthetic placeholder benchmark used for CI and lightweight regression only.",
        }
        dataset_manifest_path = None
    else:
        df_2d = load_external_afc_validation_dataset(external_2d_dataset)
        dataset_source = str(external_2d_dataset)
        if external_2d_manifest in {None, "", "null"}:
            inferred_manifest = _infer_dataset_manifest_path(external_2d_dataset)
            external_2d_manifest = inferred_manifest
        dataset_manifest = _load_json_if_exists(external_2d_manifest)
        dataset_manifest_path = (
            None if external_2d_manifest is None else str(Path(external_2d_manifest))
        )

    zero_afc = run_2d_zero_afc_consistency_check(
        alphas=(-4.0, 0.0, 6.0, 12.0),
        Re=float(benchmark_case["reynolds"]),
        mach=float(benchmark_case["mach"]),
        model_size=model_size,
        tolerance=float(thresholds.get("zero_afc_abs_tolerance", 1e-12)),
        benchmark_case=benchmark_case,
        weights_directory=weights_directory,
        training_statistics_path=training_statistics_path,
    )
    surrogate_error = compute_2d_surrogate_error(
        df_2d,
        max_cases=int(report_config.get("max_2d_cases", 25)),
        model_size=model_size,
        weights_directory=weights_directory,
        training_statistics_path=training_statistics_path,
    )
    surrogate_error["criteria"] = {
        "CL_mae_max": float(thresholds.get("cl_mae_target_first_release", 0.03)),
        "CD_rmse_max": float(thresholds.get("cd_relative_error_target_first_release", 0.15)),
        "CM_rmse_max": float(thresholds.get("cm_rmse_target_first_release", 0.02)),
    }
    surrogate_error["passed"] = bool(
        surrogate_error["metrics"]["CL"]["mae"] <= surrogate_error["criteria"]["CL_mae_max"]
        and surrogate_error["metrics"]["CD"]["rmse"] <= surrogate_error["criteria"]["CD_rmse_max"]
        and surrogate_error["metrics"]["CM"]["rmse"] <= surrogate_error["criteria"]["CM_rmse_max"]
    )

    polar_comparison = compute_2d_polar_comparison(
        df_2d,
        max_cases=int(report_config.get("max_2d_cases", 25)),
        model_size=model_size,
        weights_directory=weights_directory,
        training_statistics_path=training_statistics_path,
    )
    baseline_3d = run_3d_no_afc_baseline_check(
        spanwise_resolution=int(benchmark_case["spanwise_resolution"]),
        model_size=model_size,
        include_avl=bool(checks_config.get("include_avl", False)),
        benchmark_case=benchmark_case,
        weights_directory=weights_directory,
        training_statistics_path=training_statistics_path,
    )
    afc_trend_3d = run_3d_afc_monotonicity_check(
        c_mu_values=tuple(benchmark_case["c_mu_sweep"]),
        spanwise_resolution=int(benchmark_case["spanwise_resolution"]),
        model_size=model_size,
        benchmark_case=benchmark_case,
        weights_directory=weights_directory,
        training_statistics_path=training_statistics_path,
    )
    cambered_vlm = run_3d_cambered_vlm_diagnostic_check(
        spanwise_resolution=int(benchmark_case["spanwise_resolution"]),
        model_size=model_size,
        benchmark_case=benchmark_case,
        weights_directory=weights_directory,
        training_statistics_path=training_statistics_path,
    )
    swap_trend = run_swap_trend_validation()

    optimization_output_dir = Path(
        output_directory if output_directory is not None else report_config.get("output_directory", "validation_outputs/afc")
    ) / "figures" / "optimization_case"
    opti_report = run_end_to_end_optimization_validation(output_directory=optimization_output_dir)

    sections = [
        {
            "id": "2d_zero_afc_consistency",
            "name": "2D zero-AFC consistency",
            "data_provenance": "internal_regression_check",
            "passed": zero_afc["passed"],
            "criteria": {"zero_afc_abs_tolerance": float(thresholds.get("zero_afc_abs_tolerance", 1e-12))},
            "results": zero_afc,
        },
        {
            "id": "2d_model_error",
            "name": "2D model error",
            "data_provenance": dataset_truth_label,
            "passed": surrogate_error["passed"],
            "criteria": surrogate_error["criteria"],
            "results": surrogate_error,
        },
        {
            "id": "2d_polar_comparison",
            "name": "2D polar comparison",
            "data_provenance": dataset_truth_label,
            "passed": polar_comparison["passed"],
            "criteria": {"status": "informational trend comparison"},
            "results": polar_comparison,
        },
        {
            "id": "3d_baseline_comparison",
            "name": "3D baseline comparison",
            "data_provenance": "baseline_solver_comparison",
            "passed": baseline_3d["passed"],
            "criteria": {
                "three_d_no_afc_abs_tolerance": float(
                    thresholds.get("three_d_no_afc_abs_tolerance", 1e-10)
                )
            },
            "results": baseline_3d,
        },
        {
            "id": "3d_afc_gain_trend",
            "name": "AFC lift gain trend",
            "data_provenance": "surrogate_trend_check",
            "passed": afc_trend_3d["passed"],
            "criteria": {"status": "CL should increase monotonically with C_mu sweep"},
            "results": afc_trend_3d,
        },
        {
            "id": "3d_cambered_vlm_diagnostics",
            "name": "3D cambered VLM diagnostics",
            "data_provenance": "surrogate_solver_diagnostics",
            "passed": cambered_vlm["passed"],
            "criteria": {"status": "Required output fields present and residual decreases"},
            "results": cambered_vlm,
        },
        {
            "id": "swap_trend_validation",
            "name": "SWaP trend validation",
            "data_provenance": "placeholder_system_model",
            "passed": swap_trend["passed"],
            "criteria": {"status": "required_power and mass_flow must increase with target C_mu"},
            "results": swap_trend,
        },
        {
            "id": "end_to_end_optimization",
            "name": "End-to-end optimization before/after",
            "data_provenance": "placeholder_integrated_workflow",
            "passed": opti_report["passed"],
            "criteria": {"status": "optimized objective should not regress and solve must succeed"},
            "results": opti_report,
        },
    ]

    metadata = {
        "code_version": {
            "git_commit": _git_commit_hash(),
            "package_version": __import__("aerosandbox").__version__,
        },
        "model_weights_path": None if weights_directory in {None, "", "null"} else str(weights_directory),
        "training_statistics_path": None
        if training_statistics_path in {None, "", "null"}
        else str(training_statistics_path),
        "dataset_source": dataset_source,
        "dataset_manifest_path": dataset_manifest_path,
        "dataset_manifest": dataset_manifest,
        "config_path": None if config_path is None else str(Path(config_path)),
        "run_time": _runtime_summary(start_time),
    }

    report = {
        "report_type": "afc_validation_acceptance",
        "benchmark_case": benchmark_case,
        "metadata": metadata,
        "pass_fail_standards": thresholds,
        "sections": sections,
        "passed": all(section["passed"] for section in sections),
    }
    return report


def write_afc_validation_report(
    report: Dict[str, object],
    output_directory: Union[str, Path],
) -> Dict[str, Path]:
    """
    Writes a JSON and Markdown AFC validation report.
    """
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    if "sections" in report:
        return _write_formal_afc_validation_report(report, output_directory)

    json_path = output_directory / "afc_validation_report.json"
    markdown_path = output_directory / "afc_validation_report.md"

    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=_json_default)

    paths = {
        "json": json_path,
        "markdown": markdown_path,
    }

    comparison_check = next(
        (check for check in report["checks"] if "comparison_table" in check),
        None,
    )
    if comparison_check is not None:
        comparison_path = output_directory / "afc_3d_comparison_table.csv"
        pd.DataFrame(comparison_check["comparison_table"]).to_csv(
            comparison_path,
            index=False,
        )
        paths["comparison_table"] = comparison_path

    monotonicity_check = next(
        (check for check in report["checks"] if check["name"] == "3D AFC monotonicity and continuity"),
        None,
    )
    if monotonicity_check is not None:
        import matplotlib.pyplot as plt

        data = monotonicity_check["spanwise_plot_data"]
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), dpi=150)
        axes[0].plot(data["spanwise_y"], data["spanwise_load"], "o-")
        axes[0].set_xlabel("spanwise y [m]")
        axes[0].set_ylabel("spanwise load [N/m]")
        axes[0].grid(True, alpha=0.3)
        axes[1].plot(data["spanwise_y"], data["local_C_mu"], "s-", label="C_mu")
        axes[1].plot(data["spanwise_y"], data["section_CL"], "o-", label="section CL")
        axes[1].plot(data["spanwise_y"], data["section_CD"], "^-", label="section CD")
        axes[1].set_xlabel("spanwise y [m]")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        fig.suptitle("AFC 3D Spanwise Validation")
        fig.tight_layout()
        spanwise_plot_path = output_directory / "afc_3d_spanwise_load.png"
        fig.savefig(spanwise_plot_path, bbox_inches="tight")
        plt.close(fig)
        paths["spanwise_load_plot"] = spanwise_plot_path

    convergence_check = next(
        (check for check in report["checks"] if "convergence_history" in check),
        None,
    )
    if convergence_check is not None:
        import matplotlib.pyplot as plt

        history = convergence_check["convergence_history"]
        fig, ax = plt.subplots(figsize=(6, 3.5), dpi=150)
        iterations = [row["iteration"] for row in history]
        ax.semilogy(
            iterations,
            [row["max_abs_residual_CL"] for row in history],
            "o-",
            label="max |residual CL|",
        )
        ax.semilogy(
            iterations,
            [row["rms_residual_CL"] for row in history],
            "s--",
            label="RMS residual CL",
        )
        if any(row["max_abs_residual_CM"] > 0 for row in history):
            ax.semilogy(
                iterations,
                [row["max_abs_residual_CM"] for row in history],
                "^-",
                label="max |residual CM|",
            )
        ax.set_xlabel("iteration")
        ax.set_ylabel("residual")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        ax.set_title("NeuralFoilCamberedVLM Residual History")
        fig.tight_layout()
        residual_plot_path = output_directory / "afc_3d_residual_convergence.png"
        fig.savefig(residual_plot_path, bbox_inches="tight")
        plt.close(fig)
        paths["residual_convergence_plot"] = residual_plot_path

    lines = [
        "# AFC Validation Report",
        "",
        f"Overall status: {'PASS' if report['passed'] else 'FAIL'}",
        "",
        f"Dataset source: `{report['dataset_source']}`",
        "",
        "## Checks",
        "",
    ]
    for check in report["checks"]:
        lines.append(f"### {check['name']}")
        lines.append("")
        lines.append(f"Status: {'PASS' if check.get('passed', False) else 'FAIL'}")
        lines.append("")
        if "comparison_table" in check:
            lines.append("| solver | CL | CD | CY | Cl | Cm | Cn |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|")
            for row in check["comparison_table"]:
                lines.append(
                    "| {solver} | {CL:.6g} | {CD:.6g} | {CY:.6g} | {Cl:.6g} | {Cm:.6g} | {Cn:.6g} |".format(
                        **{
                            key: (0.0 if row.get(key) is None else row.get(key))
                            for key in ["CL", "CD", "CY", "Cl", "Cm", "Cn"]
                        },
                        solver=row["solver"],
                    )
                )
            lines.append("")
        lines.append("```json")
        lines.append(json.dumps(check, indent=2, default=_json_default))
        lines.append("```")
        lines.append("")

    if "spanwise_load_plot" in paths:
        lines.extend(["## Generated Figures", ""])
        lines.append(f"- 3D comparison table: `{paths.get('comparison_table')}`")
        lines.append(f"- Spanwise load plot: `{paths['spanwise_load_plot']}`")
        if "residual_convergence_plot" in paths:
            lines.append(f"- Residual convergence plot: `{paths['residual_convergence_plot']}`")
        lines.append("")

    markdown_path.write_text("\n".join(lines))

    return paths


def _write_formal_afc_validation_report(
    report: Dict[str, object],
    output_directory: Path,
) -> Dict[str, Path]:
    import matplotlib.pyplot as plt

    output_directory.mkdir(parents=True, exist_ok=True)
    figures_dir = output_directory / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_directory / "report.json"
    markdown_path = output_directory / "report.md"
    json_path.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")

    paths = {
        "json": json_path,
        "markdown": markdown_path,
        "figures": figures_dir,
    }

    sections = {section["id"]: section for section in report["sections"]}

    error_section = sections.get("2d_model_error")
    if error_section is not None:
        metrics = error_section["results"]["metrics"]
        fields = ["CL", "CD", "CM"]
        mae = [metrics[field]["mae"] for field in fields]
        rmse = [metrics[field]["rmse"] for field in fields]
        x = onp.arange(len(fields))
        fig, ax = plt.subplots(figsize=(5.5, 3.4), dpi=150)
        ax.bar(x - 0.18, mae, width=0.36, label="MAE")
        ax.bar(x + 0.18, rmse, width=0.36, label="RMSE")
        ax.set_xticks(x)
        ax.set_xticklabels(fields)
        ax.set_ylabel("Error")
        ax.set_title("2D Surrogate Error Summary")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        path = figures_dir / "2d_error_summary.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths["2d_error_summary"] = path

    polar_section = sections.get("2d_polar_comparison")
    if polar_section is not None and polar_section["results"]["polar_slices"]:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), dpi=150)
        for polar_slice in polar_section["results"]["polar_slices"]:
            label = f"C_mu={polar_slice['C_mu']:.3f}"
            axes[0].plot(polar_slice["alpha"], polar_slice["actual_CL"], "o-", label=f"{label} actual")
            axes[0].plot(polar_slice["alpha"], polar_slice["predicted_CL"], "--", label=f"{label} pred")
            axes[1].plot(polar_slice["alpha"], polar_slice["actual_CD"], "o-", label=f"{label} actual")
            axes[1].plot(polar_slice["alpha"], polar_slice["predicted_CD"], "--", label=f"{label} pred")
        axes[0].set_xlabel("alpha [deg]")
        axes[0].set_ylabel("CL")
        axes[0].grid(True, alpha=0.3)
        axes[1].set_xlabel("alpha [deg]")
        axes[1].set_ylabel("CD")
        axes[1].grid(True, alpha=0.3)
        axes[0].set_title("2D Polar Comparison")
        axes[1].set_title("2D Drag Polar Slices")
        axes[1].legend(fontsize=7, ncol=2)
        fig.tight_layout()
        path = figures_dir / "2d_polar_comparison.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths["2d_polar_comparison"] = path

    baseline_section = sections.get("3d_baseline_comparison")
    if baseline_section is not None:
        table = pd.DataFrame(baseline_section["results"]["comparison_table"])
        table_path = figures_dir / "3d_baseline_comparison.csv"
        table.to_csv(table_path, index=False)
        paths["3d_baseline_comparison_table"] = table_path
        fig, ax = plt.subplots(figsize=(6.2, 3.6), dpi=150)
        x = onp.arange(len(table))
        ax.bar(x - 0.18, table["CL"], width=0.36, label="CL")
        ax.bar(x + 0.18, table["CD"], width=0.36, label="CD")
        ax.set_xticks(x)
        ax.set_xticklabels(table["solver"], rotation=20, ha="right")
        ax.set_title("3D Baseline Solver Comparison")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        path = figures_dir / "3d_baseline_comparison.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths["3d_baseline_comparison"] = path

    trend_section = sections.get("3d_afc_gain_trend")
    if trend_section is not None:
        cases = trend_section["results"]["cases"]
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), dpi=150)
        c_mu = [case["C_mu"] for case in cases]
        axes[0].plot(c_mu, [case["CL"] for case in cases], "o-")
        axes[0].set_xlabel("C_mu")
        axes[0].set_ylabel("CL")
        axes[0].set_title("AFC Lift Gain Trend")
        axes[0].grid(True, alpha=0.3)
        data = trend_section["results"]["spanwise_plot_data"]
        axes[1].plot(data["spanwise_y"], data["spanwise_load"], "o-", label="spanwise load")
        axes[1].plot(data["spanwise_y"], data["local_C_mu"], "s--", label="local C_mu")
        axes[1].set_xlabel("spanwise y [m]")
        axes[1].set_title("Spanwise Load / Actuation")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(fontsize=8)
        fig.tight_layout()
        path = figures_dir / "3d_afc_gain_trend.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths["3d_afc_gain_trend"] = path

    cambered_section = sections.get("3d_cambered_vlm_diagnostics")
    if cambered_section is not None:
        history = cambered_section["results"]["convergence_history"]
        fig, ax = plt.subplots(figsize=(5.6, 3.6), dpi=150)
        iterations = [row["iteration"] for row in history]
        ax.semilogy(iterations, [row["max_abs_residual_CL"] for row in history], "o-", label="max |residual CL|")
        ax.semilogy(iterations, [row["rms_residual_CL"] for row in history], "s--", label="RMS residual CL")
        ax.semilogy(iterations, [max(row["max_abs_residual_CM"], 1e-16) for row in history], "^-", label="max |residual CM|")
        ax.set_xlabel("iteration")
        ax.set_ylabel("residual")
        ax.set_title("3D Residual Convergence")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = figures_dir / "3d_residual_convergence.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths["3d_residual_convergence"] = path

    swap_section = sections.get("swap_trend_validation")
    if swap_section is not None:
        cases = pd.DataFrame(swap_section["results"]["cases"])
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), dpi=150)
        axes[0].plot(cases["C_mu"], cases["mass_flow"], "o-")
        axes[0].set_xlabel("target C_mu")
        axes[0].set_ylabel("mass flow")
        axes[0].set_title("SWaP Trend: Mass Flow")
        axes[0].grid(True, alpha=0.3)
        axes[1].plot(cases["C_mu"], cases["required_power"], "o-")
        axes[1].set_xlabel("target C_mu")
        axes[1].set_ylabel("required power [W]")
        axes[1].set_title("SWaP Trend: Required Power")
        axes[1].grid(True, alpha=0.3)
        fig.tight_layout()
        path = figures_dir / "swap_trend_validation.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths["swap_trend_validation"] = path

    optimization_section = sections.get("end_to_end_optimization")
    if optimization_section is not None:
        initial = optimization_section["results"]["initial"]
        optimized = optimization_section["results"]["optimized"]
        if optimized is not None:
            labels = ["CD_total", "required_power", "system_weight", "CL"]
            initial_values = [initial[label] for label in labels]
            optimized_values = [optimized[label] for label in labels]
            x = onp.arange(len(labels))
            fig, ax = plt.subplots(figsize=(6.2, 3.6), dpi=150)
            ax.bar(x - 0.18, initial_values, width=0.36, label="initial")
            ax.bar(x + 0.18, optimized_values, width=0.36, label="optimized")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=15)
            ax.set_title("End-to-End Optimization Before / After")
            ax.grid(True, axis="y", alpha=0.3)
            ax.legend()
            fig.tight_layout()
            path = figures_dir / "end_to_end_optimization_comparison.png"
            fig.savefig(path, bbox_inches="tight")
            plt.close(fig)
            paths["end_to_end_optimization_comparison"] = path

    lines = [
        "# AFC Validation Acceptance Report",
        "",
        f"Overall status: {'PASS' if report['passed'] else 'FAIL'}",
        "",
        "## Metadata",
        "",
        f"- Git commit: `{report['metadata']['code_version'].get('git_commit')}`",
        f"- Package version: `{report['metadata']['code_version'].get('package_version')}`",
        f"- Model weights path: `{report['metadata'].get('model_weights_path')}`",
        f"- Training statistics path: `{report['metadata'].get('training_statistics_path')}`",
        f"- Dataset source: `{report['metadata'].get('dataset_source')}`",
        f"- Dataset manifest path: `{report['metadata'].get('dataset_manifest_path')}`",
        f"- Config path: `{report['metadata'].get('config_path')}`",
        f"- Run time [s]: `{report['metadata']['run_time']['elapsed_seconds']:.3f}`",
        "",
        "## Pass/Fail Standards",
        "",
        "```json",
        json.dumps(report["pass_fail_standards"], indent=2, default=_json_default),
        "```",
        "",
        "## Sections",
        "",
    ]

    for section in report["sections"]:
        lines.append(f"### {section['name']}")
        lines.append("")
        lines.append(f"- Status: {'PASS' if section['passed'] else 'FAIL'}")
        lines.append(f"- Data provenance: `{section['data_provenance']}`")
        lines.append("")
        lines.append("Criteria:")
        lines.append("```json")
        lines.append(json.dumps(section["criteria"], indent=2, default=_json_default))
        lines.append("```")
        lines.append("")
        lines.append("Results:")
        lines.append("```json")
        lines.append(json.dumps(section["results"], indent=2, default=_json_default))
        lines.append("```")
        lines.append("")

    lines.extend(
        [
            "## Figures",
            "",
            f"- figures directory: `{figures_dir}`",
            "- placeholder/synthetic sections are explicitly marked in `Data provenance` above.",
            "- large external CFD / wind-tunnel datasets are not stored in this repository.",
            "",
        ]
    )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return paths
