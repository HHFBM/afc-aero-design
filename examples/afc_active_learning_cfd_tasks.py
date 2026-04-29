"""
Generate the next batch of AFC CFD task recommendations without running CFD.

This example keeps the AFC 2D training-data schema separate from the CFD task schema:
- Training data: inputs + labels (`CL`, `CD`, `CM`, ...)
- CFD task list: inputs + solver/mesh/boundary metadata only

Run from the repository root:

    .venv-aerosandbox/bin/python examples/afc_active_learning_cfd_tasks.py
"""

from pathlib import Path

import aerosandbox as asb
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(__file__).parent / "generated_afc_active_learning_tasks"
TRAINING_DATA_PATH = (
    ROOT
    / "external_data/flowcontrollab/converted/fcl_naca0018_quasi_steady_afc_schema_with_baseline.csv"
)
TRAINED_WEIGHTS_DIR = (
    ROOT
    / "runs/afc_neuralfoil/retrain_20260423_fcl_medium/exported_weights"
)
TRAINING_STATISTICS_PATH = (
    ROOT
    / "runs/afc_neuralfoil/retrain_20260423_fcl_medium/afc_confidence_training_statistics.json"
)


def _make_model_callable_if_available():
    if not (TRAINED_WEIGHTS_DIR.exists() and TRAINING_STATISTICS_PATH.exists()):
        return None, "default_current_model"

    def model_callable(candidates: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, row in candidates.iterrows():
            airfoil = asb.KulfanAirfoil(
                name="AFC active-learning example",
                upper_weights=[row[column] for column in asb.KULFAN_UPPER_COLUMNS],
                lower_weights=[row[column] for column in asb.KULFAN_LOWER_COLUMNS],
                leading_edge_weight=float(row["leading_edge_weight"]),
                TE_thickness=float(row["TE_thickness"]),
            )
            aero = airfoil.get_aero_from_afc_neuralfoil(
                alpha=float(row["alpha"]),
                Re=float(row["Re"]),
                mach=float(row["Mach"]),
                C_mu=float(row["C_mu"]),
                x_jet=float(row["x_jet"]),
                theta_jet=float(row["theta_jet"]),
                model_size="medium",
                weights_directory=TRAINED_WEIGHTS_DIR,
                training_statistics_path=TRAINING_STATISTICS_PATH,
            )
            rows.append(
                {
                    "CL": aero["CL"],
                    "CD": aero["CD"],
                    "CM": aero["CM"],
                    "Top_Xtr": aero["Top_Xtr"],
                    "Bot_Xtr": aero["Bot_Xtr"],
                    "analysis_confidence": aero["analysis_confidence"],
                }
            )
        return pd.DataFrame(rows)

    return model_callable, str(TRAINED_WEIGHTS_DIR)


def run_example(output_dir: Path = OUTPUT_DIR) -> dict:
    if TRAINING_DATA_PATH.exists():
        training_data = asb.read_afc_dataset(TRAINING_DATA_PATH)
        training_source = str(TRAINING_DATA_PATH)
    else:
        training_data = asb.make_fake_afc_dataset(
            n_cases=120,
            random_seed=11,
            source="synthetic_fallback",
        )
        training_source = "synthetic_fallback"

    anchor = training_data.iloc[0]
    fixed_values = {
        **{column: float(anchor[column]) for column in asb.GEOMETRY_COLUMNS},
        "n_crit": float(training_data["n_crit"].median()),
        "xtr_upper": float(training_data["xtr_upper"].median()),
        "xtr_lower": float(training_data["xtr_lower"].median()),
    }
    design_space = {
        "alpha": (-2.0, 18.0),
        "Re": (
            max(6.0e4, float(training_data["Re"].min()) * 0.85),
            min(8.0e5, float(training_data["Re"].max()) * 1.10),
        ),
        "Mach": (0.0, max(0.12, float(training_data["Mach"].max()))),
        "C_mu": (0.0, 0.12),
        "x_jet": (
            max(0.02, float(training_data["x_jet"].min()) - 0.02),
            min(0.20, float(training_data["x_jet"].max()) + 0.02),
        ),
        "theta_jet": (
            max(5.0, float(training_data["theta_jet"].min()) - 10.0),
            min(60.0, float(training_data["theta_jet"].max()) + 10.0),
        ),
    }
    optimum = {
        "alpha": 8.0,
        "Re": float(training_data["Re"].median()),
        "Mach": 0.04,
        "C_mu": 0.055,
        "x_jet": float(training_data["x_jet"].median()),
        "theta_jet": float(training_data["theta_jet"].median()),
    }

    model_callable, model_source = _make_model_callable_if_available()

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "afc_cfd_tasks.csv"
    parquet_path = output_dir / "afc_cfd_tasks.parquet"
    jsonl_path = output_dir / "afc_cfd_tasks.jsonl"

    tasks = asb.recommend_afc_cfd_tasks(
        training_data,
        design_space,
        n_candidates=320,
        n_recommend=50,
        n_initial_lhs=10,
        optimum=optimum,
        local_refinement_fraction=0.30,
        local_radius_fraction=0.12,
        fixed_values=fixed_values,
        model_callable=model_callable,
        model_size="medium",
        random_seed=17,
        solver_type="SU2_RANS",
        mesh_config={
            "family": "c_grid_placeholder",
            "target_y_plus": 1.0,
            "chordwise_points": 321,
            "normal_layers": 128,
            "note": "placeholder_mesh_recipe_for_future_batch_cfd",
        },
        boundary_condition_config={
            "freestream": "specified_by_alpha_Re_Mach",
            "actuation": "specified_by_C_mu_x_jet_theta_jet",
            "transition": "placeholder_user_pipeline_choice",
        },
    )

    asb.write_afc_cfd_tasks(tasks, csv_path)
    jsonl_written = asb.write_afc_cfd_tasks(tasks, jsonl_path)
    parquet_written = None
    try:
        parquet_written = asb.write_afc_cfd_tasks(tasks, parquet_path)
    except ImportError as exc:
        print(f"Parquet skipped: {exc}")

    reason_counts = asb.summarize_afc_cfd_task_reasons(tasks)

    print("AFC active-learning CFD task generation")
    print("=======================================")
    print(f"training data source : {training_source}")
    print(f"model source         : {model_source}")
    print(f"tasks generated      : {len(tasks)}")
    print(f"CSV                  : {csv_path}")
    print(f"JSONL                : {jsonl_written}")
    print(f"Parquet              : {parquet_written if parquet_written else 'not available'}")
    print("\nReason counts")
    print(reason_counts.to_string(index=False))

    return {
        "tasks": tasks,
        "reason_counts": reason_counts,
        "paths": {
            "csv": str(csv_path),
            "jsonl": str(jsonl_path),
            "parquet": str(parquet_path) if parquet_written else None,
        },
        "training_data_source": training_source,
        "model_source": model_source,
    }


if __name__ == "__main__":
    run_example()
