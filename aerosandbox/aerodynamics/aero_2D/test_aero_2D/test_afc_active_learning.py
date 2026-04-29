import tempfile
from pathlib import Path

import aerosandbox as asb
import numpy as np
import pandas as pd


def _make_design_space():
    return {
        "alpha": (-1.0, 16.0),
        "Re": (1.2e5, 2.2e5),
        "Mach": (0.01, 0.08),
        "C_mu": (0.0, 0.10),
        "x_jet": (0.08, 0.16),
        "theta_jet": (20.0, 40.0),
    }


def _make_fixed_values(df: pd.DataFrame):
    anchor = df.iloc[0]
    return {
        **{column: float(anchor[column]) for column in asb.GEOMETRY_COLUMNS},
        "n_crit": float(anchor["n_crit"]),
        "xtr_upper": float(anchor["xtr_upper"]),
        "xtr_lower": float(anchor["xtr_lower"]),
    }


def _confidence_weighted_model(candidates: pd.DataFrame) -> pd.DataFrame:
    c_mu = candidates["C_mu"].to_numpy(dtype=float)
    alpha = candidates["alpha"].to_numpy(dtype=float)
    confidence = np.clip(0.98 - 6.5 * c_mu - 0.015 * np.maximum(alpha - 10.0, 0.0), 0.01, 0.99)
    cl = 0.12 * alpha + 2.8 * c_mu
    cd = 0.015 + 0.030 * c_mu**2 + 0.0005 * alpha**2
    cm = -0.03 - 0.20 * c_mu
    top_xtr = np.clip(0.65 - 3.0 * c_mu, 0.02, 0.95)
    bot_xtr = np.clip(0.85 - 1.5 * c_mu, 0.05, 0.98)
    return pd.DataFrame(
        {
            "CL": cl,
            "CD": cd,
            "CM": cm,
            "Top_Xtr": top_xtr,
            "Bot_Xtr": bot_xtr,
            "analysis_confidence": confidence,
        }
    )


def test_afc_cfd_task_schema_is_complete():
    training_data = asb.make_fake_afc_dataset(n_cases=40, random_seed=21)
    tasks = asb.recommend_afc_cfd_tasks(
        training_data,
        _make_design_space(),
        n_candidates=80,
        n_recommend=12,
        n_initial_lhs=3,
        fixed_values=_make_fixed_values(training_data),
        model_callable=_confidence_weighted_model,
        random_seed=9,
    )

    assert list(tasks.columns) == asb.AFC_CFD_TASK_COLUMNS
    assert len(tasks) == 12
    assert tasks["task_id"].is_unique
    assert tasks["priority"].tolist() == list(range(1, 13))
    assert (tasks["solver_type"] == "SU2_RANS").all()
    assert tasks["mesh_config"].astype(str).str.len().min() > 5
    assert tasks["boundary_condition_config"].astype(str).str.len().min() > 5


def test_afc_cfd_task_prioritizes_high_c_mu_when_confidence_is_low():
    training_data = asb.make_fake_afc_dataset(n_cases=36, random_seed=22)
    tasks = asb.recommend_afc_cfd_tasks(
        training_data,
        _make_design_space(),
        n_candidates=140,
        n_recommend=10,
        n_initial_lhs=0,
        fixed_values=_make_fixed_values(training_data),
        model_callable=_confidence_weighted_model,
        score_weights={
            "uncertainty": 0.70,
            "gradient": 0.10,
            "optimization": 0.05,
            "stall": 0.10,
            "novelty": 0.05,
        },
        random_seed=4,
    )

    selected_mean = float(tasks["C_mu"].mean())
    top_value = float(tasks.iloc[0]["C_mu"])

    assert selected_mean >= 0.08
    assert top_value >= 0.08
    assert tasks["reason"].astype(str).str.contains("低confidence").any()


def test_afc_cfd_task_file_roundtrip_csv_parquet_jsonl():
    training_data = asb.make_fake_afc_dataset(n_cases=30, random_seed=23)
    tasks = asb.recommend_afc_cfd_tasks(
        training_data,
        _make_design_space(),
        n_candidates=60,
        n_recommend=8,
        n_initial_lhs=2,
        fixed_values=_make_fixed_values(training_data),
        model_callable=_confidence_weighted_model,
        random_seed=2,
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        csv_path = tmp_path / "tasks.csv"
        jsonl_path = tmp_path / "tasks.jsonl"
        parquet_path = tmp_path / "tasks.parquet"

        asb.write_afc_cfd_tasks(tasks, csv_path)
        asb.write_afc_cfd_tasks(tasks, jsonl_path)
        csv_loaded = asb.read_afc_cfd_tasks(csv_path)
        jsonl_loaded = asb.read_afc_cfd_tasks(jsonl_path)

        assert list(csv_loaded.columns) == asb.AFC_CFD_TASK_COLUMNS
        assert list(jsonl_loaded.columns) == asb.AFC_CFD_TASK_COLUMNS
        assert np.allclose(csv_loaded["priority_score"], tasks["priority_score"])
        assert np.allclose(jsonl_loaded["C_mu"], tasks["C_mu"])

        try:
            asb.write_afc_cfd_tasks(tasks, parquet_path)
            parquet_loaded = asb.read_afc_cfd_tasks(parquet_path)
            assert np.allclose(parquet_loaded["alpha"], tasks["alpha"])
        except ImportError:
            pass


if __name__ == "__main__":
    test_afc_cfd_task_schema_is_complete()
    test_afc_cfd_task_prioritizes_high_c_mu_when_confidence_is_low()
    test_afc_cfd_task_file_roundtrip_csv_parquet_jsonl()
