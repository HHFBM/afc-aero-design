import aerosandbox as asb
import aerosandbox.numpy as np
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_afc_validation_2d_zero_afc_consistency():
    report = asb.run_2d_zero_afc_consistency_check()
    assert report["passed"]


def test_afc_validation_2d_surrogate_error_is_computable():
    df = asb.make_fake_afc_dataset(n_cases=5, random_seed=101)
    report = asb.compute_2d_surrogate_error(df, max_cases=5)
    assert report["passed"]
    assert report["n_cases"] == 5
    for field in ["CL", "CD", "CM", "Top_Xtr", "Bot_Xtr"]:
        assert np.isfinite(report["metrics"][field]["rmse"])


def test_afc_validation_3d_no_afc_matches_lifting_line():
    report = asb.run_3d_no_afc_baseline_check()
    assert report["passed"]
    assert len(report["comparison_table"]) >= 3
    solver_names = [row["solver"] for row in report["comparison_table"]]
    assert "AeroSandbox VLM inviscid" in solver_names


def test_afc_validation_3d_afc_monotonicity_and_continuity():
    report = asb.run_3d_afc_monotonicity_check()
    assert report["passed"]
    assert report["cases"][-1]["CL"] >= report["cases"][0]["CL"]
    assert "spanwise_plot_data" in report


def test_afc_validation_3d_cambered_vlm_diagnostics():
    report = asb.run_3d_cambered_vlm_diagnostic_check()
    assert report["passed"]
    assert report["missing_fields"] == []
    assert len(report["convergence_history"]) >= 1
    assert report["final_max_abs_residual_CL"] <= report["initial_max_abs_residual_CL"]


def test_afc_validation_opti_converges():
    report = asb.run_opti_convergence_check()
    assert report["passed"]
    assert 0 <= report["C_mu"] <= 0.1


def test_afc_validation_suite_runs():
    report = asb.run_afc_validation_suite(synthetic_2d_cases=4)
    assert report["passed"]
    assert len(report["checks"]) == 6


def test_afc_validation_report_writes_3d_artifacts(tmp_path):
    report = asb.run_afc_validation_suite(synthetic_2d_cases=3)
    paths = asb.write_afc_validation_report(report, output_directory=tmp_path)
    for key in [
        "json",
        "markdown",
        "comparison_table",
        "spanwise_load_plot",
        "residual_convergence_plot",
    ]:
        assert key in paths
        assert paths[key].exists()


def test_afc_validation_acceptance_report_from_config(tmp_path):
    config_path = REPO_ROOT / "configs" / "afc" / "validation_config.example.yaml"
    config = asb.read_afc_config(config_path)
    config = asb.merge_afc_configs(
        config,
        {
            "benchmark": {"model_size": "xxsmall", "spanwise_resolution": 2},
            "external_data": {"synthetic_2d_cases": 3},
            "report": {"output_directory": str(tmp_path), "max_2d_cases": 6},
        },
    )
    report = asb.generate_afc_validation_acceptance_report(
        config,
        config_path=config_path,
        output_directory=tmp_path,
    )
    paths = asb.write_afc_validation_report(report, output_directory=tmp_path)

    assert "sections" in report
    assert report["metadata"]["config_path"] == str(config_path)
    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "figures").exists()
    for key in ["2d_error_summary", "2d_polar_comparison", "3d_baseline_comparison", "swap_trend_validation"]:
        assert key in paths
        assert paths[key].exists()


if __name__ == "__main__":
    test_afc_validation_2d_zero_afc_consistency()
    test_afc_validation_2d_surrogate_error_is_computable()
    test_afc_validation_3d_no_afc_matches_lifting_line()
    test_afc_validation_3d_afc_monotonicity_and_continuity()
    test_afc_validation_3d_cambered_vlm_diagnostics()
    test_afc_validation_opti_converges()
    test_afc_validation_suite_runs()
