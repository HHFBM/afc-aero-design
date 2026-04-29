import importlib.util
import json
from pathlib import Path


def _load_example_module():
    repo_root = Path(__file__).resolve().parents[3]
    module_path = repo_root / "examples" / "afc_concept_design_optimization.py"
    spec = importlib.util.spec_from_file_location(
        "afc_concept_design_optimization",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_afc_concept_design_optimization_example_writes_outputs(tmp_path):
    example = _load_example_module()
    summary = example.run_example(output_directory=tmp_path)

    assert summary["status"] in {"solved", "failed"}
    assert "failure_reason" in summary
    assert "initial" in summary
    assert summary["initial"]["failure_reason"] is None

    outputs = summary["outputs"]
    for key in [
        "comparison_table",
        "spanwise_c_mu_plot",
        "spanwise_load_plot",
        "section_cl_cd_plot",
        "swap_breakdown_plot",
    ]:
        assert key in outputs
        assert Path(outputs[key]).exists()

    summary_path = tmp_path / "summary.json"
    assert summary_path.exists()
    loaded = json.loads(summary_path.read_text())
    assert loaded["status"] == summary["status"]
    assert "design_variables" in loaded
    assert loaded["design_variables"]["C_mu_segments"] == 4


if __name__ == "__main__":
    import tempfile

    test_afc_concept_design_optimization_example_writes_outputs(
        Path(tempfile.mkdtemp())
    )
