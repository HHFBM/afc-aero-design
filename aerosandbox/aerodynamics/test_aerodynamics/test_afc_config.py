import tempfile
from pathlib import Path

import aerosandbox as asb


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "configs" / "afc"


CONFIG_REQUIRED_FIELDS = {
    "dataset_config.example.yaml": [
        "afc_config_version",
        "config_type",
        "schema.name",
        "naming.afc_inputs",
        "units.C_mu",
        "columns.outputs",
        "split.train_fraction",
    ],
    "model_config.example.yaml": [
        "afc_config_version",
        "config_type",
        "model.model_size",
        "inputs.encoded_feature_count",
        "outputs.code_fields",
        "training.script",
    ],
    "aircraft_config.example.yaml": [
        "afc_config_version",
        "config_type",
        "airplane.reference.s_ref",
        "operating_point.velocity",
        "analysis.solver",
        "afc.C_mu",
        "outputs.totals",
    ],
    "optimization_config.example.yaml": [
        "afc_config_version",
        "config_type",
        "problem.optimizer",
        "design_variables.C_mu_segments",
        "constraints.CL_min",
        "outputs.summary_fields",
    ],
    "validation_config.example.yaml": [
        "afc_config_version",
        "config_type",
        "benchmark.model_size",
        "checks.run_2d_zero_afc_consistency",
        "thresholds.zero_afc_abs_tolerance",
        "report.output_directory",
    ],
}


def test_afc_example_configs_load_and_have_required_fields():
    for filename, required_fields in CONFIG_REQUIRED_FIELDS.items():
        config = asb.read_afc_config(
            CONFIG_DIR / filename,
            required_fields=required_fields,
        )

        assert config["afc_config_version"] == 1
        assert config["config_type"] in {
            "dataset",
            "model",
            "aircraft",
            "optimization",
            "validation",
        }

    dataset_config = asb.read_afc_config(CONFIG_DIR / "dataset_config.example.yaml")
    assert asb.get_afc_config_value(dataset_config, "units.C_mu") == "nondimensional"
    assert "C_mu" in asb.get_afc_config_value(dataset_config, "naming.afc_inputs")


def test_afc_config_merge_json_and_required_field_check():
    defaults = {
        "afc_config_version": 1,
        "config_type": "unit_test",
        "model": {
            "model_size": "small",
            "weights_directory": "default_weights",
        },
    }
    override = {
        "model": {
            "model_size": "large",
        },
    }

    merged = asb.merge_afc_configs(defaults, override)
    assert merged["model"]["model_size"] == "large"
    assert merged["model"]["weights_directory"] == "default_weights"

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "config.json"
        asb.write_afc_config(merged, path)
        loaded = asb.read_afc_config(
            path,
            required_fields=["afc_config_version", "model.model_size"],
        )

    assert loaded["model"]["model_size"] == "large"

    try:
        asb.require_afc_config_fields(loaded, ["missing.field"])
    except asb.AFCConfigError:
        return

    raise AssertionError("Expected missing config field to raise AFCConfigError.")


if __name__ == "__main__":
    test_afc_example_configs_load_and_have_required_fields()
    test_afc_config_merge_json_and_required_field_check()

