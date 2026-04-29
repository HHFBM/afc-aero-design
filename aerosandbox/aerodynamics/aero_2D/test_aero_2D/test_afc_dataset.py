import tempfile
from pathlib import Path

import aerosandbox as asb
import aerosandbox.numpy as np
import pandas as pd


def test_afc_dataset_schema_validation_split_and_csv_io():
    df = asb.make_fake_afc_dataset(n_cases=25, random_seed=3)
    asb.validate_afc_dataframe(df)

    train, val, test = asb.split_afc_dataset(
        df,
        train_fraction=0.6,
        val_fraction=0.2,
        test_fraction=0.2,
        random_seed=4,
    )

    assert len(train) == 15
    assert len(val) == 5
    assert len(test) == 5

    report = asb.afc_dataset_statistics(df)
    assert report["n_cases"] == 25
    assert 0 <= report["converged_fraction"] <= 1

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "afc_dataset.csv"
        asb.write_afc_dataset(df, path)
        loaded = asb.read_afc_dataset(path)

    assert list(loaded.columns) == asb.AFC_DATASET_COLUMNS
    assert len(loaded) == len(df)
    assert np.allclose(loaded["CL"], df["CL"])


def test_afc_dataset_range_validation_catches_bad_c_mu():
    df = asb.make_fake_afc_dataset(n_cases=5, random_seed=5)
    df.loc[0, "C_mu"] = -0.1

    try:
        asb.validate_afc_dataframe(df)
    except asb.AFCDatasetValidationError:
        return

    raise AssertionError("Expected invalid C_mu to raise AFCDatasetValidationError.")


def test_afc_dataset_manifest_duplicates_and_grouped_statistics():
    df = asb.make_fake_afc_dataset(n_cases=12, random_seed=7, source="unit_test")
    manifest = asb.make_afc_dataset_manifest(
        df,
        dataset_id="unit_test_dataset",
        split_method="random_seed_7",
        filters={"converged_only": False},
        known_limitations=["synthetic data for unit tests"],
    )

    assert manifest["dataset_id"] == "unit_test_dataset"
    assert manifest["n_cases"] == len(df)
    assert "C_mu" in manifest["input_ranges"]

    duplicated_df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    duplicate_report = asb.check_duplicate_afc_samples(duplicated_df, decimals=10)
    assert duplicate_report["n_duplicate_rows"] == 2

    tables = asb.afc_dataset_standard_statistics_tables(df)
    assert "by_source" in tables
    assert int(tables["by_source"]["n_cases"].sum()) == len(df)

    with tempfile.TemporaryDirectory() as tmp:
        dataset_path = Path(tmp) / "afc_dataset.csv"
        manifest_path = Path(tmp) / "manifest.json"
        asb.write_afc_dataset(
            df,
            dataset_path,
            manifest=manifest,
            manifest_path=manifest_path,
        )
        loaded, loaded_manifest = asb.read_afc_dataset(
            dataset_path,
            manifest_path=manifest_path,
            return_manifest=True,
        )

    assert len(loaded) == len(df)
    assert loaded_manifest["dataset_id"] == "unit_test_dataset"


def test_flow_control_lab_naca0018_conversion_tool():
    with tempfile.TemporaryDirectory() as tmp:
        input_dir = Path(tmp) / "fcl"
        input_dir.mkdir()
        raw_path = input_dir / "data_#1.txt"
        raw_path.write_text(
            "\n".join(
                [
                    "measurement #1,  quasi-steady pitch,  Re=150000",
                    "time_[s]\tphase_angle_[deg]\tAoA_[deg]\tU_infty_[m/s]\tq_[Pa]\tCl\tCd\tCm\tCmu_[%]",
                    "0\t0\t-2\t6.62\t26.9\t-0.18\t0.095\t-0.007\t7.24",
                    "0.1\t2\t0\t6.63\t27.0\t0.02\t0.096\t-0.003\t7.25",
                ]
            ),
            encoding="utf-8",
        )

        output_path = Path(tmp) / "converted.csv"
        manifest_path = Path(tmp) / "converted.manifest.json"
        df = asb.convert_flow_control_lab_naca0018_directory(
            input_dir,
            output_path=output_path,
            manifest_path=manifest_path,
            x_jet=0.12,
            theta_jet=35.0,
        )

        manifest = asb.read_afc_dataset_manifest(manifest_path)
        output_exists = output_path.exists()

    assert len(df) == 2
    assert output_exists
    assert np.allclose(df["C_mu"], [0.0724, 0.0725])
    assert np.allclose(df["x_jet"], 0.12)
    assert np.allclose(df["theta_jet"], 35.0)
    assert manifest["dataset_id"] == "fcl_naca0018_quasi_steady_afc_schema"
    assert "assumptions" in manifest


if __name__ == "__main__":
    test_afc_dataset_schema_validation_split_and_csv_io()
    test_afc_dataset_range_validation_catches_bad_c_mu()
    test_afc_dataset_manifest_duplicates_and_grouped_statistics()
    test_flow_control_lab_naca0018_conversion_tool()
