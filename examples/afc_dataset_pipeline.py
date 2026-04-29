"""
Build and inspect a placeholder AFC 2D surrogate dataset.

This example does not call CFD. It demonstrates the common schema expected for future SU2/RANS/modified-XFoil/experiment
data, along with validation, IO, splitting, and basic statistics.

Run from the repository root:

    .venv-aerosandbox/bin/python examples/afc_dataset_pipeline.py
"""

from pathlib import Path

import aerosandbox as asb


output_dir = Path(__file__).parent / "generated_afc_dataset"
csv_path = output_dir / "afc_airfoil_synthetic.csv"
parquet_path = output_dir / "afc_airfoil_synthetic.parquet"

df = asb.make_fake_afc_dataset(
    n_cases=120,
    random_seed=1,
    source="synthetic_placeholder",
)

asb.validate_afc_dataframe(df)

asb.write_afc_dataset(df, csv_path)
df_from_csv = asb.read_afc_dataset(csv_path)

parquet_written = False
try:
    asb.write_afc_dataset(df, parquet_path)
    df_from_parquet = asb.read_afc_dataset(parquet_path)
    parquet_written = len(df_from_parquet) == len(df)
except ImportError as e:
    print(f"Parquet skipped: {e}")

train, val, test = asb.split_afc_dataset(
    df_from_csv,
    train_fraction=0.70,
    val_fraction=0.15,
    test_fraction=0.15,
    random_seed=2,
)

report = asb.afc_dataset_statistics(df_from_csv)

print("AFC 2D dataset pipeline")
print("=======================")
print(f"canonical columns      : {len(asb.AFC_DATASET_COLUMNS)}")
print(f"rows                   : {report['n_cases']}")
print(f"converged fraction     : {report['converged_fraction']:.3f}")
print(f"sources                : {report['sources']}")
print(f"confidence mean        : {report['confidence_label_mean']:.3f}")
print(f"split sizes            : train={len(train)}, val={len(val)}, test={len(test)}")
print(f"CSV written            : {csv_path}")
print(f"Parquet written        : {parquet_path if parquet_written else 'not available'}")

print("\nSelected numeric summary")
print(report["numeric_summary"].loc[["alpha", "Re", "Mach", "C_mu", "CL", "CD", "CM"]])
