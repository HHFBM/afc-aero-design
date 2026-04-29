# AFC Validation Data Interface

This directory contains lightweight benchmark metadata only.

Do not commit large CFD, RANS, wind-tunnel, or proprietary validation datasets here. Store those datasets outside the
source tree and pass them to the validation report script with:

```bash
python examples/afc_validation_report.py \
  --config configs/afc/validation_config.example.yaml \
  --external-2d-dataset /path/to/dataset.parquet
```

External datasets must follow the canonical schema in:

```text
aerosandbox/aerodynamics/aero_2D/afc_dataset.py
```

If available, place the dataset manifest alongside the dataset as one of:

```text
dataset.manifest.json
dataset.parquet -> dataset.manifest.json
dataset.csv -> dataset.manifest.json
```

The formal acceptance report records whether each section is based on:

- real external validation data
- synthetic placeholder data
- internal solver-to-solver regression
- placeholder SWaP / integrated workflow trends
