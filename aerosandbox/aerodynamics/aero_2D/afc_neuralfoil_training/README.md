# AFC-NeuralFoil PyTorch Training Skeleton

This directory contains the first-pass training pipeline for the AFC-NeuralFoil MLP. It is an engineering scaffold, not a tuned production model.

## Inputs

Training reads the flat AFC dataset schema defined in `aerosandbox.aerodynamics.aero_2D.afc_dataset`. Use `.parquet` when a Parquet engine is installed, or `.csv` for small/debug runs.

Required physical data columns include:

- Kulfan geometry: `kulfan_upper_weights_0..7`, `kulfan_lower_weights_0..7`, `leading_edge_weight`, `TE_thickness`
- Flow and transition inputs: `alpha`, `Re`, `Mach`, `n_crit`, `xtr_upper`, `xtr_lower`
- AFC inputs: `C_mu`, `x_jet`, `theta_jet`
- Outputs: `CL`, `CD`, `CM`, `Top_Xtr`, `Bot_Xtr`
- Metadata: `converged`, `source`, `confidence_label`

The current runtime inference convention predicts AFC corrections to a NeuralFoil baseline. During training, the script adds these baseline columns if they are not already present:

- `CL_baseline`, `CD_baseline`, `CM_baseline`
- `Top_Xtr_baseline`, `Bot_Xtr_baseline`
- `analysis_confidence_baseline`

For large datasets, precompute and store these columns once instead of recomputing them every training run.

## Run

Create or point to a dataset first. The existing synthetic data example can make a small CSV:

```bash
.venv-aerosandbox/bin/python examples/afc_dataset_pipeline.py
```

Install PyTorch in the training environment, then launch a small run:

```bash
.venv-aerosandbox/bin/python -m aerosandbox.aerodynamics.aero_2D.afc_neuralfoil_training.train_afc_neuralfoil \
  --config aerosandbox/aerodynamics/aero_2D/afc_neuralfoil_training/config_example.json
```

Useful overrides:

```bash
.venv-aerosandbox/bin/python -m aerosandbox.aerodynamics.aero_2D.afc_neuralfoil_training.train_afc_neuralfoil \
  --config aerosandbox/aerodynamics/aero_2D/afc_neuralfoil_training/config_example.json \
  --data path/to/afc_dataset.parquet \
  --output-directory runs/afc_neuralfoil/medium_001 \
  --model-size medium \
  --epochs 200
```

## What The Trainer Does

- Builds the same encoded inputs used by the `aerosandbox.numpy` inference path:
  - Kulfan weights
  - `sin/cos` angle encodings for `alpha` and `theta_jet`
  - log-scaled `Re`
  - AFC fields `C_mu`, `x_jet`, `theta_jet`
- Fits input and output normalization on the training split.
- Trains an MLP with Swish activations.
- Uses Huber losses on `CL`, `logCD`, and `CM`.
- Adds small Huber losses on transition outputs.
- Uses BCE on `confidence_label`.
- Supports AdamW or RAdam, early stopping, `ReduceLROnPlateau`, and checkpoints.

## Outputs

The run directory contains:

- `resolved_config.json`
- `checkpoint_latest.pt`
- `checkpoint_best.pt`
- `history.json`
- `model_card.md`
- `test_predictions.csv`
- `test_error_summary.csv`
- `prediction_vs_truth/*.png`
- `afc_confidence_training_statistics.json`
- `exported_weights/nn-{model_size}.npz`
- `exported_weights/nn-{model_size}.json`

The `.npz` contains only NeuralFoil-style layer arrays:

- `net.0.weight`, `net.0.bias`
- `net.2.weight`, `net.2.bias`
- ...

Input/output normalizers are folded into the first and last linear layers during export, so the runtime inference module can evaluate the `.npz` directly with `aerosandbox.numpy`.

`afc_confidence_training_statistics.json` stores lightweight feature means/stds used by the runtime OOD heuristic.
Pass it as `training_statistics_path` to the AFC-NeuralFoil inference functions to enable training-distance risk in
`ood_score`. This confidence path is a heuristic applicability signal, not strict uncertainty quantification.

## Manual Export

To export a checkpoint manually:

```bash
.venv-aerosandbox/bin/python -m aerosandbox.aerodynamics.aero_2D.afc_neuralfoil_training.export_weights \
  runs/afc_neuralfoil/small_debug/checkpoint_best.pt \
  --output-directory aerosandbox/aerodynamics/aero_2D/afc_neuralfoil_weights \
  --model-size small \
  --overwrite
```

The exported weights are then loadable by:

```python
from aerosandbox.aerodynamics.aero_2D.afc_neuralfoil import get_aero_from_afc_kulfan_parameters
```

## Public Flow Control Lab NACA0018 Data

Raw Flow Control Lab files should stay outside the source tree. Convert an extracted FCL pressure-data directory into
the canonical AFC schema with:

```bash
.venv-aerosandbox/bin/python scripts/afc_convert_flowcontrollab_naca0018.py \
  --input-directory /path/to/FCL_pressure_data \
  --output /path/to/fcl_naca0018_quasi_steady_afc_schema.csv \
  --manifest /path/to/fcl_naca0018_quasi_steady_manifest.json
```

The converter records the fixed assumptions for `x_jet`, `theta_jet`, `Top_Xtr`, and `Bot_Xtr` in the manifest because
those fields are not provided as per-sample labels in the public text tables.

Train a large model from that converted dataset with:

```bash
.venv-aerosandbox/bin/python -m aerosandbox.aerodynamics.aero_2D.afc_neuralfoil_training.train_afc_neuralfoil \
  --config aerosandbox/aerodynamics/aero_2D/afc_neuralfoil_training/config_example.json \
  --data /path/to/fcl_naca0018_quasi_steady_afc_schema.csv \
  --output-directory runs/afc_neuralfoil/fcl_naca0018_large \
  --model-size large \
  --epochs 300
```
