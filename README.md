# afc-aero-design

`afc-aero-design` is an engineering prototype for active flow control (AFC) conceptual design built on top of AeroSandbox.

The project connects four layers into one workflow:

1. `2D AFC surrogate`
2. `3D finite-wing analysis`
3. `jet-system SWaP estimation`
4. `end-to-end optimization and validation`

The goal is not to replace a CFD solver. The goal is to reduce how often expensive CFD has to be run during concept design, while keeping a clean path for adding new CFD or experimental data back into the model.

## What the project does

### 2D AFC airfoil analysis

The 2D model predicts aerodynamic section quantities from:

- airfoil geometry
- `alpha`
- `Re`
- `Mach`
- `C_mu`
- `x_jet`
- `theta_jet`

Current outputs include:

- `CL`
- `CD`
- `CM`
- `Top_Xtr`
- `Bot_Xtr`
- `analysis_confidence`
- `ood_score`
- `stall_risk`
- `confidence_reason`

Core files:

- [`aerosandbox/aerodynamics/aero_2D/afc_neuralfoil.py`](./aerosandbox/aerodynamics/aero_2D/afc_neuralfoil.py)
- [`aerosandbox/aerodynamics/aero_2D/afc_confidence.py`](./aerosandbox/aerodynamics/aero_2D/afc_confidence.py)
- [`aerosandbox/aerodynamics/aero_2D/afc_dataset.py`](./aerosandbox/aerodynamics/aero_2D/afc_dataset.py)

### 3D finite-wing AFC analysis

The project extends the 2D section model to finite-wing analysis using AeroSandbox 3D solvers.

Supported 3D paths:

- `AFCNeuralFoilLiftingLine`
- `AFCNeuralFoilNonlinearLiftingLine`
- `NeuralFoilCamberedVLM`

3D outputs include:

- `CL`, `CD`, `CDi`, `CDp`, `CM`
- `CY`, `Cl`, `Cn` where available
- `spanwise_y`
- `section_cl`, `section_cd`, `section_cm`
- `local_alpha`, `local_Re`, `local_C_mu`
- `converged`
- `failure_reason`
- convergence history and residual diagnostics

Core files:

- [`aerosandbox/aerodynamics/aero_3D/afc_lifting_line.py`](./aerosandbox/aerodynamics/aero_3D/afc_lifting_line.py)
- [`aerosandbox/aerodynamics/aero_3D/afc_nonlinear_lifting_line.py`](./aerosandbox/aerodynamics/aero_3D/afc_nonlinear_lifting_line.py)
- [`aerosandbox/aerodynamics/aero_3D/afc_cambered_vlm.py`](./aerosandbox/aerodynamics/aero_3D/afc_cambered_vlm.py)

### Jet-system SWaP estimation

The AFC system module estimates first-pass system penalties for jet-based active flow control:

- required mass flow
- required power
- estimated weight
- estimated volume
- component-level breakdown

Core files:

- [`aerosandbox/afc_system/network.py`](./aerosandbox/afc_system/network.py)
- [`aerosandbox/afc_system/components.py`](./aerosandbox/afc_system/components.py)
- [`aerosandbox/afc_system/sizing.py`](./aerosandbox/afc_system/sizing.py)

### End-to-end optimization

The optimization examples connect 2D AFC, 3D wing analysis, SWaP, and `asb.Opti` into one loop.

Representative design variables:

- tip chord
- half-span or aspect ratio
- linear twist
- spanwise `C_mu` control segments
- optional jet-system variables

Representative constraints:

- target `CL`
- span limit
- root bending moment
- confidence threshold
- power limit
- weight limit

Main example:

- [`examples/afc_concept_design_optimization.py`](./examples/afc_concept_design_optimization.py)

## Current repository structure

```text
afc-aero-design/
├── aerosandbox/
│   ├── aerodynamics/
│   │   ├── aero_2D/
│   │   ├── aero_3D/
│   │   ├── afc_config.py
│   │   └── afc_validation.py
│   └── afc_system/
├── configs/afc/
├── docs/
├── examples/
├── external_data/
└── scripts/
```

## Data used right now

The current main training dataset is:

- [`external_data/flowcontrollab/converted/fcl_naca0018_quasi_steady_afc_schema_with_baseline.csv`](./external_data/flowcontrollab/converted/fcl_naca0018_quasi_steady_afc_schema_with_baseline.csv)

Key facts:

- source: `Flow Control Lab`
- primary geometry: `NACA0018`
- size: `1656` samples
- role: current AFC-NeuralFoil training and regression data

There is also a lightweight synthetic dataset for smoke tests and pipeline verification:

- [`examples/generated_afc_dataset/afc_airfoil_synthetic.csv`](./examples/generated_afc_dataset/afc_airfoil_synthetic.csv)

## CFD and benchmark role

This repository treats CFD as a data-generation and validation backend, not as the day-to-day analysis engine.

Current status:

- active-learning can recommend the next batch of CFD points
- task lists can be exported as CSV and JSONL
- a local SU2 installation can be used to generate small benchmark cases
- benchmark truth can be compared against the surrogate to compute loss

Important:

- the custom benchmark currently evaluates `baseline C_mu = 0` capability
- it does not yet represent a full AFC jet-enabled CFD dataset

## Validation and reporting

The project includes a formal validation report generator.

Outputs:

- `report.md`
- `report.json`
- figures directory

Main entry point:

- [`examples/afc_validation_report.py`](./examples/afc_validation_report.py)

Config:

- [`configs/afc/validation_config.example.yaml`](./configs/afc/validation_config.example.yaml)

The validation framework records:

- code version / git commit
- model weights path
- dataset manifest path
- config path
- runtime
- pass/fail criteria

## Installation

### Recommended Python environment

Use a dedicated virtual environment:

```bash
python3 -m venv .venv-aerosandbox
source .venv-aerosandbox/bin/activate
pip install -r requirements.txt
pip install -e .
```

If you want plotting in headless mode, use:

```bash
export MPLBACKEND=Agg
export MPLCONFIGDIR=/tmp/mpl
export XDG_CACHE_HOME=/tmp/xdg
```

### Optional CFD tooling

CFD is optional for most repository workflows.

If you want local SU2 benchmarks:

- install SU2 separately
- install `gmsh` separately if you want custom mesh generation

These tools are not required for the standard optimization and surrogate workflows.

## Quick start

### 1. Run 2D AFC inference

```python
import aerosandbox as asb
from pathlib import Path

root = Path(".")
airfoil = asb.Airfoil("naca0018")

aero = airfoil.get_aero_from_afc_neuralfoil(
    alpha=6,
    Re=150000,
    mach=0.02,
    C_mu=0.04,
    x_jet=0.10,
    theta_jet=30.0,
    model_size="medium",
    weights_directory=root / "runs/afc_neuralfoil/retrain_20260423_fcl_medium/exported_weights",
    training_statistics_path=root / "runs/afc_neuralfoil/retrain_20260423_fcl_medium/afc_confidence_training_statistics.json",
)
```

### 2. Generate the next batch of CFD tasks

```bash
.venv-aerosandbox/bin/python examples/afc_active_learning_cfd_tasks.py
```

### 3. Run end-to-end concept design optimization

```bash
MPLBACKEND=Agg .venv-aerosandbox/bin/python examples/afc_concept_design_optimization.py
```

### 4. Generate a validation report

```bash
MPLBACKEND=Agg .venv-aerosandbox/bin/python \
  examples/afc_validation_report.py \
  --config configs/afc/validation_config.example.yaml
```

## Main examples

- [`examples/afc_dataset_pipeline.py`](./examples/afc_dataset_pipeline.py)
- [`examples/afc_active_learning_cfd_tasks.py`](./examples/afc_active_learning_cfd_tasks.py)
- [`examples/afc_lifting_line_comparison.py`](./examples/afc_lifting_line_comparison.py)
- [`examples/afc_cambered_vlm_comparison.py`](./examples/afc_cambered_vlm_comparison.py)
- [`examples/afc_system_sizing.py`](./examples/afc_system_sizing.py)
- [`examples/afc_concept_design_optimization.py`](./examples/afc_concept_design_optimization.py)
- [`examples/afc_validation_report.py`](./examples/afc_validation_report.py)

## Known limitations

This project is currently an engineering prototype, not a production-grade AFC design system.

Main limitations:

- the 2D AFC dataset is still narrow
- the primary real dataset is dominated by `NACA0018`
- confidence is heuristic, not calibrated uncertainty quantification
- parts of the SWaP model are still placeholder or first-pass engineering estimates
- the current custom SU2 benchmark is baseline-only and inviscid
- full AFC jet CFD data generation is still under construction

## Recommended next steps

The most valuable next development steps are:

1. expand 2D AFC CFD data coverage
2. build a repeatable SU2 AFC case generator
3. ingest CFD truth back into the canonical AFC dataset schema
4. retrain with geometry holdout splits
5. validate against external CFD and experiments
6. calibrate confidence against real error

## Repository positioning

This repository should be understood as:

`AFC conceptual design and data-closure platform`

It is not:

- a replacement for SU2
- a production-ready uncertainty-quantified CFD system
- a finalized certified engineering workflow

It is:

- a fast AFC design workflow
- a surrogate-driven optimization environment
- a framework for integrating new CFD and experimental data

## Attribution

This work builds on the AeroSandbox codebase and extends it with AFC-specific 2D, 3D, SWaP, optimization, validation, and data-closure components.
