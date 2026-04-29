# AFC System SWaP Prototype

This package is the first-pass FR-SYS engineering prototype for active-flow-control system sizing.

It is a trend-level model, not a high-fidelity pneumatic or thermal solver. Component correlations are placeholders
intended to be replaced by calibrated compressor maps, valve data, duct loss data, actuator tests, and thermal models.

## Modules

- `nodes.py`: node thermodynamic states: pressure, temperature, mass flow, Mach.
- `components.py`: inlet, duct, bend, valve, compressor, plenum, steady jet actuator.
- `network.py`: serial node-component network with design and off-design modes.
- `sizing.py`: `C_mu` and mass-flow conversion utilities.
- `configs.py`: JSON/YAML config loading and execution helpers.

## C_mu Convention

The current system-level convention is:

```text
C_mu = mdot * V_jet / (q_inf * S_ref)
```

For 2D section studies, use a unit-span reference area:

```text
S_ref = chord * 1 m
mdot = kg/s per meter span
```

For finite-wing studies, use the controlled wing reference area and total actuator mass flow.

## Run Example

```bash
.venv-aerosandbox/bin/python examples/afc_system_sizing.py
```

Or run from a config file:

```python
import aerosandbox as asb

result = asb.run_afc_system_config_file("configs/afc/afc_system_config.example.yaml")
```

