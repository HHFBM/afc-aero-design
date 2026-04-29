"""
First-pass AFC steady-jet system SWaP sizing example.

Run:
    .venv-aerosandbox/bin/python examples/afc_system_sizing.py
"""

import aerosandbox as asb


network = asb.AFCSystemNetwork(
    name="rectangular_wing_steady_jet_afc",
    components=[
        asb.AFCInlet(name="inlet", area=0.018, pressure_recovery=0.98),
        asb.AFCCompressor(
            name="compressor",
            pressure_ratio=1.8,
            fixed_weight=4.0,
            fixed_volume=0.018,
        ),
        asb.AFCDuct(
            name="fuselage_to_wing_duct",
            length=2.5,
            hydraulic_diameter=0.08,
        ),
        asb.AFCBend(name="wing_root_bend", hydraulic_diameter=0.08),
        asb.AFCValve(
            name="control_valve",
            hydraulic_diameter=0.08,
            opening_fraction=0.85,
            fixed_weight=0.5,
            fixed_volume=0.002,
        ),
        asb.AFCPlenum(name="wing_plenum", volume=0.018, reference_area=0.02),
        asb.AFCSteadyJetActuator(
            name="steady_slot_actuator",
            jet_area=0.003,
            discharge_coefficient=0.85,
            jet_velocity_ratio=2.2,
            fixed_weight=0.4,
            fixed_volume=0.001,
        ),
    ],
)


result = network.design(
    freestream_velocity=55.0,
    density_inf=1.225,
    S_ref=12.0,
    target_C_mu=0.005,
)

print("AFC steady-jet system sizing")
print(f"C_mu             : {result['C_mu']:.4f}")
print(f"mass_flow        : {result['mass_flow']:.4f} kg/s")
print(f"jet_velocity     : {result['jet_velocity']:.1f} m/s")
print(f"required_power   : {result['required_power'] / 1000:.2f} kW")
print(f"estimated_weight : {result['estimated_weight']:.2f} kg")
print(f"estimated_volume : {result['estimated_volume']:.4f} m^3")

print("\nComponent pressure losses:")
for name, pressure_loss in result["component_pressure_losses"].items():
    print(f"  {name:24s} {pressure_loss:10.1f} Pa")

print("\nConfig-file equivalent:")
config_result = asb.run_afc_system_config_file("configs/afc/afc_system_config.example.yaml")
print(f"  config C_mu          : {config_result['C_mu']:.4f}")
print(f"  config required_power: {config_result['required_power'] / 1000:.2f} kW")
