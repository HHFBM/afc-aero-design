from pathlib import Path

import aerosandbox as asb


REPO_ROOT = Path(__file__).resolve().parents[3]


def make_test_network(duct_length: float = 1.0) -> asb.AFCSystemNetwork:
    return asb.AFCSystemNetwork(
        components=[
            asb.AFCInlet(name="inlet", area=0.02),
            asb.AFCCompressor(name="compressor", pressure_ratio=1.8),
            asb.AFCDuct(
                name="duct",
                length=duct_length,
                hydraulic_diameter=0.05,
            ),
            asb.AFCValve(name="valve", hydraulic_diameter=0.05),
            asb.AFCPlenum(name="plenum", volume=0.01, reference_area=0.01),
            asb.AFCSteadyJetActuator(
                name="actuator",
                jet_area=0.002,
                jet_velocity_ratio=2.0,
            ),
        ]
    )


def test_required_power_increases_with_target_c_mu():
    network = make_test_network()
    low = network.design(
        freestream_velocity=50.0,
        density_inf=1.225,
        S_ref=10.0,
        target_C_mu=0.01,
    )
    high = network.design(
        freestream_velocity=50.0,
        density_inf=1.225,
        S_ref=10.0,
        target_C_mu=0.05,
    )

    assert high["mass_flow"] > low["mass_flow"]
    assert high["required_power"] > low["required_power"]


def test_duct_length_increases_pressure_loss():
    short = make_test_network(duct_length=0.5).design(
        freestream_velocity=50.0,
        density_inf=1.225,
        S_ref=10.0,
        target_C_mu=0.03,
    )
    long = make_test_network(duct_length=3.0).design(
        freestream_velocity=50.0,
        density_inf=1.225,
        S_ref=10.0,
        target_C_mu=0.03,
    )

    assert long["component_pressure_losses"]["duct"] > short["component_pressure_losses"]["duct"]


def test_afc_system_output_fields_and_config_loading():
    result = asb.run_afc_system_config_file(
        REPO_ROOT / "configs" / "afc" / "afc_system_config.example.yaml"
    )

    for field in [
        "node_states",
        "component_pressure_losses",
        "mass_flow",
        "jet_velocity",
        "required_power",
        "estimated_weight",
        "estimated_volume",
        "total_SWAP",
        "C_mu",
    ]:
        assert field in result

    assert result["mass_flow"] > 0
    assert result["jet_velocity"] > 0
    assert result["required_power"] > 0
    assert result["estimated_weight"] > 0
    assert result["estimated_volume"] > 0


def test_off_design_calculates_achievable_c_mu():
    network = make_test_network()
    result = network.off_design(
        freestream_velocity=50.0,
        density_inf=1.225,
        S_ref=10.0,
    )
    assert result["mode"] == "off_design"
    assert result["mass_flow"] >= 0
    assert result["C_mu"] >= 0


if __name__ == "__main__":
    test_required_power_increases_with_target_c_mu()
    test_duct_length_increases_pressure_loss()
    test_afc_system_output_fields_and_config_loading()
    test_off_design_calculates_achievable_c_mu()

