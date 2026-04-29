from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Union

from aerosandbox.afc_system.network import AFCSystemNetwork


def read_afc_system_config(filepath: Union[str, Path]) -> Dict[str, object]:
    """
    Reads a JSON/YAML AFC system sizing config.

    YAML uses the lightweight AFC config reader already used by the AFC aerodynamics tools: PyYAML if installed, with a
    conservative built-in fallback for simple nested mappings.
    """
    from aerosandbox.aerodynamics.afc_config import read_afc_config

    return read_afc_config(filepath)


def afc_system_network_from_config(config: Mapping[str, object]) -> AFCSystemNetwork:
    network_config = dict(config.get("network", {}))
    component_configs = config.get("components", [])
    if not isinstance(component_configs, list):
        raise TypeError("AFC system config field `components` must be a list.")
    return AFCSystemNetwork.from_component_configs(
        component_configs,
        name=network_config.get("name", "afc_system"),
    )


def run_afc_system_config(config: Mapping[str, object]) -> Dict[str, object]:
    """
    Builds and runs an AFC system network from a config dictionary.
    """
    network = afc_system_network_from_config(config)
    operating = dict(config.get("operating_condition", {}))
    sizing = dict(config.get("sizing", {}))
    mode = str(sizing.get("mode", "design")).lower()

    common_kwargs = {
        "freestream_velocity": operating.get("freestream_velocity", 50.0),
        "S_ref": operating.get("S_ref", 10.0),
        "density_inf": operating.get("density_inf", None),
        "q_inf": operating.get("q_inf", None),
        "ambient_pressure": operating.get("ambient_pressure", 101325.0),
        "ambient_temperature": operating.get("ambient_temperature", 288.15),
    }

    if mode == "design":
        return network.design(
            **common_kwargs,
            target_C_mu=sizing.get("target_C_mu", None),
            target_mass_flow=sizing.get("target_mass_flow", None),
            jet_velocity=sizing.get("jet_velocity", None),
        )
    if mode in {"off_design", "off-design"}:
        return network.off_design(
            **common_kwargs,
            mass_flow=sizing.get("mass_flow", None),
            max_mass_flow=sizing.get("max_mass_flow", None),
        )
    raise ValueError('AFC system config `sizing.mode` must be "design" or "off_design".')


def run_afc_system_config_file(filepath: Union[str, Path]) -> Dict[str, object]:
    return run_afc_system_config(read_afc_system_config(filepath))

