from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import aerosandbox.numpy as np

from aerosandbox.afc_system.nodes import AFCSystemNode, _maybe_float


def circular_area(diameter: float):
    return np.pi * diameter**2 / 4


def flow_velocity_from_mass_flow(mass_flow, density, area):
    return mass_flow / np.maximum(density * area, 1e-12)


def flow_dynamic_pressure(mass_flow, density, area):
    velocity = flow_velocity_from_mass_flow(mass_flow, density, area)
    return 0.5 * density * velocity**2


@dataclass
class AFCComponent:
    """
    Base class for AFC system components.

    The first implementation uses steady, incompressible-style pressure loss placeholders where appropriate. These are
    meant for trends and architecture trade studies, not high-fidelity pneumatic design.
    """

    name: str
    fixed_weight: float = 0.0  # kg
    fixed_volume: float = 0.0  # m^3

    component_type: str = "component"

    def pressure_loss(self, inlet: AFCSystemNode, mass_flow: float) -> float:
        return 0.0

    def pressure_rise(
        self,
        inlet: AFCSystemNode,
        mass_flow: float,
        *,
        required_outlet_pressure: Optional[float] = None,
    ) -> float:
        return 0.0

    def power_required(
        self,
        inlet: AFCSystemNode,
        mass_flow: float,
        *,
        outlet_pressure: Optional[float] = None,
    ) -> float:
        return 0.0

    def estimated_weight(self) -> float:
        return self.fixed_weight

    def estimated_volume(self) -> float:
        return self.fixed_volume

    def result_dict(
        self,
        *,
        pressure_loss: float = 0.0,
        pressure_rise: float = 0.0,
        power_required: float = 0.0,
    ) -> Dict[str, object]:
        return {
            "type": self.component_type,
            "pressure_loss": _maybe_float(pressure_loss),
            "pressure_rise": _maybe_float(pressure_rise),
            "power_required": _maybe_float(power_required),
            "estimated_weight": _maybe_float(self.estimated_weight()),
            "estimated_volume": _maybe_float(self.estimated_volume()),
        }


@dataclass
class AFCInlet(AFCComponent):
    area: float = 0.01  # m^2
    loss_coefficient: float = 0.20
    pressure_recovery: float = 0.98

    component_type: str = "inlet"

    def pressure_loss(self, inlet: AFCSystemNode, mass_flow: float) -> float:
        q = flow_dynamic_pressure(mass_flow, inlet.density, self.area)
        recovery_loss = (1 - self.pressure_recovery) * inlet.pressure
        return recovery_loss + self.loss_coefficient * q


@dataclass
class AFCDuct(AFCComponent):
    length: float = 1.0  # m
    hydraulic_diameter: float = 0.05  # m
    friction_factor: float = 0.025
    loss_coefficient: float = 0.0
    wall_thickness: float = 0.001  # m
    material_density: float = 2700.0  # kg/m^3

    component_type: str = "duct"

    @property
    def area(self):
        return circular_area(self.hydraulic_diameter)

    def pressure_loss(self, inlet: AFCSystemNode, mass_flow: float) -> float:
        q = flow_dynamic_pressure(mass_flow, inlet.density, self.area)
        K = self.friction_factor * self.length / max(self.hydraulic_diameter, 1e-12)
        return (K + self.loss_coefficient) * q

    def estimated_weight(self) -> float:
        shell_volume = np.pi * self.hydraulic_diameter * self.length * self.wall_thickness
        return self.fixed_weight + self.material_density * shell_volume

    def estimated_volume(self) -> float:
        return self.fixed_volume + self.area * self.length


@dataclass
class AFCBend(AFCComponent):
    hydraulic_diameter: float = 0.05  # m
    angle_degrees: float = 90.0
    loss_coefficient_90deg: float = 0.45
    wall_thickness: float = 0.001  # m
    material_density: float = 2700.0  # kg/m^3

    component_type: str = "bend"

    @property
    def area(self):
        return circular_area(self.hydraulic_diameter)

    def pressure_loss(self, inlet: AFCSystemNode, mass_flow: float) -> float:
        q = flow_dynamic_pressure(mass_flow, inlet.density, self.area)
        return self.loss_coefficient_90deg * abs(self.angle_degrees) / 90 * q

    def estimated_weight(self) -> float:
        bend_length = np.pi * self.hydraulic_diameter * abs(self.angle_degrees) / 360
        shell_volume = np.pi * self.hydraulic_diameter * bend_length * self.wall_thickness
        return self.fixed_weight + self.material_density * shell_volume

    def estimated_volume(self) -> float:
        bend_length = np.pi * self.hydraulic_diameter * abs(self.angle_degrees) / 360
        return self.fixed_volume + self.area * bend_length


@dataclass
class AFCValve(AFCComponent):
    hydraulic_diameter: float = 0.05  # m
    loss_coefficient: float = 2.0
    opening_fraction: float = 1.0

    component_type: str = "valve"

    @property
    def area(self):
        return circular_area(self.hydraulic_diameter)

    def pressure_loss(self, inlet: AFCSystemNode, mass_flow: float) -> float:
        q = flow_dynamic_pressure(mass_flow, inlet.density, self.area)
        opening = np.maximum(self.opening_fraction, 1e-3)
        return self.loss_coefficient / opening**2 * q


@dataclass
class AFCCompressor(AFCComponent):
    pressure_ratio: Optional[float] = None
    isentropic_efficiency: float = 0.72
    motor_efficiency: float = 0.90
    gamma: float = 1.4
    gas_constant: float = 287.05  # J/(kg*K)

    component_type: str = "compressor"

    @property
    def cp(self):
        return self.gamma * self.gas_constant / (self.gamma - 1)

    def outlet_pressure(
        self,
        inlet: AFCSystemNode,
        *,
        required_outlet_pressure: Optional[float] = None,
    ):
        if required_outlet_pressure is not None:
            return required_outlet_pressure
        if self.pressure_ratio is None:
            return inlet.pressure
        return inlet.pressure * self.pressure_ratio

    def pressure_rise(
        self,
        inlet: AFCSystemNode,
        mass_flow: float,
        *,
        required_outlet_pressure: Optional[float] = None,
    ) -> float:
        return self.outlet_pressure(
            inlet,
            required_outlet_pressure=required_outlet_pressure,
        ) - inlet.pressure

    def power_required(
        self,
        inlet: AFCSystemNode,
        mass_flow: float,
        *,
        outlet_pressure: Optional[float] = None,
    ) -> float:
        if outlet_pressure is None:
            outlet_pressure = self.outlet_pressure(inlet)
        pressure_ratio = np.maximum(outlet_pressure / np.maximum(inlet.pressure, 1e-12), 1.0)
        ideal_temperature_ratio = pressure_ratio ** ((self.gamma - 1) / self.gamma)
        shaft_power = (
            mass_flow
            * self.cp
            * inlet.temperature
            * (ideal_temperature_ratio - 1)
            / max(self.isentropic_efficiency * self.motor_efficiency, 1e-6)
        )
        return np.maximum(shaft_power, 0.0)


@dataclass
class AFCPlenum(AFCComponent):
    volume: float = 0.01  # m^3
    loss_coefficient: float = 0.05
    reference_area: float = 0.01  # m^2
    mass_per_volume: float = 45.0  # kg/m^3 placeholder packaging estimate

    component_type: str = "plenum"

    def pressure_loss(self, inlet: AFCSystemNode, mass_flow: float) -> float:
        q = flow_dynamic_pressure(mass_flow, inlet.density, self.reference_area)
        return self.loss_coefficient * q

    def estimated_weight(self) -> float:
        return self.fixed_weight + self.mass_per_volume * self.volume

    def estimated_volume(self) -> float:
        return self.fixed_volume + self.volume


@dataclass
class AFCSteadyJetActuator(AFCComponent):
    """
    Steady slot/jet actuator placeholder.

    The pressure requirement uses `Delta p = 0.5 rho (V_jet / C_d)^2`. This is a first-pass engineering estimate and
    should be replaced by calibrated actuator/nozzle data when available.
    """

    jet_area: float = 1e-3  # m^2 total slot/jet area
    discharge_coefficient: float = 0.85
    jet_velocity: Optional[float] = None  # m/s
    jet_velocity_ratio: float = 2.0  # V_jet / V_inf if jet_velocity is not specified
    packaging_density: float = 80.0  # kg/m^3 placeholder
    packaging_volume_factor: float = 8.0

    component_type: str = "actuator"

    def resolve_jet_velocity(
        self,
        *,
        freestream_velocity: float,
        mass_flow: Optional[float] = None,
        density: Optional[float] = None,
    ):
        if self.jet_velocity is not None:
            return self.jet_velocity
        if mass_flow is not None and density is not None:
            return mass_flow / np.maximum(density * self.jet_area, 1e-12)
        return self.jet_velocity_ratio * freestream_velocity

    def required_pressure_drop(
        self,
        *,
        jet_velocity: float,
        density: float,
    ):
        return 0.5 * density * (jet_velocity / max(self.discharge_coefficient, 1e-6)) ** 2

    def pressure_loss(self, inlet: AFCSystemNode, mass_flow: float) -> float:
        jet_velocity = self.resolve_jet_velocity(
            freestream_velocity=0.0,
            mass_flow=mass_flow,
            density=inlet.density,
        )
        return self.required_pressure_drop(
            jet_velocity=jet_velocity,
            density=inlet.density,
        )

    def estimated_weight(self) -> float:
        return self.fixed_weight + self.packaging_density * self.estimated_volume()

    def estimated_volume(self) -> float:
        return self.fixed_volume + self.packaging_volume_factor * self.jet_area ** 1.5


COMPONENT_TYPE_MAP = {
    "inlet": AFCInlet,
    "duct": AFCDuct,
    "bend": AFCBend,
    "valve": AFCValve,
    "compressor": AFCCompressor,
    "plenum": AFCPlenum,
    "actuator": AFCSteadyJetActuator,
    "steady_jet_actuator": AFCSteadyJetActuator,
}


def component_from_config(config: Dict[str, object]) -> AFCComponent:
    config = dict(config)
    component_type = str(config.pop("type", config.pop("component_type", ""))).lower()
    if component_type not in COMPONENT_TYPE_MAP:
        raise ValueError(
            f'Unknown AFC system component type "{component_type}". '
            f"Available types: {sorted(COMPONENT_TYPE_MAP)}"
        )
    return COMPONENT_TYPE_MAP[component_type](**config)

