from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import aerosandbox.numpy as np

from aerosandbox.afc_system.components import (
    AFCComponent,
    AFCCompressor,
    AFCSteadyJetActuator,
    component_from_config,
)
from aerosandbox.afc_system.nodes import AFCSystemNode, _maybe_float
from aerosandbox.afc_system.sizing import (
    c_mu_from_mass_flow,
    dynamic_pressure_from_density_velocity,
    mass_flow_from_c_mu,
)


@dataclass
class AFCSystemNetwork:
    """
    Serial node-component AFC/SWaP network.

    This first version intentionally supports a simple serial topology. It still exposes node states and per-component
    results so later work can replace individual component models or generalize to branched networks.
    """

    components: List[AFCComponent] = field(default_factory=list)
    name: str = "afc_system"

    @classmethod
    def from_component_configs(
        cls,
        component_configs: Iterable[Dict[str, object]],
        *,
        name: str = "afc_system",
    ) -> "AFCSystemNetwork":
        return cls(
            components=[component_from_config(config) for config in component_configs],
            name=name,
        )

    @property
    def actuator(self) -> AFCSteadyJetActuator:
        actuators = [
            component
            for component in self.components
            if isinstance(component, AFCSteadyJetActuator)
        ]
        if len(actuators) == 0:
            raise ValueError("AFC system network must include a steady jet actuator.")
        return actuators[-1]

    @property
    def compressor_index(self) -> Optional[int]:
        for i, component in enumerate(self.components):
            if isinstance(component, AFCCompressor):
                return i
        return None

    def _passive_downstream_pressure_loss_estimate(
        self,
        *,
        start_index: int,
        stop_before_actuator: bool,
        reference_node: AFCSystemNode,
        mass_flow: float,
    ) -> float:
        loss = 0.0
        for component in self.components[start_index:]:
            if isinstance(component, AFCSteadyJetActuator) and stop_before_actuator:
                break
            if isinstance(component, AFCCompressor):
                continue
            if isinstance(component, AFCSteadyJetActuator):
                continue
            loss = loss + component.pressure_loss(reference_node, mass_flow)
        return loss

    def _evaluate_with_mass_flow(
        self,
        *,
        mass_flow: float,
        freestream_velocity: float,
        S_ref: float,
        density_inf: Optional[float] = None,
        q_inf: Optional[float] = None,
        ambient_pressure: float = 101325.0,
        ambient_temperature: float = 288.15,
        compressor_required_outlet_pressure: Optional[float] = None,
        jet_velocity: Optional[float] = None,
        mode: str,
    ) -> Dict[str, object]:
        if density_inf is None:
            density_inf = ambient_pressure / (287.05 * ambient_temperature)
        if q_inf is None:
            q_inf = dynamic_pressure_from_density_velocity(
                density=density_inf,
                velocity=freestream_velocity,
            )

        actuator = self.actuator
        if jet_velocity is None:
            jet_velocity = actuator.resolve_jet_velocity(
                freestream_velocity=freestream_velocity,
                mass_flow=mass_flow,
                density=density_inf,
            )

        current = AFCSystemNode(
            name="ambient",
            pressure=ambient_pressure,
            temperature=ambient_temperature,
            mass_flow=mass_flow,
            Mach=0.0,
        )
        node_states = {current.name: current.to_dict()}
        component_results: Dict[str, Dict[str, object]] = {}
        component_pressure_losses: Dict[str, object] = {}

        required_power = 0.0
        actuator_inlet_pressure = None
        actuator_required_pressure_drop = None

        for i, component in enumerate(self.components):
            if isinstance(component, AFCCompressor):
                outlet_pressure = component.outlet_pressure(
                    current,
                    required_outlet_pressure=compressor_required_outlet_pressure,
                )
                pressure_rise = component.pressure_rise(
                    current,
                    mass_flow,
                    required_outlet_pressure=compressor_required_outlet_pressure,
                )
                power = component.power_required(
                    current,
                    mass_flow,
                    outlet_pressure=outlet_pressure,
                )
                pressure_ratio = outlet_pressure / np.maximum(current.pressure, 1e-12)
                ideal_temperature_ratio = pressure_ratio ** (
                    (component.gamma - 1) / component.gamma
                )
                outlet_temperature = current.temperature * (
                    1
                    + (ideal_temperature_ratio - 1)
                    / max(component.isentropic_efficiency, 1e-6)
                )
                required_power = required_power + power
                result = component.result_dict(
                    pressure_loss=0.0,
                    pressure_rise=pressure_rise,
                    power_required=power,
                )
                result["pressure_ratio"] = _maybe_float(pressure_ratio)
                current = current.copy_with(
                    name=f"node_{i + 1}_after_{component.name}",
                    pressure=outlet_pressure,
                    temperature=outlet_temperature,
                )

            elif isinstance(component, AFCSteadyJetActuator):
                actuator_inlet_pressure = current.pressure
                actuator_required_pressure_drop = component.required_pressure_drop(
                    jet_velocity=jet_velocity,
                    density=density_inf,
                )
                available_pressure_drop = current.pressure - ambient_pressure
                result = component.result_dict(
                    pressure_loss=actuator_required_pressure_drop,
                    pressure_rise=0.0,
                    power_required=0.0,
                )
                result["available_pressure_drop"] = _maybe_float(available_pressure_drop)
                result["available_inlet_pressure"] = _maybe_float(current.pressure)
                result["jet_area"] = _maybe_float(component.jet_area)
                result["discharge_coefficient"] = _maybe_float(
                    component.discharge_coefficient
                )
                current = current.copy_with(
                    name=f"node_{i + 1}_after_{component.name}",
                    pressure=ambient_pressure,
                    Mach=jet_velocity / current.speed_of_sound,
                )

            else:
                pressure_loss = component.pressure_loss(current, mass_flow)
                outlet_pressure = np.maximum(current.pressure - pressure_loss, 1.0)
                result = component.result_dict(
                    pressure_loss=pressure_loss,
                    pressure_rise=0.0,
                    power_required=0.0,
                )
                current = current.copy_with(
                    name=f"node_{i + 1}_after_{component.name}",
                    pressure=outlet_pressure,
                )

            component_results[component.name] = result
            component_pressure_losses[component.name] = result["pressure_loss"]
            node_states[current.name] = current.to_dict()

        C_mu = c_mu_from_mass_flow(
            mass_flow=mass_flow,
            jet_velocity=jet_velocity,
            q_inf=q_inf,
            S_ref=S_ref,
        )
        estimated_weight = sum(component.estimated_weight() for component in self.components)
        estimated_volume = sum(component.estimated_volume() for component in self.components)

        total_SWAP = {
            "required_power": _maybe_float(required_power),
            "estimated_weight": _maybe_float(estimated_weight),
            "estimated_volume": _maybe_float(estimated_volume),
            "score": _maybe_float(required_power + 100 * estimated_weight + 1e4 * estimated_volume),
            "score_note": "Placeholder weighted score: W + 100*kg + 1e4*m^3.",
        }

        return {
            "mode": mode,
            "network_name": self.name,
            "node_states": node_states,
            "component_pressure_losses": component_pressure_losses,
            "component_results": component_results,
            "mass_flow": _maybe_float(mass_flow),
            "jet_velocity": _maybe_float(jet_velocity),
            "C_mu": _maybe_float(C_mu),
            "q_inf": _maybe_float(q_inf),
            "S_ref": _maybe_float(S_ref),
            "actuator_inlet_pressure": _maybe_float(actuator_inlet_pressure),
            "actuator_required_pressure_drop": _maybe_float(
                actuator_required_pressure_drop
            ),
            "required_power": _maybe_float(required_power),
            "estimated_weight": _maybe_float(estimated_weight),
            "estimated_volume": _maybe_float(estimated_volume),
            "total_SWAP": total_SWAP,
            "assumptions": [
                "First-pass placeholder model for AFC system sizing and SWaP trends.",
                "Passive components use steady incompressible-style K-factor pressure loss estimates.",
                "Compressor uses ideal-gas isentropic compression with user-specified efficiencies.",
                "Steady jet actuator pressure drop uses Delta p = 0.5*rho*(V_jet/Cd)^2.",
            ],
        }

    def design(
        self,
        *,
        freestream_velocity: float,
        S_ref: float,
        target_C_mu: Optional[float] = None,
        target_mass_flow: Optional[float] = None,
        density_inf: Optional[float] = None,
        q_inf: Optional[float] = None,
        ambient_pressure: float = 101325.0,
        ambient_temperature: float = 288.15,
        jet_velocity: Optional[float] = None,
    ) -> Dict[str, object]:
        """
        Design mode: given target `C_mu` or target mass flow, estimate required pressure ratio, power, weight, volume.
        """
        if target_C_mu is None and target_mass_flow is None:
            raise ValueError("Provide either `target_C_mu` or `target_mass_flow`.")
        if density_inf is None:
            density_inf = ambient_pressure / (287.05 * ambient_temperature)
        if q_inf is None:
            q_inf = dynamic_pressure_from_density_velocity(
                density=density_inf,
                velocity=freestream_velocity,
            )

        actuator = self.actuator
        if jet_velocity is None:
            jet_velocity = actuator.resolve_jet_velocity(
                freestream_velocity=freestream_velocity,
                density=density_inf,
            )
        if target_mass_flow is None:
            mass_flow = mass_flow_from_c_mu(
                C_mu=target_C_mu,
                jet_velocity=jet_velocity,
                q_inf=q_inf,
                S_ref=S_ref,
            )
        else:
            mass_flow = target_mass_flow

        actuator_pressure_drop = actuator.required_pressure_drop(
            jet_velocity=jet_velocity,
            density=density_inf,
        )
        required_actuator_inlet_pressure = ambient_pressure + actuator_pressure_drop

        compressor_index = self.compressor_index
        compressor_required_outlet_pressure = None
        if compressor_index is not None:
            reference_node = AFCSystemNode(
                name="downstream_loss_reference",
                pressure=required_actuator_inlet_pressure,
                temperature=ambient_temperature,
                mass_flow=mass_flow,
            )
            downstream_loss = self._passive_downstream_pressure_loss_estimate(
                start_index=compressor_index + 1,
                stop_before_actuator=True,
                reference_node=reference_node,
                mass_flow=mass_flow,
            )
            compressor_required_outlet_pressure = (
                required_actuator_inlet_pressure + downstream_loss
            )

        result = self._evaluate_with_mass_flow(
            mass_flow=mass_flow,
            freestream_velocity=freestream_velocity,
            S_ref=S_ref,
            density_inf=density_inf,
            q_inf=q_inf,
            ambient_pressure=ambient_pressure,
            ambient_temperature=ambient_temperature,
            compressor_required_outlet_pressure=compressor_required_outlet_pressure,
            jet_velocity=jet_velocity,
            mode="design",
        )
        result["target_C_mu"] = _maybe_float(target_C_mu)
        result["target_mass_flow"] = _maybe_float(target_mass_flow)
        return result

    def off_design(
        self,
        *,
        freestream_velocity: float,
        S_ref: float,
        density_inf: Optional[float] = None,
        q_inf: Optional[float] = None,
        ambient_pressure: float = 101325.0,
        ambient_temperature: float = 288.15,
        mass_flow: Optional[float] = None,
        max_mass_flow: Optional[float] = None,
        tolerance: float = 1e-7,
        max_iterations: int = 80,
    ) -> Dict[str, object]:
        """
        Off-design mode: with fixed component parameters, compute achievable `C_mu`.

        If `mass_flow` is not supplied, a scalar bisection solve finds the mass flow where available actuator pressure
        drop equals the steady-jet actuator pressure requirement.
        """
        if density_inf is None:
            density_inf = ambient_pressure / (287.05 * ambient_temperature)
        if q_inf is None:
            q_inf = dynamic_pressure_from_density_velocity(
                density=density_inf,
                velocity=freestream_velocity,
            )

        def residual(mdot):
            trial = self._evaluate_with_mass_flow(
                mass_flow=mdot,
                freestream_velocity=freestream_velocity,
                S_ref=S_ref,
                density_inf=density_inf,
                q_inf=q_inf,
                ambient_pressure=ambient_pressure,
                ambient_temperature=ambient_temperature,
                mode="off_design",
            )
            actuator_result = trial["component_results"][self.actuator.name]
            return (
                actuator_result["available_pressure_drop"]
                - actuator_result["pressure_loss"]
            )

        if mass_flow is None:
            low = 0.0
            high = max_mass_flow
            if high is None:
                high = density_inf * self.actuator.jet_area * max(
                    freestream_velocity * self.actuator.jet_velocity_ratio,
                    1.0,
                )
                high = max(high, 1e-4)
                while residual(high) > 0 and high < 1e3:
                    high *= 2

            if residual(0.0) <= 0:
                mass_flow = 0.0
            else:
                for _ in range(max_iterations):
                    mid = 0.5 * (low + high)
                    if residual(mid) > 0:
                        low = mid
                    else:
                        high = mid
                    if high - low < tolerance * max(high, 1.0):
                        break
                mass_flow = 0.5 * (low + high)

        return self._evaluate_with_mass_flow(
            mass_flow=mass_flow,
            freestream_velocity=freestream_velocity,
            S_ref=S_ref,
            density_inf=density_inf,
            q_inf=q_inf,
            ambient_pressure=ambient_pressure,
            ambient_temperature=ambient_temperature,
            mode="off_design",
        )

