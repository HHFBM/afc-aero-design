from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import aerosandbox.numpy as np


def _maybe_float(value):
    try:
        return float(value)
    except Exception:
        return value


@dataclass
class AFCSystemNode:
    """
    Thermodynamic node state for the first-pass AFC/SWaP system model.

    This is a low-order engineering placeholder. The state assumes a perfect gas and does not model humidity,
    heat transfer, or transient storage.
    """

    name: str
    pressure: float = 101325.0  # Pa
    temperature: float = 288.15  # K
    mass_flow: float = 0.0  # kg/s
    Mach: float = 0.0  # -
    gamma: float = 1.4
    gas_constant: float = 287.05  # J/(kg*K)

    @property
    def density(self):
        return self.pressure / (self.gas_constant * self.temperature)

    @property
    def speed_of_sound(self):
        return np.sqrt(self.gamma * self.gas_constant * self.temperature)

    @property
    def velocity(self):
        return self.Mach * self.speed_of_sound

    @property
    def dynamic_pressure(self):
        return 0.5 * self.density * self.velocity**2

    def copy_with(
        self,
        *,
        name: Optional[str] = None,
        pressure: Optional[float] = None,
        temperature: Optional[float] = None,
        mass_flow: Optional[float] = None,
        Mach: Optional[float] = None,
    ) -> "AFCSystemNode":
        return AFCSystemNode(
            name=self.name if name is None else name,
            pressure=self.pressure if pressure is None else pressure,
            temperature=self.temperature if temperature is None else temperature,
            mass_flow=self.mass_flow if mass_flow is None else mass_flow,
            Mach=self.Mach if Mach is None else Mach,
            gamma=self.gamma,
            gas_constant=self.gas_constant,
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "pressure": _maybe_float(self.pressure),
            "temperature": _maybe_float(self.temperature),
            "mass_flow": _maybe_float(self.mass_flow),
            "Mach": _maybe_float(self.Mach),
            "density": _maybe_float(self.density),
            "speed_of_sound": _maybe_float(self.speed_of_sound),
            "velocity": _maybe_float(self.velocity),
            "dynamic_pressure": _maybe_float(self.dynamic_pressure),
        }
