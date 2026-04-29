from __future__ import annotations

from typing import Optional


def dynamic_pressure_from_density_velocity(
    *,
    density: float,
    velocity: float,
) -> float:
    return 0.5 * density * velocity**2


def c_mu_from_mass_flow(
    *,
    mass_flow: float,
    jet_velocity: float,
    q_inf: Optional[float] = None,
    density_inf: Optional[float] = None,
    velocity_inf: Optional[float] = None,
    S_ref: float,
) -> float:
    """
    Converts AFC steady-jet mass flow to momentum coefficient.

    Convention:
        `C_mu = mdot * V_jet / (q_inf * S_ref)`

    Here `S_ref` is the aerodynamic reference area for the system being sized. For 2D section studies, use a unit span
    convention (`S_ref = chord * 1 m`) and interpret `mass_flow` as kg/s per meter of span. For 3D finite-wing studies,
    use the actual controlled wing reference area and total actuator mass flow.
    """
    if q_inf is None:
        if density_inf is None or velocity_inf is None:
            raise ValueError("Provide either `q_inf` or both `density_inf` and `velocity_inf`.")
        q_inf = dynamic_pressure_from_density_velocity(
            density=density_inf,
            velocity=velocity_inf,
        )
    return mass_flow * jet_velocity / (q_inf * S_ref)


def mass_flow_from_c_mu(
    *,
    C_mu: float,
    jet_velocity: float,
    q_inf: Optional[float] = None,
    density_inf: Optional[float] = None,
    velocity_inf: Optional[float] = None,
    S_ref: float,
) -> float:
    """
    Converts target AFC momentum coefficient to total mass flow.

    Uses the same convention as `c_mu_from_mass_flow()`.
    """
    if q_inf is None:
        if density_inf is None or velocity_inf is None:
            raise ValueError("Provide either `q_inf` or both `density_inf` and `velocity_inf`.")
        q_inf = dynamic_pressure_from_density_velocity(
            density=density_inf,
            velocity=velocity_inf,
        )
    return C_mu * q_inf * S_ref / jet_velocity

