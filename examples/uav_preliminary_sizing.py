"""
Real-world AeroSandbox example: preliminary sizing for a small electric UAV.

This is a concept-design workflow, not a CFD replacement. It shows the normal
first pass in an aircraft project:

1. State mission requirements.
2. Put conservative engineering assumptions in one place.
3. Create optimization variables for the major design choices.
4. Add physics and constraints.
5. Solve, inspect the resulting design, then replace assumptions with better
   data as the project matures.

Run from the repository root after installing dependencies:

    python examples/uav_preliminary_sizing.py

Recommended editable install for this checkout:

    python -m pip install -e ".[full]"
"""

import aerosandbox as asb
import aerosandbox.numpy as np


def main():
    # -------------------------------------------------------------------------
    # Data you normally need for a real first-pass sizing study.
    # Replace these values with your aircraft or mission data.
    # -------------------------------------------------------------------------

    # Mission requirements
    cruise_altitude = 300  # m
    required_endurance = 90 * 60  # s
    max_stall_speed = 11.0  # m/s
    max_wingspan = 3.0  # m

    # Payload and known hardware masses
    payload_mass = 0.8  # kg
    avionics_mass = 0.25  # kg
    propulsion_mass = 0.55  # kg, motor + prop + ESC
    fuselage_tail_mass = 0.75  # kg, fuselage, tail, landing gear, wiring, etc.

    # Aerodynamic assumptions. These can later come from NeuralFoil, VLM,
    # wind-tunnel data, or flight-test system identification.
    cd0 = 0.040  # zero-lift drag coefficient of the whole aircraft
    oswald_efficiency = 0.80
    cl_max = 1.35
    max_cruise_cl = 0.75

    # Propulsion and battery assumptions
    propulsive_efficiency = 0.62
    battery_specific_energy = 220 * 3600  # J/kg, 220 Wh/kg
    usable_battery_fraction = 0.80
    max_continuous_electrical_power = 450  # W

    # Rough structural mass model for the wing. Calibrate these from CAD,
    # historical aircraft, or a structural sizing model as data becomes available.
    wing_area_mass_density = 0.55  # kg / m^2
    spar_mass_coefficient = 0.035  # kg / m^2, multiplied by n_limit * span^2
    limit_load_factor = 3.5

    # -------------------------------------------------------------------------
    # Optimization problem.
    # -------------------------------------------------------------------------

    opti = asb.Opti()

    wing_area = opti.variable(
        init_guess=0.65,
        lower_bound=0.25,
        upper_bound=1.60,
        log_transform=True,
    )  # m^2
    aspect_ratio = opti.variable(
        init_guess=10.0,
        lower_bound=6.0,
        upper_bound=16.0,
        log_transform=True,
    )
    cruise_speed = opti.variable(
        init_guess=18.0,
        lower_bound=10.0,
        upper_bound=35.0,
        log_transform=True,
    )  # m/s
    battery_mass = opti.variable(
        init_guess=1.1,
        lower_bound=0.25,
        upper_bound=4.0,
        log_transform=True,
    )  # kg

    rho_cruise = asb.Atmosphere(altitude=cruise_altitude).density()
    rho_sea_level = asb.Atmosphere(altitude=0).density()
    g = 9.80665

    wingspan = (wing_area * aspect_ratio) ** 0.5
    wing_mass = (
        wing_area_mass_density * wing_area
        + spar_mass_coefficient * limit_load_factor * wingspan**2
    )
    fixed_mass = payload_mass + avionics_mass + propulsion_mass + fuselage_tail_mass
    total_mass = fixed_mass + wing_mass + battery_mass
    weight = total_mass * g

    dynamic_pressure = 0.5 * rho_cruise * cruise_speed**2
    cruise_cl = weight / (dynamic_pressure * wing_area)
    induced_drag_factor = 1 / (np.pi * oswald_efficiency * aspect_ratio)
    cruise_cd = cd0 + induced_drag_factor * cruise_cl**2
    cruise_drag = dynamic_pressure * wing_area * cruise_cd
    electrical_power = cruise_drag * cruise_speed / propulsive_efficiency

    battery_energy_available = (
        battery_mass * battery_specific_energy * usable_battery_fraction
    )
    mission_energy_required = electrical_power * required_endurance
    stall_speed = (2 * weight / (rho_sea_level * wing_area * cl_max)) ** 0.5

    opti.subject_to(
        [
            wingspan <= max_wingspan,
            stall_speed <= max_stall_speed,
            cruise_cl <= max_cruise_cl,
            battery_energy_available >= mission_energy_required,
            electrical_power <= max_continuous_electrical_power,
        ]
    )

    # Minimize takeoff mass. The constraints force the aircraft to still meet
    # endurance, stall, power, and geometry requirements.
    opti.minimize(total_mass)

    sol = opti.solve(verbose=False)

    results = sol(
        {
            "wing_area_m2": wing_area,
            "aspect_ratio": aspect_ratio,
            "wingspan_m": wingspan,
            "cruise_speed_mps": cruise_speed,
            "battery_mass_kg": battery_mass,
            "wing_mass_kg": wing_mass,
            "total_mass_kg": total_mass,
            "stall_speed_mps": stall_speed,
            "cruise_cl": cruise_cl,
            "cruise_cd": cruise_cd,
            "cruise_drag_N": cruise_drag,
            "electrical_power_W": electrical_power,
            "battery_energy_Wh": battery_energy_available / 3600,
            "mission_energy_Wh": mission_energy_required / 3600,
        }
    )

    print("\nSmall electric UAV preliminary sizing")
    print("-------------------------------------")
    for key, value in results.items():
        print(f"{key:>22s}: {float(value):8.3f}")

    print("\nUse this result as a first sizing point, then replace assumptions with:")
    print("- measured component masses")
    print("- airfoil/wing aerodynamic data")
    print("- propeller and motor bench-test data")
    print("- CAD or structural sizing estimates")
    print("- real mission reserves and regulatory constraints")


if __name__ == "__main__":
    main()
