"""
Walkthrough: a rectangular wing analyzed with AeroSandbox VLM.

This is a deliberately small, visualizable test case for understanding how
AeroSandbox turns engineering inputs into a low-order aerodynamic simulation.

Important: this is not RANS CFD. VortexLatticeMethod is an inviscid potential-
flow method. It solves for vortex strengths on wing panels, captures finite-wing
lift and induced drag, and ignores viscous profile drag. In this example, we add
a simple 2D NeuralFoil section-drag estimate to show how AeroSandbox can combine
3D low-order aerodynamics with 2D viscous surrogate data.

Run from the repository root:

    .venv-aerosandbox/bin/python examples/rectangular_wing_vlm_walkthrough.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as onp

import aerosandbox as asb


def scalar(value):
    """Convert AeroSandbox/NumPy scalar-like values to a Python float."""
    return float(onp.ravel(value)[0])


def build_rectangular_wing():
    airfoil = asb.Airfoil("naca0012")

    chord = 0.30  # m
    half_span = 1.00  # m; symmetric=True mirrors this to a 2.0 m full span

    airplane = asb.Airplane(
        name="NACA0012 Rectangular Wing",
        xyz_ref=[0.25 * chord, 0, 0],  # moment reference at quarter chord
        wings=[
            asb.Wing(
                name="Main Wing",
                symmetric=True,
                xsecs=[
                    asb.WingXSec(
                        xyz_le=[0, 0, 0],
                        chord=chord,
                        twist=0,
                        airfoil=airfoil,
                    ),
                    asb.WingXSec(
                        xyz_le=[0, half_span, 0],
                        chord=chord,
                        twist=0,
                        airfoil=airfoil,
                    ),
                ],
            )
        ],
    )

    return airplane, airfoil, chord


def main():
    airplane, airfoil, chord = build_rectangular_wing()

    velocity = 20.0  # m/s
    altitude = 0.0  # m
    alphas = [0, 2, 4, 6, 8]  # degrees

    atmosphere = asb.Atmosphere(altitude=altitude)
    rho = scalar(atmosphere.density())
    mu = scalar(atmosphere.dynamic_viscosity())
    speed_of_sound = scalar(atmosphere.speed_of_sound())
    q = 0.5 * rho * velocity**2
    Re = rho * velocity * chord / mu
    mach = velocity / speed_of_sound

    spanwise_resolution = 12
    chordwise_resolution = 6

    print("\nInput geometry")
    print("--------------")
    print(f"airfoil                 : NACA0012")
    print(f"full span b_ref          : {airplane.b_ref:.3f} m")
    print(f"reference area S_ref     : {airplane.s_ref:.3f} m^2")
    print(f"reference chord c_ref    : {airplane.c_ref:.3f} m")
    print(f"aspect ratio AR          : {airplane.b_ref**2 / airplane.s_ref:.3f}")

    print("\nInput flight condition")
    print("----------------------")
    print(f"velocity                 : {velocity:.3f} m/s")
    print(f"altitude                 : {altitude:.3f} m")
    print(f"air density rho          : {rho:.4f} kg/m^3")
    print(f"dynamic pressure q       : {q:.3f} Pa")
    print(f"section Reynolds number  : {Re:.3e}")
    print(f"Mach number              : {mach:.4f}")

    results = []
    first_analysis = None

    for alpha in alphas:
        op_point = asb.OperatingPoint(
            atmosphere=atmosphere,
            velocity=velocity,
            alpha=alpha,
        )

        analysis = asb.VortexLatticeMethod(
            airplane=airplane,
            op_point=op_point,
            spanwise_resolution=spanwise_resolution,
            chordwise_resolution=chordwise_resolution,
        )
        vlm = analysis.run()
        if first_analysis is None:
            first_analysis = analysis

        section_aero = airfoil.get_aero_from_neuralfoil(
            alpha=alpha,
            Re=Re,
            mach=mach,
            model_size="large",
        )

        cd_induced = scalar(vlm["CD"])
        cd_profile = scalar(section_aero["CD"])
        cd_total_estimate = cd_induced + cd_profile

        results.append(
            {
                "alpha": alpha,
                "CL_3D": scalar(vlm["CL"]),
                "CDi_VLM": cd_induced,
                "CDp_2D_NeuralFoil": cd_profile,
                "CD_total_est": cd_total_estimate,
                "Cm_3D": scalar(vlm["Cm"]),
                "L_N": scalar(vlm["L"]),
                "D_induced_N": scalar(vlm["D"]),
                "D_total_est_N": q * airplane.s_ref * cd_total_estimate,
                "section_confidence": scalar(section_aero["analysis_confidence"]),
            }
        )

    print("\nDiscretization used by VLM")
    print("--------------------------")
    print(f"spanwise panels per half wing : {spanwise_resolution}")
    print(f"chordwise panels              : {chordwise_resolution}")
    print(f"total VLM panels solved       : {len(first_analysis.vortex_centers)}")

    print("\nAlpha sweep results")
    print("-------------------")
    print(
        "alpha  CL_3D   CDi_VLM  CDp_2D   CD_total  Cm_3D   L[N]    D_total[N]  NF_conf"
    )
    for r in results:
        print(
            f"{r['alpha']:>5.1f}"
            f"  {r['CL_3D']:>6.3f}"
            f"  {r['CDi_VLM']:>7.4f}"
            f"  {r['CDp_2D_NeuralFoil']:>7.4f}"
            f"  {r['CD_total_est']:>8.4f}"
            f"  {r['Cm_3D']:>6.3f}"
            f"  {r['L_N']:>6.1f}"
            f"  {r['D_total_est_N']:>10.2f}"
            f"  {r['section_confidence']:>7.3f}"
        )

    output_dir = Path(__file__).parent
    figure_path = output_dir / "rectangular_wing_vlm_walkthrough.png"

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    axes[0].plot([r["alpha"] for r in results], [r["CL_3D"] for r in results], "o-")
    axes[0].set_xlabel("Angle of attack [deg]")
    axes[0].set_ylabel("3D lift coefficient CL")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        [r["alpha"] for r in results],
        [r["CDi_VLM"] for r in results],
        "o-",
        label="VLM induced drag",
    )
    axes[1].plot(
        [r["alpha"] for r in results],
        [r["CD_total_est"] for r in results],
        "s-",
        label="Induced + 2D profile estimate",
    )
    axes[1].set_xlabel("Angle of attack [deg]")
    axes[1].set_ylabel("Drag coefficient CD")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(figure_path, dpi=200)
    print(f"\nSaved plot: {figure_path}")


if __name__ == "__main__":
    main()
