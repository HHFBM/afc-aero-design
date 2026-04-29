"""
Joint finite-wing geometry and active-flow-control optimization.

This example optimizes a simple tapered wing and a spanwise AFC momentum-coefficient distribution using
`AFCNeuralFoilLiftingLine`. The AFC model used here is the first-pass smooth placeholder surrogate, so this is an API
and workflow example rather than validated AFC design guidance.

Run from the repository root:

    .venv-aerosandbox/bin/python examples/afc_wing_distribution_optimization.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as onp

import aerosandbox as asb
import aerosandbox.numpy as np


def scalar(value):
    return float(onp.ravel(value)[0])


root_chord = 1.0  # m, fixed
airfoil = asb.Airfoil("naca4412")

velocity = 30.0  # m/s
alpha = 4.0  # deg, fixed for this small example
op_point = asb.OperatingPoint(
    velocity=velocity,
    alpha=alpha,
)

spanwise_resolution = 4
n_afc_segments = 4
model_size = "xxsmall"

CL_target = 0.70
span_limit = 7.0  # m
root_bending_limit = 1150.0  # N-m, per wing root, estimated from section lift
analysis_confidence_min = 0.50
lambda_afc = 0.03

initial_tip_chord = 0.70
initial_half_span = 2.50
initial_twist_tip = -2.0
initial_C_mu = np.zeros(n_afc_segments)


def make_airplane(tip_chord, half_span, twist_tip) -> asb.Airplane:
    s_ref = (root_chord + tip_chord) * half_span
    b_ref = 2 * half_span
    c_ref = s_ref / b_ref

    return asb.Airplane(
        name="AFC Optimized Wing",
        xyz_ref=[0.25 * root_chord, 0, 0],
        s_ref=s_ref,
        c_ref=c_ref,
        b_ref=b_ref,
        wings=[
            asb.Wing(
                name="Main Wing",
                symmetric=True,
                xsecs=[
                    asb.WingXSec(
                        xyz_le=[0, 0, 0],
                        chord=root_chord,
                        twist=0,
                        airfoil=airfoil,
                    ),
                    asb.WingXSec(
                        xyz_le=[0.25 * (root_chord - tip_chord), half_span, 0],
                        chord=tip_chord,
                        twist=twist_tip,
                        airfoil=airfoil,
                    ),
                ],
            )
        ],
    )


def analyze(tip_chord, half_span, twist_tip, C_mu):
    return asb.AFCNeuralFoilLiftingLine(
        airplane=make_airplane(tip_chord, half_span, twist_tip),
        op_point=op_point,
        C_mu=C_mu,
        x_jet=0.10,
        theta_jet=30.0,
        spanwise_resolution=spanwise_resolution,
        model_size=model_size,
    ).run()


def root_bending_moment(aero):
    # The model includes mirrored left/right panels. This estimates the bending moment at one wing root.
    return 0.5 * np.sum(np.abs(aero["spanwise_y"]) * aero["section_lift"])


def afc_cost(C_mu):
    return np.mean(C_mu**2)


def print_case(label, tip_chord, half_span, twist_tip, C_mu, aero):
    objective = aero["CD"] + lambda_afc * afc_cost(C_mu)
    print(f"\n{label}")
    print("-" * len(label))
    print(f"tip chord              : {scalar(tip_chord):8.4f} m")
    print(f"half span              : {scalar(half_span):8.4f} m")
    print(f"full span              : {scalar(2 * half_span):8.4f} m")
    print(f"aspect ratio           : {scalar((2 * half_span) ** 2 / ((root_chord + tip_chord) * half_span)):8.4f}")
    print(f"tip twist              : {scalar(twist_tip):8.4f} deg")
    print(f"C_mu segments          : {onp.array(C_mu, dtype=float)}")
    print(f"CL                     : {scalar(aero['CL']):8.4f}")
    print(f"CD total               : {scalar(aero['CD']):8.5f}")
    print(f"CDi / CDp              : {scalar(aero['CDi']):8.5f} / {scalar(aero['CDp']):8.5f}")
    print(f"objective              : {scalar(objective):8.5f}")
    print(f"root bending moment    : {scalar(root_bending_moment(aero)):8.2f} N-m")
    print(f"min analysis confidence: {scalar(np.min(aero['analysis_confidence'])):8.4f}")


initial_aero = analyze(
    tip_chord=initial_tip_chord,
    half_span=initial_half_span,
    twist_tip=initial_twist_tip,
    C_mu=initial_C_mu,
)

opti = asb.Opti()

tip_chord = opti.variable(
    init_guess=initial_tip_chord,
    lower_bound=0.35,
    upper_bound=1.20,
)
half_span = opti.variable(
    init_guess=initial_half_span,
    lower_bound=1.80,
    upper_bound=span_limit / 2,
)
twist_tip = opti.variable(
    init_guess=initial_twist_tip,
    lower_bound=-8.0,
    upper_bound=4.0,
)
C_mu_segments = opti.variable(
    init_guess=0.01 * np.ones(n_afc_segments),
    lower_bound=0.0,
    upper_bound=0.10,
)

aero = analyze(
    tip_chord=tip_chord,
    half_span=half_span,
    twist_tip=twist_tip,
    C_mu=C_mu_segments,
)

opti.subject_to(aero["CL"] >= CL_target)
opti.subject_to(2 * half_span <= span_limit)
opti.subject_to(root_bending_moment(aero) <= root_bending_limit)
opti.subject_to(np.min(aero["analysis_confidence"]) >= analysis_confidence_min)

# The airfoil is fixed in this example, so the thickness constraint is a constant design requirement.
min_airfoil_thickness = 0.10
if scalar(airfoil.max_thickness()) < min_airfoil_thickness:
    raise ValueError("The selected fixed airfoil does not satisfy the minimum thickness requirement.")

opti.minimize(
    aero["CD"] + lambda_afc * afc_cost(C_mu_segments)
)

sol = opti.solve(verbose=False)

opt_tip_chord = sol(tip_chord)
opt_half_span = sol(half_span)
opt_twist_tip = sol(twist_tip)
opt_C_mu_segments = sol(C_mu_segments)
opt_aero = analyze(
    tip_chord=opt_tip_chord,
    half_span=opt_half_span,
    twist_tip=opt_twist_tip,
    C_mu=opt_C_mu_segments,
)

print("AFC finite-wing optimization")
print("============================")
print(f"CL target               : {CL_target:.3f}")
print(f"span limit              : {span_limit:.3f} m")
print(f"root bending limit      : {root_bending_limit:.1f} N-m")
print(f"confidence threshold    : {analysis_confidence_min:.3f}")
print(f"fixed airfoil max t/c   : {scalar(airfoil.max_thickness()):.3f}")

print_case(
    "Initial design",
    initial_tip_chord,
    initial_half_span,
    initial_twist_tip,
    initial_C_mu,
    initial_aero,
)
print_case(
    "Optimized design",
    opt_tip_chord,
    opt_half_span,
    opt_twist_tip,
    opt_C_mu_segments,
    opt_aero,
)


def positive_side(aero):
    y = onp.array(aero["spanwise_y"], dtype=float)
    keep = y >= 0
    order = onp.argsort(y[keep])
    return {
        key: onp.array(aero[key], dtype=float)[keep][order]
        for key in [
            "spanwise_y",
            "local_C_mu",
            "section_lift",
            "section_CL",
            "section_CD",
        ]
    }


initial_side = positive_side(initial_aero)
opt_side = positive_side(opt_aero)

fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))

axes[0].plot(initial_side["spanwise_y"], initial_side["local_C_mu"], "o--", label="Initial")
axes[0].plot(opt_side["spanwise_y"], opt_side["local_C_mu"], "s-", label="Optimized")
axes[0].set_xlabel("Spanwise y [m]")
axes[0].set_ylabel(r"$C_\mu$")
axes[0].grid(True, alpha=0.3)
axes[0].legend()

axes[1].plot(initial_side["spanwise_y"], initial_side["section_lift"], "o--", label="Initial")
axes[1].plot(opt_side["spanwise_y"], opt_side["section_lift"], "s-", label="Optimized")
axes[1].set_xlabel("Spanwise y [m]")
axes[1].set_ylabel("Panel lift [N]")
axes[1].grid(True, alpha=0.3)

axes[2].plot(initial_side["spanwise_y"], initial_side["section_CL"], "o--", label="Initial CL")
axes[2].plot(opt_side["spanwise_y"], opt_side["section_CL"], "s-", label="Optimized CL")
axes[2].plot(initial_side["spanwise_y"], initial_side["section_CD"], "o:", label="Initial CD")
axes[2].plot(opt_side["spanwise_y"], opt_side["section_CD"], "s:", label="Optimized CD")
axes[2].set_xlabel("Spanwise y [m]")
axes[2].set_ylabel("Section coefficient")
axes[2].grid(True, alpha=0.3)
axes[2].legend()

fig.tight_layout()
figure_path = Path(__file__).with_suffix(".png")
fig.savefig(figure_path, dpi=200)
print(f"\nSaved plot: {figure_path}")

if __name__ == "__main__":
    pass
