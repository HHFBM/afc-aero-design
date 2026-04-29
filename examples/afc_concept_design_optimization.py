"""
End-to-end AFC concept design optimization.

This example connects:

- 2D `Airfoil.get_aero_from_afc_neuralfoil()` section aerodynamics,
- 3D `AFCNeuralFoilLiftingLine`,
- first-pass AFC system SWaP sizing via `AFCSystemNetwork.design()`,
- `asb.Opti` design optimization.

The AFC aerodynamic correction and SWaP component models used here are placeholders for workflow validation and early
trade studies. They are not calibrated CFD, wind-tunnel, compressor-map, or actuator-test models.

Run from the repository root:

    .venv-aerosandbox/bin/python examples/afc_concept_design_optimization.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as onp
import pandas as pd

import aerosandbox as asb
import aerosandbox.numpy as np


OUTPUT_DIR = Path(__file__).parent / "generated_afc_concept_design_optimization"
PLACEHOLDER_NOTES = [
    "AFC aerodynamic correction is the current smooth placeholder AFC-NeuralFoil interface.",
    "SWaP is a first-pass placeholder model using steady pressure-loss and ideal-gas compressor estimates.",
    "System target C_mu is the mean of the spanwise C_mu design segments.",
]

root_chord = 1.0  # m, fixed at the root for this compact example.
airfoil = asb.Airfoil("naca4412")

velocity = 30.0  # m/s
op_point = asb.OperatingPoint(velocity=velocity, alpha=4.0)

spanwise_resolution = 4
n_afc_segments = 4
model_size = "xxsmall"

target_CL = 0.80
span_limit = 6.0  # m
root_bending_limit = 1250.0  # N-m, approximate mirrored-wing root-bending proxy.
analysis_confidence_min = 0.30
power_limit = 12_000.0  # W
weight_limit = 15.0  # kg

lambda_power = 2.0e-6
lambda_weight = 5.0e-4

initial_design = {
    "tip_chord": 0.72,
    "half_span": 2.55,
    "twist_tip": -2.0,
    "C_mu_segments": onp.array([0.004, 0.006, 0.008, 0.010]),
    "jet_velocity_ratio": 2.2,
}


def scalar(value) -> float:
    return float(onp.ravel(onp.asarray(value, dtype=float))[0])


def array1d(value):
    return onp.ravel(onp.asarray(value, dtype=float))


def jsonable(value):
    if value is None:
        return None
    if isinstance(value, (str, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    try:
        arr = onp.asarray(value, dtype=float)
        if arr.shape == ():
            return float(arr)
        return arr.tolist()
    except Exception:
        return value


def clean_small_values(value, *, tolerance: float = 1e-9):
    arr = array1d(value).copy()
    arr[onp.abs(arr) < tolerance] = 0.0
    return arr


def make_airplane(tip_chord, half_span, twist_tip) -> asb.Airplane:
    s_ref = (root_chord + tip_chord) * half_span
    b_ref = 2 * half_span
    c_ref = s_ref / b_ref

    return asb.Airplane(
        name="AFC Concept Design Wing",
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
                        twist=0.0,
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


def make_afc_system_network() -> asb.AFCSystemNetwork:
    return asb.AFCSystemNetwork(
        name="concept_steady_jet_afc_system",
        components=[
            asb.AFCInlet(name="inlet", area=0.018, pressure_recovery=0.98),
            asb.AFCCompressor(
                name="compressor",
                fixed_weight=4.0,
                fixed_volume=0.018,
            ),
            asb.AFCDuct(
                name="main_duct",
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
                fixed_weight=0.4,
                fixed_volume=0.001,
            ),
        ],
    )


def analyze_aero(tip_chord, half_span, twist_tip, C_mu_segments):
    return asb.AFCNeuralFoilLiftingLine(
        airplane=make_airplane(tip_chord, half_span, twist_tip),
        op_point=op_point,
        C_mu=C_mu_segments,
        x_jet=0.10,
        theta_jet=30.0,
        spanwise_resolution=spanwise_resolution,
        model_size=model_size,
    ).run()


def size_system(tip_chord, half_span, C_mu_segments, jet_velocity_ratio):
    s_ref = (root_chord + tip_chord) * half_span
    target_system_C_mu = np.mean(C_mu_segments)
    return make_afc_system_network().design(
        freestream_velocity=velocity,
        density_inf=op_point.atmosphere.density(),
        S_ref=s_ref,
        target_C_mu=target_system_C_mu,
        jet_velocity=jet_velocity_ratio * velocity,
    )


def objective(aero, system):
    return (
        aero["CD"]
        + lambda_power * system["required_power"]
        + lambda_weight * system["estimated_weight"]
    )


def summarize_case(label: str, design: Dict[str, object], aero, system) -> Dict[str, object]:
    return {
        "label": label,
        "tip_chord": scalar(design["tip_chord"]),
        "half_span": scalar(design["half_span"]),
        "full_span": scalar(2 * design["half_span"]),
        "aspect_ratio": scalar(
            (2 * design["half_span"]) ** 2
            / ((root_chord + design["tip_chord"]) * design["half_span"])
        ),
        "twist_tip": scalar(design["twist_tip"]),
        "C_mu_segments": clean_small_values(design["C_mu_segments"]).tolist(),
        "mean_C_mu": scalar(np.mean(clean_small_values(design["C_mu_segments"]))),
        "jet_velocity_ratio": scalar(design["jet_velocity_ratio"]),
        "CL": scalar(aero["CL"]),
        "CD_total": scalar(aero["CD"]),
        "CDi": scalar(aero["CDi"]),
        "CDp": scalar(aero["CDp"]),
        "CM": scalar(aero["CM"]),
        "converged": bool(aero.get("converged", True)),
        "failure_reason": aero.get("failure_reason"),
        "root_bending_moment": scalar(aero["root_bending_moment"]),
        "min_analysis_confidence": scalar(np.min(aero["analysis_confidence"])),
        "required_power": scalar(system["required_power"]),
        "system_weight": scalar(system["estimated_weight"]),
        "system_volume": scalar(system["estimated_volume"]),
        "mass_flow": scalar(system["mass_flow"]),
        "jet_velocity": scalar(system["jet_velocity"]),
        "objective": scalar(objective(aero, system)),
    }


def positive_spanwise(aero):
    y = array1d(aero["spanwise_y"])
    keep = y >= 0
    order = onp.argsort(y[keep])
    fields = [
        "spanwise_y",
        "local_C_mu",
        "section_lift",
        "section_CL",
        "section_CD",
    ]
    out = {field: array1d(aero[field])[keep][order] for field in fields}
    out["local_C_mu"] = clean_small_values(out["local_C_mu"])
    return out


def component_breakdown(system):
    component_results = system["component_results"]
    names = list(component_results.keys())
    return {
        "names": names,
        "pressure_loss": [
            scalar(system["component_pressure_losses"].get(name, 0.0)) for name in names
        ],
        "weight": [scalar(component_results[name]["estimated_weight"]) for name in names],
        "volume": [scalar(component_results[name]["estimated_volume"]) for name in names],
    }


def write_outputs(
    summary,
    initial_aero,
    optimized_aero,
    initial_system,
    optimized_system,
    *,
    output_directory: Path = OUTPUT_DIR,
):
    output_directory.mkdir(parents=True, exist_ok=True)

    rows = [summary["initial"]]
    if summary["optimized"] is not None:
        rows.append(summary["optimized"])
    comparison = pd.DataFrame(rows)
    comparison_path = output_directory / "comparison_table.csv"
    comparison.to_csv(comparison_path, index=False)
    print("\nOptimization comparison")
    print(comparison.to_string(index=False))

    if optimized_aero is None:
        aero_cases = {"Initial": initial_aero}
        system_cases = {"Initial": initial_system}
    else:
        aero_cases = {"Initial": initial_aero, "Optimized": optimized_aero}
        system_cases = {"Initial": initial_system, "Optimized": optimized_system}

    fig, ax = plt.subplots(figsize=(6, 3.6))
    for label, aero in aero_cases.items():
        side = positive_spanwise(aero)
        ax.plot(side["spanwise_y"], side["local_C_mu"], "o-", label=label)
    ax.set_xlabel("Spanwise y [m]")
    ax.set_ylabel(r"$C_\mu$")
    ax.set_title("Spanwise AFC Distribution")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    c_mu_plot = output_directory / "spanwise_c_mu.png"
    fig.savefig(c_mu_plot, dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.6))
    for label, aero in aero_cases.items():
        side = positive_spanwise(aero)
        ax.plot(side["spanwise_y"], side["section_lift"], "o-", label=label)
    ax.set_xlabel("Spanwise y [m]")
    ax.set_ylabel("Section lift [N]")
    ax.set_title("Spanwise Load")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    load_plot = output_directory / "spanwise_load.png"
    fig.savefig(load_plot, dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.6))
    for label, aero in aero_cases.items():
        side = positive_spanwise(aero)
        ax.plot(side["spanwise_y"], side["section_CL"], "o-", label=f"{label} cl")
        ax.plot(side["spanwise_y"], side["section_CD"], "s--", label=f"{label} cd")
    ax.set_xlabel("Spanwise y [m]")
    ax.set_ylabel("Section coefficient")
    ax.set_title("Section cl/cd")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    section_plot = output_directory / "section_cl_cd.png"
    fig.savefig(section_plot, dpi=200)
    plt.close(fig)

    final_system = optimized_system if optimized_system is not None else initial_system
    breakdown = component_breakdown(final_system)
    x = onp.arange(len(breakdown["names"]))
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].bar(x, onp.array(breakdown["pressure_loss"]) / 1000)
    axes[0].set_ylabel("Pressure loss [kPa]")
    axes[0].set_title("SWaP Breakdown - Placeholder Component Models")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[1].bar(x - 0.18, breakdown["weight"], width=0.36, label="weight [kg]")
    axes[1].bar(x + 0.18, onp.array(breakdown["volume"]) * 1000, width=0.36, label="volume [L]")
    axes[1].set_ylabel("Weight / Volume")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(breakdown["names"], rotation=25, ha="right")
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    swap_plot = output_directory / "swap_breakdown_placeholder.png"
    fig.savefig(swap_plot, dpi=200)
    plt.close(fig)

    summary["outputs"] = {
        "comparison_table": str(comparison_path),
        "spanwise_c_mu_plot": str(c_mu_plot),
        "spanwise_load_plot": str(load_plot),
        "section_cl_cd_plot": str(section_plot),
        "swap_breakdown_plot": str(swap_plot),
    }
    summary_path = output_directory / "summary.json"
    summary_path.write_text(json.dumps(jsonable(summary), indent=2, allow_nan=False))
    print(f"\nWrote outputs to {output_directory}")
    print(f"JSON summary: {summary_path}")
    return summary


def run_example(*, output_directory: Path = OUTPUT_DIR):
    initial_aero = analyze_aero(
        tip_chord=initial_design["tip_chord"],
        half_span=initial_design["half_span"],
        twist_tip=initial_design["twist_tip"],
        C_mu_segments=initial_design["C_mu_segments"],
    )
    initial_system = size_system(
        tip_chord=initial_design["tip_chord"],
        half_span=initial_design["half_span"],
        C_mu_segments=initial_design["C_mu_segments"],
        jet_velocity_ratio=initial_design["jet_velocity_ratio"],
    )

    opti = asb.Opti()
    tip_chord = opti.variable(init_guess=initial_design["tip_chord"], lower_bound=0.40, upper_bound=1.15)
    half_span = opti.variable(init_guess=initial_design["half_span"], lower_bound=2.0, upper_bound=span_limit / 2)
    twist_tip = opti.variable(init_guess=initial_design["twist_tip"], lower_bound=-7.0, upper_bound=3.0)
    C_mu_segments = opti.variable(
        init_guess=initial_design["C_mu_segments"],
        lower_bound=0.0,
        upper_bound=0.10,
    )
    jet_velocity_ratio = opti.variable(
        init_guess=initial_design["jet_velocity_ratio"],
        lower_bound=1.6,
        upper_bound=3.0,
    )

    aero = analyze_aero(tip_chord, half_span, twist_tip, C_mu_segments)
    system = size_system(tip_chord, half_span, C_mu_segments, jet_velocity_ratio)

    opti.subject_to(aero["CL"] >= target_CL)
    opti.subject_to(2 * half_span <= span_limit)
    opti.subject_to(aero["root_bending_moment"] <= root_bending_limit)
    opti.subject_to(np.min(aero["analysis_confidence"]) >= analysis_confidence_min)
    opti.subject_to(system["required_power"] <= power_limit)
    opti.subject_to(system["estimated_weight"] <= weight_limit)

    opti.minimize(objective(aero, system))

    summary = {
        "status": "not_solved",
        "failure_reason": None,
        "assumptions": PLACEHOLDER_NOTES,
        "design_variables": {
            "tip_chord": "optimized",
            "half_span": "optimized",
            "twist_tip": "optimized",
            "C_mu_segments": int(n_afc_segments),
            "jet_velocity_ratio": "optimized_placeholder_system_variable",
        },
        "constraints": {
            "target_CL": target_CL,
            "span_limit": span_limit,
            "root_bending_limit": root_bending_limit,
            "analysis_confidence_min": analysis_confidence_min,
            "power_limit": power_limit,
            "weight_limit": weight_limit,
        },
        "initial": summarize_case("Initial", initial_design, initial_aero, initial_system),
        "optimized": None,
    }

    optimized_aero = None
    optimized_system = None
    try:
        sol = opti.solve(verbose=False)
        optimized_design = {
            "tip_chord": sol(tip_chord),
            "half_span": sol(half_span),
            "twist_tip": sol(twist_tip),
            "C_mu_segments": sol(C_mu_segments),
            "jet_velocity_ratio": sol(jet_velocity_ratio),
        }
        optimized_aero = analyze_aero(
            tip_chord=optimized_design["tip_chord"],
            half_span=optimized_design["half_span"],
            twist_tip=optimized_design["twist_tip"],
            C_mu_segments=optimized_design["C_mu_segments"],
        )
        optimized_system = size_system(
            tip_chord=optimized_design["tip_chord"],
            half_span=optimized_design["half_span"],
            C_mu_segments=optimized_design["C_mu_segments"],
            jet_velocity_ratio=optimized_design["jet_velocity_ratio"],
        )
        summary["status"] = "solved"
        summary["optimized"] = summarize_case(
            "Optimized",
            optimized_design,
            optimized_aero,
            optimized_system,
        )
    except Exception as e:
        summary["status"] = "failed"
        summary["failure_reason"] = f"{type(e).__name__}: {e}"
        print("\nOptimization failed")
        print(summary["failure_reason"])

    return write_outputs(
        summary,
        initial_aero,
        optimized_aero,
        initial_system,
        optimized_system,
        output_directory=output_directory,
    )


def main():
    run_example(output_directory=OUTPUT_DIR)


if __name__ == "__main__":
    main()
