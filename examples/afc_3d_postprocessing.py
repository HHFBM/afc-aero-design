"""
Post-process and plot a 3D AFC finite-wing analysis.

This example uses `NeuralFoilCamberedVLM`, but the same post-processing schema and plotting function also work with
`AFCNeuralFoilLiftingLine` and `AFCNeuralFoilNonlinearLiftingLine`.
"""

from pathlib import Path

import aerosandbox as asb
import aerosandbox.numpy as np


airplane = asb.Airplane(
    name="AFC Postprocessing Demo Wing",
    xyz_ref=[0.25, 0, 0],
    s_ref=4,
    c_ref=1,
    b_ref=4,
    wings=[
        asb.Wing(
            symmetric=True,
            xsecs=[
                asb.WingXSec(
                    xyz_le=[0, 0, 0],
                    chord=1.0,
                    airfoil=asb.Airfoil("naca4412"),
                ),
                asb.WingXSec(
                    xyz_le=[0.1, 2, 0],
                    chord=0.75,
                    twist=-2,
                    airfoil=asb.Airfoil("naca4412"),
                ),
            ],
        )
    ],
)

result = asb.NeuralFoilCamberedVLM(
    airplane=airplane,
    op_point=asb.OperatingPoint(velocity=30, alpha=5),
    C_mu=np.array([0.0, 0.015, 0.025]),
    x_jet=0.1,
    theta_jet=30,
    spanwise_resolution=4,
    chordwise_resolution=4,
    model_size="xsmall",
    match_CM=True,
    max_iterations=15,
).run()

print("AFC 3D post-processing summary")
print(f"CL                 : {float(result['CL']): .5f}")
print(f"CD                 : {float(result['CD']): .5f}")
print(f"CDi                : {float(result['CDi']): .5f}")
print(f"CDp                : {float(result['CDp']): .5f}")
print(f"CM                 : {float(result['CM']): .5f}")
print(f"root bending moment: {float(result['root_bending_moment']): .3f} N-m")
print(f"AFC power proxy    : {float(result['afc_power_proxy']): .5f}")
print(f"min confidence     : {float(np.min(result['analysis_confidence'])): .5f}")
print(f"max stall proxy    : {float(np.max(result['local_stall_indicator'])): .5f}")

output_path = Path(__file__).parent / "afc_3d_postprocessing.png"
asb.plot_spanwise_results(
    result,
    title="AFC 3D Postprocessing Demo",
    show=False,
    savefig=output_path,
)
print(f"Wrote {output_path}")

