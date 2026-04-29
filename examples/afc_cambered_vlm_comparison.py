"""
Compare standard VLM, NonlinearLiftingLine, and AFC-NeuralFoil CamberedVLM.

The CamberedVLM implementation uses virtual local normal-vector corrections to match AFC-NeuralFoil section CL and CM.
This example is intentionally small enough to run quickly.
"""

import aerosandbox as asb
import aerosandbox.numpy as np


airplane = asb.Airplane(
    name="AFC CamberedVLM Demo Wing",
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

op_point = asb.OperatingPoint(velocity=30, alpha=5)

vlm = asb.VortexLatticeMethod(
    airplane=airplane,
    op_point=op_point,
    spanwise_resolution=4,
    chordwise_resolution=4,
).run()

nll = asb.NonlinearLiftingLine(
    airplane=airplane,
    op_point=op_point,
    spanwise_resolution=4,
).run()

cambered = asb.NeuralFoilCamberedVLM(
    airplane=airplane,
    op_point=op_point,
    C_mu=np.array([0.0, 0.015, 0.025]),
    x_jet=0.1,
    theta_jet=30,
    spanwise_resolution=4,
    chordwise_resolution=4,
    model_size="xsmall",
    match_CM=True,
    max_iterations=25,
).run()

print("Method              CL        CD        CDi       CDp       Cm")
print(
    f"standard VLM     {float(vlm['CL']): .5f}  {float(vlm['CD']): .5f}"
    f"       n/a       n/a  {float(vlm['Cm']): .5f}"
)
print(
    f"NonlinearLL      {float(nll['CL']): .5f}  {float(nll['CD']): .5f}"
    f"  {float(nll['CDi']): .5f}  {float(nll['CDp']): .5f}  {float(nll['Cm']): .5f}"
)
print(
    f"CamberedVLM      {float(cambered['CL']): .5f}  {float(cambered['CD']): .5f}"
    f"  {float(cambered['CDi']): .5f}  {float(cambered['CDp']): .5f}"
    f"  {float(cambered['Cm']): .5f}"
)

last = cambered["convergence_history"][-1]
print("\nCamberedVLM final residuals:")
print(f"max |CL residual| = {last['max_abs_residual_CL']:.3e}")
print(f"max |CM residual| = {last['max_abs_residual_CM']:.3e}")
print(f"iterations        = {len(cambered['convergence_history'])}")

print("\nSpanwise diagnostics:")
print("y        C_mu     d_alpha   d_camber  CL_vlm   CL_2D    CM_eff   CM_2D")
for i in range(np.length(cambered["spanwise_y"])):
    print(
        f"{float(cambered['spanwise_y'][i]): .3f}  "
        f"{float(cambered['local_C_mu'][i]): .4f}  "
        f"{float(cambered['delta_alpha'][i]): .3f}  "
        f"{float(cambered['delta_camber'][i]): .3f}  "
        f"{float(cambered['section_CL_vlm'][i]): .4f}  "
        f"{float(cambered['section_CL'][i]): .4f}  "
        f"{float(cambered['section_CM_vlm'][i]): .4f}  "
        f"{float(cambered['section_CM'][i]): .4f}"
    )
