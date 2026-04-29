import aerosandbox as asb
import aerosandbox.numpy as np


def make_wing_airplane(name: str, root_chord: float, tip_chord: float) -> asb.Airplane:
    semi_span = 3.0
    area = (root_chord + tip_chord) * semi_span
    span = 2 * semi_span

    return asb.Airplane(
        name=name,
        xyz_ref=[0.25 * root_chord, 0, 0],
        s_ref=area,
        c_ref=area / span,
        b_ref=span,
        wings=[
            asb.Wing(
                name="Main Wing",
                symmetric=True,
                xsecs=[
                    asb.WingXSec(
                        xyz_le=[0, 0, 0],
                        chord=root_chord,
                        twist=0,
                        airfoil=asb.Airfoil("naca4412"),
                    ),
                    asb.WingXSec(
                        xyz_le=[0.25 * (root_chord - tip_chord), semi_span, 0],
                        chord=tip_chord,
                        twist=0,
                        airfoil=asb.Airfoil("naca4412"),
                    ),
                ],
            )
        ],
    )


op_point = asb.OperatingPoint(
    velocity=30,
    alpha=5,
)

airplanes = [
    make_wing_airplane("Rectangular Wing", root_chord=1.0, tip_chord=1.0),
    make_wing_airplane("Tapered Wing", root_chord=1.2, tip_chord=0.6),
]

for airplane in airplanes:
    print(f"\n{airplane.name}")
    for C_mu in [0.0, 0.01, 0.02]:
        aero = asb.AFCNeuralFoilLiftingLine(
            airplane=airplane,
            op_point=op_point,
            C_mu=C_mu,
            x_jet=0.1,
            theta_jet=30,
            spanwise_resolution=5,
            model_size="xsmall",
        ).run()

        print(
            f"C_mu={C_mu:0.3f}  "
            f"CL={float(aero['CL']):0.3f}  "
            f"CD={float(aero['CD']):0.4f}  "
            f"CL/CD={float(aero['CL'] / aero['CD']):0.1f}"
        )


spanwise_C_mu = np.array([0.00, 0.01, 0.02])
aero_distributed = asb.AFCNeuralFoilLiftingLine(
    airplane=airplanes[0],
    op_point=op_point,
    C_mu=spanwise_C_mu,
    spanwise_resolution=5,
    model_size="xsmall",
).run()

print("\nDistributed AFC C_mu on rectangular wing:")
print("spanwise_y:", aero_distributed["spanwise_y"])
print("local_C_mu:", aero_distributed["local_C_mu"])
print("section_CL:", aero_distributed["section_CL"])
