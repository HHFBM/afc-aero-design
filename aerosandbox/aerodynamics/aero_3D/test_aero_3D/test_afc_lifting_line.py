import aerosandbox as asb
import aerosandbox.numpy as np


def make_test_airplane() -> asb.Airplane:
    return asb.Airplane(
        name="AFC LL Test Wing",
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
                        chord=1,
                        airfoil=asb.Airfoil("naca0012"),
                    ),
                    asb.WingXSec(
                        xyz_le=[0, 2, 0],
                        chord=1,
                        airfoil=asb.Airfoil("naca0012"),
                    ),
                ],
            )
        ],
    )


def test_afc_lifting_line_runs_with_scalar_and_distributed_c_mu():
    airplane = make_test_airplane()
    op_point = asb.OperatingPoint(velocity=25, alpha=5)

    scalar_result = asb.AFCNeuralFoilLiftingLine(
        airplane=airplane,
        op_point=op_point,
        C_mu=0.01,
        spanwise_resolution=3,
        model_size="xsmall",
    ).run()

    distributed_result = asb.AFCNeuralFoilLiftingLine(
        airplane=airplane,
        op_point=op_point,
        C_mu=np.array([0.0, 0.01, 0.02]),
        spanwise_resolution=3,
        model_size="xsmall",
    ).run()

    required_scalar_keys = ["CL", "CD", "CDi", "CDp", "CM", "CY", "Cl", "Cn"]
    required_section_keys = [
        "spanwise_y",
        "local_alpha",
        "local_Re",
        "local_C_mu",
        "section_CL",
        "section_CD",
        "section_CM",
        "section_cl",
        "section_cd",
        "section_cm",
        "residual_CL",
        "residual_CM",
        "analysis_confidence",
    ]

    for key in required_scalar_keys:
        assert np.isfinite(scalar_result[key])
        assert np.isfinite(distributed_result[key])

    n_sections = np.length(scalar_result["spanwise_y"])
    assert n_sections > 0

    for key in required_section_keys:
        assert np.length(scalar_result[key]) == n_sections
        assert np.length(distributed_result[key]) == n_sections
        assert np.all(np.isfinite(scalar_result[key]))
        assert np.all(np.isfinite(distributed_result[key]))

    assert np.allclose(scalar_result["local_C_mu"], 0.01)
    assert np.min(distributed_result["local_C_mu"]) >= 0
    assert np.max(distributed_result["local_C_mu"]) <= 0.02
    assert scalar_result["converged"] is True
    assert scalar_result["failure_reason"] is None
    assert scalar_result["iteration_count"] == 1
    assert scalar_result["max_residual"] == 0.0
    assert len(scalar_result["convergence_history"]) == 1


def test_afc_lifting_line_zero_c_mu_matches_lifting_line():
    airplane = make_test_airplane()
    op_point = asb.OperatingPoint(velocity=25, alpha=5)

    baseline = asb.LiftingLine(
        airplane=airplane,
        op_point=op_point,
        spanwise_resolution=3,
        model_size="xsmall",
    ).run()

    afc = asb.AFCNeuralFoilLiftingLine(
        airplane=airplane,
        op_point=op_point,
        C_mu=0.0,
        spanwise_resolution=3,
        model_size="xsmall",
    ).run()

    for key in ["CL", "CD", "Cm"]:
        assert np.allclose(afc[key], baseline[key], rtol=0, atol=1e-12)


if __name__ == "__main__":
    test_afc_lifting_line_runs_with_scalar_and_distributed_c_mu()
    test_afc_lifting_line_zero_c_mu_matches_lifting_line()
