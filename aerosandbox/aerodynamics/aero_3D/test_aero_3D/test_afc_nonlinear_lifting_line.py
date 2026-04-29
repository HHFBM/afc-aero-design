import aerosandbox as asb
import aerosandbox.numpy as np


def make_test_airplane() -> asb.Airplane:
    return asb.Airplane(
        name="AFC Nonlinear LL Test Wing",
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


def test_afc_nonlinear_lifting_line_solve_true_runs_and_matches_residuals():
    airplane = make_test_airplane()
    op_point = asb.OperatingPoint(velocity=25, alpha=5)

    result = asb.AFCNeuralFoilNonlinearLiftingLine(
        airplane=airplane,
        op_point=op_point,
        C_mu=0.01,
        spanwise_resolution=2,
        model_size="xsmall",
    ).run(solve=True)

    for key in ["CL", "CD", "CDi", "CDp", "CM", "CY", "Cl", "Cn"]:
        assert np.isfinite(result[key])

    n_sections = np.length(result["spanwise_y"])
    for key in [
        "gamma",
        "residuals",
        "local_alpha",
        "local_Re",
        "local_C_mu",
        "section_CL_from_gamma",
        "section_CL",
        "section_CD",
        "section_CM",
        "section_cl",
        "section_cd",
        "section_cm",
        "residual_CL",
        "residual_CM",
        "analysis_confidence",
    ]:
        assert np.length(result[key]) == n_sections
        assert np.all(np.isfinite(result[key]))

    assert np.max(np.abs(result["residuals"])) < 1e-6
    assert np.max(np.abs(result["section_CL_from_gamma"] - result["section_CL"])) < 1e-6
    assert result["converged"] is True
    assert result["failure_reason"] is None
    assert result["iteration_count"] >= 1
    assert len(result["convergence_history"]) >= 1


def test_afc_nonlinear_lifting_line_solve_false_exposes_external_residuals():
    airplane = make_test_airplane()
    op_point = asb.OperatingPoint(velocity=25, alpha=4)
    opti = asb.Opti()

    result = asb.AFCNeuralFoilNonlinearLiftingLine(
        airplane=airplane,
        op_point=op_point,
        C_mu=np.array([0.0, 0.02]),
        spanwise_resolution=2,
        model_size="xsmall",
        opti=opti,
    ).run(solve=False)

    assert np.length(result["residuals"]) == np.length(result["gamma"])
    assert result["converged"] is False
    assert result["failure_reason"] == "solve_false_external_residuals_not_constrained"
    opti.subject_to(result["residuals"] == 0)
    sol = opti.solve(verbose=False)

    assert np.max(np.abs(sol(result["residuals"]))) < 1e-6
    assert np.all(np.isfinite(sol(result["gamma"])))


if __name__ == "__main__":
    test_afc_nonlinear_lifting_line_solve_true_runs_and_matches_residuals()
    test_afc_nonlinear_lifting_line_solve_false_exposes_external_residuals()
