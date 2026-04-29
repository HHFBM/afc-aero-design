import aerosandbox as asb
import aerosandbox.numpy as np


def make_test_airplane() -> asb.Airplane:
    return asb.Airplane(
        name="AFC Cambered VLM Test Wing",
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


def test_afc_cambered_vlm_runs_and_reduces_cl_residual():
    airplane = make_test_airplane()
    op_point = asb.OperatingPoint(velocity=25, alpha=5)

    cl_only = asb.AFCNeuralFoilCamberedVLM(
        airplane=airplane,
        op_point=op_point,
        C_mu=0.01,
        spanwise_resolution=2,
        chordwise_resolution=3,
        model_size="xsmall",
        match_CM=False,
        max_iterations=8,
    ).run()

    for key in ["CL", "CD", "CDi", "CDp", "CM", "CY", "Cl", "Cn"]:
        assert np.isfinite(cl_only[key])

    history = cl_only["convergence_history"]
    assert len(history) > 0
    assert history[-1]["max_abs_residual_CL"] < history[0]["max_abs_residual_CL"]
    assert np.length(cl_only["residual_CL"]) == np.length(cl_only["spanwise_y"])
    assert np.length(cl_only["section_cl"]) == np.length(cl_only["spanwise_y"])
    assert np.length(cl_only["section_cd"]) == np.length(cl_only["spanwise_y"])
    assert np.length(cl_only["section_cm"]) == np.length(cl_only["spanwise_y"])
    assert "converged" in cl_only
    assert "failure_reason" in cl_only
    assert cl_only["iteration_count"] == len(history)
    assert "max_residual" in history[-1]


def test_afc_cambered_vlm_cm_mode_exposes_cm_residual_history():
    airplane = make_test_airplane()
    op_point = asb.OperatingPoint(velocity=25, alpha=5)

    result = asb.AFCNeuralFoilCamberedVLM(
        airplane=airplane,
        op_point=op_point,
        C_mu=np.array([0.0, 0.02]),
        spanwise_resolution=2,
        chordwise_resolution=3,
        model_size="xsmall",
        match_CM=True,
        max_iterations=5,
    ).run()

    assert np.length(result["residual_CM"]) == np.length(result["spanwise_y"])
    assert np.length(result["delta_camber"]) == np.length(result["spanwise_y"])
    assert "max_residual" in result
    assert "converged" in result
    assert "failure_reason" in result
    assert "max_abs_residual_CM" in result["convergence_history"][-1]
    assert (
        result["convergence_history"][-1]["max_abs_residual_CM"]
        < result["convergence_history"][0]["max_abs_residual_CM"]
    )


if __name__ == "__main__":
    test_afc_cambered_vlm_runs_and_reduces_cl_residual()
    test_afc_cambered_vlm_cm_mode_exposes_cm_residual_history()
