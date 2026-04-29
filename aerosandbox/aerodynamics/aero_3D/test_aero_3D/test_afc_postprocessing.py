import aerosandbox as asb
import aerosandbox.numpy as np


def make_test_airplane() -> asb.Airplane:
    return asb.Airplane(
        name="AFC Postprocessing Test Wing",
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


def test_afc_postprocessing_schema_and_keys():
    result = asb.AFCNeuralFoilLiftingLine(
        airplane=make_test_airplane(),
        op_point=asb.OperatingPoint(velocity=25, alpha=5),
        C_mu=np.array([0.0, 0.02]),
        spanwise_resolution=2,
        model_size="xsmall",
    ).run()

    for key in [
        "CL",
        "CD",
        "CDi",
        "CDp",
        "CM",
        "root_bending_moment",
        "spanwise_load",
        "local_stall_indicator",
        "analysis_confidence",
        "afc_power_proxy",
        "afc_post",
        "postprocess_schema",
    ]:
        assert key in result

    assert np.isfinite(result["root_bending_moment"])
    assert np.isfinite(result["afc_power_proxy"])
    assert np.length(result["spanwise_load"]) == np.length(result["spanwise_y"])
    assert np.length(result["local_stall_indicator"]) == np.length(result["spanwise_y"])
    assert np.all(result["local_stall_indicator"] >= 0)
    assert np.all(result["local_stall_indicator"] <= 1)


def test_plot_spanwise_results_smoke():
    result = asb.AFCNeuralFoilCamberedVLM(
        airplane=make_test_airplane(),
        op_point=asb.OperatingPoint(velocity=25, alpha=5),
        C_mu=0.01,
        spanwise_resolution=2,
        chordwise_resolution=3,
        model_size="xsmall",
        max_iterations=3,
    ).run()

    fig, axes = asb.plot_spanwise_results(result, show=False)
    assert len(axes) == 4
    import matplotlib.pyplot as plt

    plt.close(fig)


if __name__ == "__main__":
    test_afc_postprocessing_schema_and_keys()
    test_plot_spanwise_results_smoke()

