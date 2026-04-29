import aerosandbox as asb
import aerosandbox.numpy as np


def test_afc_neuralfoil_zero_c_mu_matches_neuralfoil():
    airfoil = asb.Airfoil("naca0012")
    alpha = np.array([-2.0, 0.0, 6.0, 12.0])

    baseline = airfoil.get_aero_from_neuralfoil(
        alpha=alpha,
        Re=1e6,
        mach=0.05,
        model_size="xsmall",
    )

    afc = airfoil.get_aero_from_afc_neuralfoil(
        alpha=alpha,
        Re=1e6,
        mach=0.05,
        C_mu=0.0,
        x_jet=0.1,
        theta_jet=30.0,
        model_size="xsmall",
    )

    for key in ["CL", "CD", "CM", "Top_Xtr", "Bot_Xtr", "analysis_confidence"]:
        assert np.allclose(afc[key], baseline[key], rtol=0, atol=1e-12)

    assert "ood_score" in afc
    assert "stall_risk" in afc
    assert "confidence_reason" in afc


def test_afc_confidence_decreases_outside_nominal_domain():
    airfoil = asb.Airfoil("naca0012")

    nominal = airfoil.get_aero_from_afc_neuralfoil(
        alpha=6.0,
        Re=1e6,
        mach=0.05,
        C_mu=0.02,
        x_jet=0.1,
        theta_jet=30.0,
        model_size="xsmall",
    )
    out_of_domain = airfoil.get_aero_from_afc_neuralfoil(
        alpha=80.0,
        Re=1e6,
        mach=0.05,
        C_mu=0.50,
        x_jet=0.1,
        theta_jet=30.0,
        model_size="xsmall",
    )

    assert out_of_domain["ood_score"][0] > nominal["ood_score"][0]
    assert out_of_domain["analysis_confidence"][0] < nominal["analysis_confidence"][0]
    assert out_of_domain["stall_risk"][0] >= nominal["stall_risk"][0]


def test_afc_confidence_is_stable_with_numpy_fp_exceptions_enabled():
    airfoil = asb.Airfoil("naca0012")
    old_err = np.geterr()
    np.seterr(all="raise")
    try:
        out_of_domain = airfoil.get_aero_from_afc_neuralfoil(
            alpha=80.0,
            Re=1e6,
            mach=0.05,
            C_mu=0.50,
            x_jet=0.1,
            theta_jet=30.0,
            model_size="xsmall",
        )
    finally:
        np.seterr(**old_err)

    assert np.all(np.isfinite(out_of_domain["analysis_confidence"]))
    assert out_of_domain["analysis_confidence"][0] <= 1.0


def test_afc_training_distance_ood_score_from_statistics():
    df = asb.make_fake_afc_dataset(n_cases=20, random_seed=13)
    stats = asb.make_afc_training_statistics(df)
    row = df.iloc[0]
    kulfan_parameters = {
        "upper_weights": [row[f"kulfan_upper_weights_{i}"] for i in range(8)],
        "lower_weights": [row[f"kulfan_lower_weights_{i}"] for i in range(8)],
        "leading_edge_weight": row["leading_edge_weight"],
        "TE_thickness": row["TE_thickness"],
    }

    near_score, near_distance = asb.training_distance_ood_score(
        kulfan_parameters,
        alpha=row["alpha"],
        Re=row["Re"],
        Mach=row["Mach"],
        C_mu=row["C_mu"],
        x_jet=row["x_jet"],
        theta_jet=row["theta_jet"],
        training_statistics=stats,
    )
    far_score, far_distance = asb.training_distance_ood_score(
        kulfan_parameters,
        alpha=90.0,
        Re=row["Re"] * 100,
        Mach=1.2,
        C_mu=0.5,
        x_jet=0.9,
        theta_jet=160.0,
        training_statistics=stats,
    )

    assert far_distance[0] > near_distance[0]
    assert far_score[0] >= near_score[0]


def test_airfoil_afc_neuralfoil_loads_external_weight_directory(tmp_path):
    weights_directory = tmp_path / "afc_weights"
    asb.generate_dummy_afc_neuralfoil_weights(
        output_directory=weights_directory,
        model_sizes=("small",),
    )

    aero = asb.Airfoil("naca0018").get_aero_from_afc_neuralfoil(
        alpha=4.0,
        Re=150e3,
        mach=0.02,
        C_mu=0.02,
        x_jet=0.1,
        theta_jet=30.0,
        model_size="small",
        weights_directory=weights_directory,
    )

    for key in ["CL", "CD", "CM", "analysis_confidence", "ood_score", "stall_risk"]:
        assert key in aero
        assert np.all(np.isfinite(aero[key]))


if __name__ == "__main__":
    test_afc_neuralfoil_zero_c_mu_matches_neuralfoil()
    test_afc_confidence_decreases_outside_nominal_domain()
    test_afc_training_distance_ood_score_from_statistics()
