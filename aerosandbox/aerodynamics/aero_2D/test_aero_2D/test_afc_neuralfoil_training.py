def test_afc_neuralfoil_training_imports_without_torch():
    from aerosandbox.aerodynamics.aero_2D.afc_neuralfoil_training.train_afc_neuralfoil import (
        TrainingConfig,
        build_feature_array,
    )
    import aerosandbox as asb

    df = asb.make_fake_afc_dataset(n_cases=4, random_seed=11)
    x = build_feature_array(df)

    assert x.shape == (4, 30)
    assert TrainingConfig(model_size="small").resolved_hidden_depth() >= 1


if __name__ == "__main__":
    test_afc_neuralfoil_training_imports_without_torch()

