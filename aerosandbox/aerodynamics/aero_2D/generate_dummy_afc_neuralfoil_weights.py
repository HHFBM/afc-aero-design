"""
Generates deterministic dummy AFC-NeuralFoil MLP weights.

These weights are placeholders for interface testing only. They are not trained aerodynamic models.
"""

from aerosandbox.aerodynamics.aero_2D.afc_neuralfoil import (
    DEFAULT_AFC_NEURALFOIL_WEIGHTS_DIR,
    generate_dummy_afc_neuralfoil_weights,
)


if __name__ == "__main__":
    written = generate_dummy_afc_neuralfoil_weights(
        output_directory=DEFAULT_AFC_NEURALFOIL_WEIGHTS_DIR,
        model_sizes=("small", "medium", "large"),
        overwrite=True,
    )
    for path in written:
        print(path)
