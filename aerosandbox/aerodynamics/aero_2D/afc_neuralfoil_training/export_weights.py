from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Union

import numpy as onp

from aerosandbox.aerodynamics.aero_2D.afc_neuralfoil_training.train_afc_neuralfoil import (
    AFCNeuralFoilMLP,
    TrainingConfig,
    _require_torch,
    torch,
)


def _as_numpy(tensor) -> onp.ndarray:
    return tensor.detach().cpu().numpy().astype(onp.float64)


def fold_normalizers_into_linear_layers(model: AFCNeuralFoilMLP) -> Dict[str, onp.ndarray]:
    """
    Converts a normalized PyTorch AFC-NeuralFoil model into plain NeuralFoil-style linear-layer arrays.

    The exported network accepts already-encoded physical features directly, exactly as produced by
    `make_afc_neuralfoil_feature_matrix()`.
    """
    _require_torch()

    weights = [_as_numpy(layer.weight) for layer in model.linears]
    biases = [_as_numpy(layer.bias) for layer in model.linears]

    input_mean = _as_numpy(model.input_mean)
    input_std = _as_numpy(model.input_std)
    output_mean = _as_numpy(model.output_mean)
    output_std = _as_numpy(model.output_std)

    weights[0] = weights[0] / input_std[None, :]
    biases[0] = biases[0] - weights[0] @ input_mean

    weights[-1] = output_std[:, None] * weights[-1]
    biases[-1] = output_std * biases[-1] + output_mean

    params = {}
    for i, (weight, bias) in enumerate(zip(weights, biases)):
        net_index = 2 * i
        params[f"net.{net_index}.weight"] = weight
        params[f"net.{net_index}.bias"] = bias
    return params


def load_model_from_checkpoint(checkpoint_path: Union[str, Path], device: str = "cpu"):
    _require_torch()
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    config = TrainingConfig(**checkpoint["config"])
    architecture = checkpoint["architecture"]

    model = AFCNeuralFoilMLP(
        hidden_width=architecture["hidden_width"],
        hidden_depth=architecture["hidden_depth"],
        input_mean=onp.zeros(architecture["input_dim"], dtype=onp.float32),
        input_std=onp.ones(architecture["input_dim"], dtype=onp.float32),
        output_mean=onp.zeros(architecture["output_dim"], dtype=onp.float32),
        output_std=onp.ones(architecture["output_dim"], dtype=onp.float32),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, config, checkpoint


def export_checkpoint_to_npz(
    checkpoint_path: Union[str, Path],
    output_directory: Union[str, Path],
    model_size: str = "small",
    *,
    overwrite: bool = False,
    device: str = "cpu",
) -> Path:
    """
    Exports a training checkpoint to `nn-{model_size}.npz` for the aerosandbox.numpy inference path.
    """
    checkpoint_path = Path(checkpoint_path)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / f"nn-{model_size}.npz"

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f'Output file "{output_path}" already exists. Pass `overwrite=True` to replace it.'
        )

    model, config, checkpoint = load_model_from_checkpoint(checkpoint_path, device=device)
    params = fold_normalizers_into_linear_layers(model)
    onp.savez(output_path, **params)

    sidecar_path = output_path.with_suffix(".json")
    with open(sidecar_path, "w") as f:
        json.dump(
            {
                "source_checkpoint": str(checkpoint_path),
                "model_size": model_size,
                "config": checkpoint.get("config", {}),
                "architecture": checkpoint.get("architecture", {}),
                "epoch": checkpoint.get("epoch"),
                "metrics": checkpoint.get("metrics", {}),
            },
            f,
            indent=2,
        )

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export an AFC-NeuralFoil PyTorch checkpoint to .npz weights."
    )
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint_best.pt.")
    parser.add_argument(
        "--output-directory",
        type=str,
        required=True,
        help="Directory where nn-{model_size}.npz will be written.",
    )
    parser.add_argument("--model-size", type=str, default="small")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = export_checkpoint_to_npz(
        checkpoint_path=args.checkpoint,
        output_directory=args.output_directory,
        model_size=args.model_size,
        overwrite=args.overwrite,
        device=args.device,
    )
    print(f"Exported AFC-NeuralFoil weights to {output_path}")


if __name__ == "__main__":
    main()
