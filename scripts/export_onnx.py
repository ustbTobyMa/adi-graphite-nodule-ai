#!/usr/bin/env python3
"""Export the released PyTorch U-Net checkpoint to ONNX for browser inference."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from graphite_nodule_analyzer import UNet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    base_channels = int(checkpoint.get("config", {}).get("base_channels", 32))
    model = UNet(base_channels=base_channels)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dummy = torch.zeros(1, 1, 512, 512, dtype=torch.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        args.output,
        input_names=["image"],
        output_names=["logits"],
        external_data=False,
        dynamic_axes={
            "image": {2: "height", 3: "width"},
            "logits": {2: "height", 3: "width"},
        },
        opset_version=17,
    )


if __name__ == "__main__":
    main()
