"""Predict normalized semantic role bbox priors for one source banner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from .graph_builder import FEATURE_NAMES, build_graph
from .model import GNNLayoutPredictor
from .orientation import get_orientation, orientation_to_onehot
from .roles import IDX_TO_ROLE, NUM_ROLES


def predict_priors(
    checkpoint_path: str | Path,
    source_path: str | Path,
    target_width: float,
    target_height: float,
) -> dict[str, Any]:
    source = load_banner(Path(source_path))
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config", {})
    model = GNNLayoutPredictor(
        in_channels=int(config.get("in_channels", len(FEATURE_NAMES))),
        hidden=int(config.get("hidden", 128)),
        dropout=float(config.get("dropout", 0.15)),
    )
    model.load_state_dict(checkpoint["model_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    orientation = get_orientation(target_width, target_height)
    data = build_graph(source)
    data.target_size = torch.tensor(
        [float(target_width) / 3000.0, float(target_height) / 3000.0, float(target_width) / float(target_height)],
        dtype=torch.float32,
    )
    data.target_orientation_onehot = torch.tensor(orientation_to_onehot(orientation), dtype=torch.float32)
    data = data.to(device)

    with torch.no_grad():
        pred = model(data).detach().cpu().view(NUM_ROLES, 4).numpy()

    priors: dict[str, dict[str, float]] = {}
    for idx, role in IDX_TO_ROLE.items():
        x, y, w, h = [float(v) for v in pred[idx]]
        priors[role] = {"x": x, "y": y, "w": w, "h": h, "confidence": 1.0}
    return {
        "orientation": orientation,
        "target_width": int(round(float(target_width))),
        "target_height": int(round(float(target_height))),
        "priors": priors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--target-width", required=True, type=float)
    parser.add_argument("--target-height", required=True, type=float)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    result = predict_priors(args.checkpoint, args.source, args.target_width, args.target_height)
    args.output.parent.mkdir(parents=True, exist_ok=True) if args.output.parent != Path(".") else None
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Wrote priors: {args.output}")


def load_banner(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        if not data:
            raise ValueError(f"{path} contains an empty list")
        data = data[0]
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object or non-empty list")
    return data


if __name__ == "__main__":
    main()
