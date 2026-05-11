"""PyTorch Geometric dataset for directed layout-transfer pairs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch_geometric.data import Dataset

from .graph_builder import build_graph
from .orientation import get_orientation, orientation_to_onehot
from .roles import NUM_ROLES
from .semantic_utils import extract_role_boxes, extract_role_mask, get_banner_size


class GNNLayoutDataset(Dataset):
    """Reads pairs.jsonl and returns source graphs with target role-box labels."""

    def __init__(self, pairs_path: str | Path, rows: list[dict[str, Any]] | None = None):
        super().__init__()
        self.pairs_path = Path(pairs_path)
        self.rows = rows if rows is not None else self._read_rows(self.pairs_path)

    def len(self) -> int:
        return len(self.rows)

    def get(self, idx: int):
        row = self.rows[idx]
        source = row.get("source")
        target = row.get("target")
        if not isinstance(source, dict) or not isinstance(target, dict):
            raise ValueError(f"pair row {idx} is missing source/target objects")

        data = build_graph(source)
        target_width = float(row.get("target_width") or 0)
        target_height = float(row.get("target_height") or 0)
        if target_width <= 0 or target_height <= 0:
            target_width, target_height = get_banner_size(target)
        if target_width <= 0 or target_height <= 0:
            raise ValueError(f"pair row {idx} has invalid target size")
        target_orientation = get_orientation(target_width, target_height)

        data.target_size = torch.tensor(
            [target_width / 3000.0, target_height / 3000.0, target_width / target_height],
            dtype=torch.float32,
        )
        data.target_orientation_onehot = torch.tensor(
            orientation_to_onehot(target_orientation),
            dtype=torch.float32,
        )
        data.y_boxes = torch.tensor(extract_role_boxes(target), dtype=torch.float32).view(NUM_ROLES, 4)
        data.y_mask = torch.tensor(extract_role_mask(target), dtype=torch.float32).view(NUM_ROLES)
        data.source_id = str(row.get("source_id") or "")
        data.target_id = str(row.get("target_id") or "")
        data.family_key = str(row.get("family_key") or "")
        return data

    @staticmethod
    def _read_rows(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(path)
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL at {path}:{line_no}: {exc}") from exc
                if isinstance(row, dict):
                    rows.append(row)
        return rows
