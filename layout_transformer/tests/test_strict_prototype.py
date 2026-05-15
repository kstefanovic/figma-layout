from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any

from layout_transformer.src.predict import StructuralLayoutTransformerService


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "data" / "clean_families" / "2_clean_fixed_semantic.json"
EXPECTED_PATH = Path(__file__).resolve().parent / "fixtures" / "expected_strict_640x720.json"
CHECKPOINT_PATH = ROOT / "checkpoints" / "layout_transformer_structural.pt"


class StrictPrototypeTest(unittest.TestCase):
    def test_strict_prototype_640x720_matches_expected_bboxes(self) -> None:
        if not SOURCE_PATH.exists() or not CHECKPOINT_PATH.exists():
            self.skipTest("layout transformer fixtures/checkpoint are not available")

        source = _load_source_frame(SOURCE_PATH)
        expected = json.loads(EXPECTED_PATH.read_text(encoding="utf-8"))
        service = StructuralLayoutTransformerService(CHECKPOINT_PATH, device="cpu")

        final_json = service.predict(source, 640, 720)

        self.assertEqual(service.last_report.get("postprocess_mode"), "strict_prototype")
        self.assertEqual(service.last_report.get("prototype_id"), "2_clean_fixed_semantic:45")

        nodes = _nodes_by_role(final_json)
        failures: list[str] = []
        for role, expected_bounds in expected.items():
            actual = nodes[role]["bounds"]
            for key, expected_value in expected_bounds.items():
                diff = abs(float(actual[key]) - float(expected_value))
                if diff > 2.0:
                    failures.append(f"{role}.{key}: actual={actual[key]} expected={expected_value} diff={diff:.2f}")
        self.assertFalse(failures, "\n".join(failures))


def _load_source_frame(path: Path) -> dict[str, Any]:
    frames = json.loads(path.read_text(encoding="utf-8"))
    for frame in frames:
        bounds = frame.get("bounds") or {}
        if round(float(bounds.get("width", 0))) == 1280 and round(float(bounds.get("height", 0))) == 576:
            names = set(_nodes_by_role(frame))
            if {"headline_group", "brand_group", "age_badge", "star_decoration_1"}.issubset(names):
                return frame
    raise AssertionError("could not find 1280x576 source frame in family 2 fixture")


def _nodes_by_role(frame: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}

    def walk(node: Any) -> None:
        if not isinstance(node, dict):
            return
        name = node.get("name")
        bounds = node.get("bounds")
        if isinstance(name, str) and isinstance(bounds, dict):
            current = out.get(name)
            area = float(bounds.get("width", 0)) * float(bounds.get("height", 0))
            current_bounds = current.get("bounds") if current else None
            current_area = (
                float(current_bounds.get("width", 0)) * float(current_bounds.get("height", 0))
                if isinstance(current_bounds, dict)
                else -1
            )
            if current is None or area > current_area:
                out[name] = node
        for child in node.get("children") or []:
            walk(child)

    walk(frame)
    return out


if __name__ == "__main__":
    unittest.main()
