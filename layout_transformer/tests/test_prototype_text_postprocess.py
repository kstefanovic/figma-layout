from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any

from layout_transformer.src.postprocess import find_by_role, get_bounds
from layout_transformer.src.predict import StructuralLayoutTransformerService
from layout_transformer.src.prototype_index import load_prototypes, select_target_prototype_match

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "data" / "clean_families" / "2_clean_fixed_semantic.json"
PROTOTYPES_PATH = ROOT / "data" / "prototypes" / "layout_prototypes.json"
CHECKPOINT_PATH = ROOT / "checkpoints" / "layout_transformer_structural.pt"

BRAND_DIRECT = frozenset(
    {"brand_name_first_part_1", "brand_name_first_part_2", "brand_name_second", "logo"}
)


class PrototypeTextPostprocessTest(unittest.TestCase):
    def test_prototype_relative_text_640x720(self) -> None:
        if not SOURCE_PATH.exists() or not CHECKPOINT_PATH.exists() or not PROTOTYPES_PATH.exists():
            self.skipTest("layout transformer fixtures/checkpoint/prototypes are not available")

        source = _load_source_frame(SOURCE_PATH)
        prototypes = load_prototypes(PROTOTYPES_PATH)
        match = select_target_prototype_match(source, 640.0, 720.0, prototypes)
        self.assertIsNotNone(match)
        prototype = match["prototype"]
        self.assertIsInstance(prototype, dict)
        rels = prototype.get("child_relative_bboxes") or {}
        self.assertIn("headline", rels)
        self.assertIn("subheadline_delivery_time", rels)
        ts = prototype.get("text_styles") or {}
        headline_style = ts.get("headline")
        self.assertIsInstance(headline_style, dict)
        for key in (
            "fontSize",
            "fontName",
            "textAlignHorizontal",
            "textAlignVertical",
            "textAutoResize",
            "lineHeight",
            "letterSpacing",
            "fills",
            "opacity",
        ):
            self.assertIn(key, headline_style, f"prototype text_styles.headline missing {key}")

        service = StructuralLayoutTransformerService(CHECKPOINT_PATH, device="cpu")
        final_json = service.predict(source, 640, 720)

        self.assertEqual(service.last_report.get("postprocess_mode"), "prototype_relative_text")
        dbg = service.last_report.get("debug") or {}
        self.assertEqual(dbg.get("text_postprocess_mode"), "prototype_relative_to_parent")
        self.assertEqual(
            dbg.get("text_roles_applied"),
            ["headline", "subheadline_delivery_time", "legal_text", "age_badge"],
        )

        hg = _find_first_by_name(final_json, "headline_group")
        self.assertIsNotNone(hg)
        child_names = [c.get("name") for c in (hg.get("children") or []) if isinstance(c, dict)]
        self.assertEqual(set(child_names), {"headline", "subheadline_delivery_time"})

        bg = _find_first_by_name(final_json, "brand_group")
        self.assertIsNotNone(bg)
        direct = {c.get("name") for c in (bg.get("children") or []) if isinstance(c, dict)}
        self.assertTrue(BRAND_DIRECT.issubset(direct))

        headline = find_by_role(final_json, "headline")
        self.assertIsNotNone(headline)
        self.assertIsInstance(headline.get("fontSize"), (int, float))
        self.assertGreater(float(headline["fontSize"]), 0.0)
        self.assertEqual(
            _text_style_subset(headline),
            {
                "fontSize": 36.0,
                "textAutoResize": "NONE",
                "textAlignHorizontal": "CENTER",
                "textAlignVertical": "CENTER",
            },
        )
        self.assertEqual(
            _text_style_subset(find_by_role(final_json, "subheadline_delivery_time")),
            {
                "fontSize": 14.0,
                "textAutoResize": "NONE",
                "textAlignHorizontal": "CENTER",
                "textAlignVertical": "CENTER",
            },
        )
        self.assertEqual(
            _text_style_subset(find_by_role(final_json, "legal_text")),
            {
                "fontSize": 6.0,
                "textAutoResize": "NONE",
                "textAlignHorizontal": "CENTER",
                "textAlignVertical": "CENTER",
            },
        )
        age_badge = find_by_role(final_json, "age_badge")
        self.assertEqual(
            _text_style_subset(age_badge),
            {
                "fontSize": 25.0,
                "textAutoResize": "NONE",
                "textAlignHorizontal": "CENTER",
                "textAlignVertical": "CENTER",
            },
        )
        pb = get_bounds(hg)
        hb = get_bounds(headline)
        rh = rels["headline"]
        exp_x = pb["x"] + float(rh["x"]) * pb["width"]
        exp_y = pb["y"] + float(rh["y"]) * pb["height"]
        self.assertLess(abs(hb["x"] - exp_x), 1.5)
        self.assertLess(abs(hb["y"] - exp_y), 1.5)

        bg_bounds = get_bounds(bg)
        for role in ("brand_name_first_part_1", "brand_name_first_part_2", "brand_name_second", "logo"):
            n = find_by_role(final_json, role)
            self.assertIsNotNone(n)
            bb = get_bounds(n)
            cx = bb["x"] + bb["width"] / 2.0
            cy = bb["y"] + bb["height"] / 2.0
            self.assertGreaterEqual(cx, bg_bounds["x"] - 5)
            self.assertLessEqual(cx, bg_bounds["x"] + bg_bounds["width"] + 5)
            self.assertGreaterEqual(cy, bg_bounds["y"] - 5)
            self.assertLessEqual(cy, bg_bounds["y"] + bg_bounds["height"] + 5)

        logo_node = find_by_role(final_json, "logo")
        lb = get_bounds(logo_node)
        for role in ("logo_back", "logo_fore"):
            n = find_by_role(final_json, role)
            self.assertIsNotNone(n)
            bb = get_bounds(n)
            cx = bb["x"] + bb["width"] / 2.0
            cy = bb["y"] + bb["height"] / 2.0
            self.assertGreaterEqual(cx, lb["x"] - 2)
            self.assertLessEqual(cx, lb["x"] + lb["width"] + 2)
            self.assertGreaterEqual(cy, lb["y"] - 2)
            self.assertLessEqual(cy, lb["y"] + lb["height"] + 2)

        self.assertEqual(service.last_report.get("legal_text_bbox_source"), "transformer")


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


def _find_first_by_name(frame: dict[str, Any], role: str) -> dict[str, Any] | None:
    found: dict[str, Any] | None = None

    def walk(node: Any) -> None:
        nonlocal found
        if found is not None or not isinstance(node, dict):
            return
        if node.get("name") == role:
            found = node
            return
        for child in node.get("children") or []:
            walk(child)

    walk(frame)
    return found


def _text_style_subset(node: dict[str, Any] | None) -> dict[str, Any]:
    assert node is not None
    keys = [
        "fontSize",
        "textAutoResize",
        "textAlignHorizontal",
        "textAlignVertical",
    ]
    return {key: node.get(key) for key in keys}


if __name__ == "__main__":
    unittest.main()
