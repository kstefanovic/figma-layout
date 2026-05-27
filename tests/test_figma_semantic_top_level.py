"""Tests for top-level Figma semantic child naming postprocess."""

from __future__ import annotations

import unittest

from figma_semantic import (
    _looks_like_word_vector_group,
    parse_top_level_names_object,
    postprocess_top_level_semantic_names,
)


def _child(
    index: int,
    node_id: str,
    role: str,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
) -> tuple[dict, dict]:
    child = {
        "index": index,
        "path": str(index),
        "id": node_id,
        "json": {
            "id": node_id,
            "type": "rectangle",
            "bounds": {"x": x, "y": y, "width": width, "height": height},
            "fills": [],
        },
    }
    name = {
        "index": index,
        "path": str(index),
        "id": node_id,
        "semantic_name": role,
        "confidence": 0.95,
    }
    return child, name


def _text_child(index: int, node_id: str, role: str, characters: str) -> tuple[dict, dict]:
    child, name = _child(index, node_id, role, x=40, y=240, width=300, height=120)
    child["json"]["type"] = "group"
    child["json"]["children"] = [
        {
            "id": f"{node_id}:text",
            "type": "text",
            "characters": characters,
            "bounds": {"x": 40, "y": 240, "width": 240, "height": 80},
            "fills": [],
        }
    ]
    return child, name


class TestFigmaSemanticTopLevel(unittest.TestCase):
    def test_word_vector_group_detector_top_strip_true(self) -> None:
        node = {
            "type": "vector",
            "bounds": {"x": 16, "y": -1, "width": 1166, "height": 230},
            "fills": [{"type": "SOLID"}],
        }
        self.assertTrue(_looks_like_word_vector_group(node, 1200, 1920))

    def test_word_vector_group_detector_brandlike_center_false(self) -> None:
        node = {
            "type": "group",
            "bounds": {"x": 280, "y": 340, "width": 620, "height": 145},
            "fills": [],
            "children": [
                {"type": "vector", "bounds": {"x": 280, "y": 340, "width": 620, "height": 145}, "fills": [{"type": "SOLID"}]}
            ],
        }
        self.assertFalse(_looks_like_word_vector_group(node, 1200, 1920))

    def test_word_vector_group_detector_large_plate_false(self) -> None:
        node = {
            "type": "rectangle",
            "bounds": {"x": 0, "y": 0, "width": 1200, "height": 1000},
            "fills": [{"type": "SOLID"}],
            "children": [{"type": "vector", "bounds": {"x": 0, "y": 0, "width": 1200, "height": 1000}, "fills": [{"type": "SOLID"}]}],
        }
        self.assertFalse(_looks_like_word_vector_group(node, 1200, 1920))

    def test_word_vector_group_detector_gradient_only_false(self) -> None:
        node = {
            "type": "vector",
            "bounds": {"x": 16, "y": -1, "width": 1166, "height": 230},
            "fills": [{"type": "GRADIENT_LINEAR"}],
        }
        self.assertFalse(_looks_like_word_vector_group(node, 1200, 1920))

    def test_word_vector_group_postprocess_overrides_background_shape(self) -> None:
        child, name = _child(7, "wvg", "background_shape", x=16.61, y=-1, width=1166.27, height=230.62)
        child["json"]["type"] = "vector"
        child["json"]["fills"] = [{"type": "SOLID"}]
        warnings: list[str] = []
        out = postprocess_top_level_semantic_names(
            [name],
            {"root": {"width": 1200, "height": 1920}, "children": [child]},
            warnings,
        )
        self.assertEqual(out[0]["semantic_name"], "word_vector_group")

    def test_word_vector_group_does_not_override_brand_group(self) -> None:
        child, name = _child(6, "brand", "brand_group", x=66, y=54, width=567, height=87)
        child["json"]["type"] = "group"
        child["json"]["children"] = [
            {"id": "word_1", "type": "vector", "bounds": {"x": 66, "y": 67, "width": 244, "height": 71}, "fills": [{"type": "SOLID"}]},
            {"id": "word_2", "type": "vector", "bounds": {"x": 420, "y": 67, "width": 208, "height": 62}, "fills": [{"type": "SOLID"}]},
            {
                "id": "logo",
                "type": "frame",
                "bounds": {"x": 325, "y": 57, "width": 81, "height": 81},
                "fills": [{"type": "SOLID", "visible": False}],
                "children": [
                    {"id": "logo_back", "type": "vector", "bounds": {"x": 325, "y": 57, "width": 81, "height": 81}, "fills": [{"type": "SOLID"}]},
                    {"id": "logo_fore", "type": "vector", "bounds": {"x": 341, "y": 78, "width": 49, "height": 45}, "fills": [{"type": "SOLID"}]},
                ],
            },
        ]
        warnings: list[str] = []
        out = postprocess_top_level_semantic_names(
            [name],
            {"root": {"width": 1080, "height": 1080}, "children": [child]},
            warnings,
        )
        self.assertEqual(out[0]["semantic_name"], "brand_group")

    def test_word_vector_group_recovers_low_confidence_false_brand_group(self) -> None:
        child, name = _child(8, "wvg", "brand_group", x=0.75, y=-2, width=598.83, height=125.43)
        child["json"]["type"] = "vector"
        child["json"]["fills"] = [{"type": "SOLID"}]
        name["confidence"] = 0.0
        warnings: list[str] = []
        out = postprocess_top_level_semantic_names(
            [name],
            {"root": {"width": 600, "height": 1024}, "children": [child]},
            warnings,
        )
        self.assertEqual(out[0]["semantic_name"], "word_vector_group")

    def test_duplicate_model_gradient_names_are_renumbered_by_position(self) -> None:
        top_child_0, name_0 = _child(0, "top", "background_gradient_1", x=100, y=10, width=300, height=120)
        top_child_1, name_1 = _child(1, "bottom", "background_gradient_1", x=40, y=400, width=300, height=120)
        top_child_2, name_2 = _child(2, "middle", "background_gradient_2", x=60, y=200, width=300, height=120)
        warnings: list[str] = []

        out = postprocess_top_level_semantic_names(
            [name_0, name_1, name_2],
            {
                "root": {"width": 800, "height": 600},
                "children": [top_child_0, top_child_1, top_child_2],
            },
            warnings,
        )

        by_id = {item["id"]: item["semantic_name"] for item in out}
        self.assertEqual(by_id["top"], "background_gradient_1")
        self.assertEqual(by_id["middle"], "background_gradient_2")
        self.assertEqual(by_id["bottom"], "background_gradient_3")
        self.assertEqual(len(set(by_id.values())), 3)

    def test_duplicate_background_shapes_keep_primary_unsuffixed(self) -> None:
        large_child, large_name = _child(0, "large", "background_shape", x=0, y=0, width=700, height=500)
        small_child, small_name = _child(1, "small", "background_shape", x=600, y=20, width=100, height=80)
        warnings: list[str] = []

        out = postprocess_top_level_semantic_names(
            [small_name, large_name],
            {
                "root": {"width": 800, "height": 600},
                "children": [large_child, small_child],
            },
            warnings,
        )

        by_id = {item["id"]: item["semantic_name"] for item in out}
        self.assertEqual(by_id["large"], "background_shape")
        self.assertEqual(by_id["small"], "background_shape_2")

    def test_offer_text_overrides_numbered_background_gradient(self) -> None:
        child, name = _text_child(0, "offer", "background_gradient_2", "299 ₽ 549")
        warnings: list[str] = []

        out = postprocess_top_level_semantic_names(
            [name],
            {
                "root": {"width": 800, "height": 600},
                "children": [child],
            },
            warnings,
        )

        self.assertEqual(out[0]["semantic_name"], "offer_group")

    def test_text_only_headline_overrides_false_hero_group(self) -> None:
        child, name = _child(0, "headline", "hero_group", x=149, y=190.39, width=782.13, height=245.84)
        child["json"]["type"] = "frame"
        child["json"]["children"] = [
            {
                "id": "t1",
                "type": "text",
                "characters": "Розыгрыш",
                "bounds": {"x": 160, "y": 200, "width": 200, "height": 60},
                "fills": [{"type": "SOLID"}],
            },
            {
                "id": "t2",
                "type": "text",
                "characters": "20 000 бонусов",
                "bounds": {"x": 160, "y": 260, "width": 420, "height": 80},
                "fills": [{"type": "SOLID"}],
            },
            {
                "id": "t3",
                "type": "text",
                "characters": "при заказе в ПВЗ Золотого Яблока",
                "bounds": {"x": 160, "y": 340, "width": 620, "height": 60},
                "fills": [{"type": "SOLID"}],
            },
        ]
        warnings: list[str] = []
        out = postprocess_top_level_semantic_names(
            [name],
            {"root": {"width": 1080, "height": 1350}, "children": [child]},
            warnings,
        )
        self.assertEqual(out[0]["semantic_name"], "headline_group")

    def test_compact_vector_wordmark_row_overrides_background_shape(self) -> None:
        child, name = _child(0, "brand", "background_group", x=40, y=36, width=377, height=55)
        child["json"]["type"] = "group"
        child["json"]["children"] = [
            {"id": "word_1", "type": "vector", "bounds": {"x": 40, "y": 43, "width": 164, "height": 48}, "fills": [{"type": "SOLID"}]},
            {"id": "word_2", "type": "vector", "bounds": {"x": 278, "y": 43, "width": 140, "height": 42}, "fills": [{"type": "SOLID"}]},
            {
                "id": "logo",
                "type": "frame",
                "bounds": {"x": 214, "y": 37, "width": 54, "height": 55},
                "fills": [{"type": "SOLID"}],
                "children": [
                    {"id": "logo_back", "type": "boolean operation", "bounds": {"x": 214, "y": 37, "width": 54, "height": 55}, "fills": [{"type": "SOLID"}]},
                    {"id": "logo_fore", "type": "vector", "bounds": {"x": 225, "y": 51, "width": 32, "height": 30}, "fills": [{"type": "SOLID"}]},
                ],
            },
        ]
        warnings: list[str] = []

        out = postprocess_top_level_semantic_names(
            [name],
            {"root": {"width": 770, "height": 562}, "children": [child]},
            warnings,
        )

        self.assertEqual(out[0]["semantic_name"], "brand_group")

    def test_compact_vector_wordmark_row_overrides_false_hero(self) -> None:
        child, name = _child(0, "brand", "hero_group", x=66, y=54, width=567, height=87)
        child["json"]["type"] = "group"
        child["json"]["children"] = [
            {"id": "word_1", "type": "vector", "bounds": {"x": 66, "y": 67, "width": 244, "height": 71}, "fills": [{"type": "SOLID"}]},
            {"id": "word_2", "type": "vector", "bounds": {"x": 420, "y": 67, "width": 208, "height": 62}, "fills": [{"type": "SOLID"}]},
            {
                "id": "logo",
                "type": "frame",
                "bounds": {"x": 325, "y": 57, "width": 81, "height": 81},
                "fills": [{"type": "SOLID", "visible": False}],
                "children": [
                    {"id": "logo_back", "type": "vector", "bounds": {"x": 325, "y": 57, "width": 81, "height": 81}, "fills": [{"type": "SOLID"}]},
                    {"id": "logo_fore", "type": "vector", "bounds": {"x": 341, "y": 78, "width": 49, "height": 45}, "fills": [{"type": "SOLID"}]},
                ],
            },
        ]
        warnings: list[str] = []

        out = postprocess_top_level_semantic_names(
            [name],
            {"root": {"width": 1080, "height": 1080}, "children": [child]},
            warnings,
        )

        self.assertEqual(out[0]["semantic_name"], "brand_group")

    def test_repeated_gradient_vector_cluster_overrides_background_gradient(self) -> None:
        child, name = _child(0, "decor", "background_group", x=-10, y=565, width=169, height=507)
        child["json"]["type"] = "group"
        child["json"]["children"] = []
        for i in range(6):
            x = -20 + i * 55
            y = 360 + (i % 3) * 38
            child["json"]["children"].append(
                {
                    "id": f"piece_{i}",
                    "type": "frame",
                    "bounds": {"x": x, "y": y, "width": 80, "height": 80},
                    "fills": [],
                    "children": [
                        {"id": f"solid_{i}", "type": "vector", "bounds": {"x": x + 8, "y": y + 4, "width": 40, "height": 70}, "fills": [{"type": "SOLID"}]},
                        {"id": f"grad_{i}", "type": "vector", "bounds": {"x": x + 8, "y": y + 4, "width": 40, "height": 70}, "fills": [{"type": "GRADIENT_LINEAR"}]},
                    ],
                }
            )
        warnings: list[str] = []

        out = postprocess_top_level_semantic_names(
            [name],
            {"root": {"width": 770, "height": 562}, "children": [child]},
            warnings,
        )

        self.assertEqual(out[0]["semantic_name"], "decoration_group")

    def test_repeated_gradient_vector_cluster_overrides_false_hero(self) -> None:
        child, name = _child(0, "decor", "hero_group", x=-50, y=952, width=326, height=798)
        child["json"]["type"] = "group"
        child["json"]["children"] = []
        for i in range(6):
            x = -64 + i * 95
            y = 610 + (i % 3) * 48
            child["json"]["children"].append(
                {
                    "id": f"piece_{i}",
                    "type": "frame",
                    "bounds": {"x": x, "y": y, "width": 145, "height": 145},
                    "fills": [],
                    "children": [
                        {"id": f"solid_{i}", "type": "vector", "bounds": {"x": x + 12, "y": y + 8, "width": 73, "height": 127}, "fills": [{"type": "SOLID"}]},
                        {"id": f"grad_{i}", "type": "vector", "bounds": {"x": x + 12, "y": y + 8, "width": 73, "height": 127}, "fills": [{"type": "GRADIENT_LINEAR"}]},
                    ],
                }
            )
        warnings: list[str] = []

        out = postprocess_top_level_semantic_names(
            [name],
            {"root": {"width": 1080, "height": 1080}, "children": [child]},
            warnings,
        )

        self.assertEqual(out[0]["semantic_name"], "decoration_group")

    def test_top_level_parser_trusts_exact_child_id_over_conflicting_path(self) -> None:
        children = [
            {"index": 0, "path": "0", "id": "decor"},
            {"index": 3, "path": "3", "id": "hero"},
        ]
        warnings: list[str] = []

        out = parse_top_level_names_object(
            '{"top_level_names":[{"index":0,"path":"0","id":"hero","semantic_name":"hero_group","confidence":0.95}]}',
            {"children": children},
            warnings,
        )

        self.assertEqual(out[0]["id"], "hero")
        self.assertEqual(out[0]["path"], "3")
        self.assertEqual(out[0]["index"], 3)


if __name__ == "__main__":
    unittest.main()
