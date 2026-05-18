"""Acceptance tests for strict Figma semantic naming."""

from __future__ import annotations

import copy
import unittest

from figma_semantic_strict import (
    FORBIDDEN_OUTPUT_NAMES,
    assign_brand_row_child_roles,
    build_semantic_json_from_strict_names,
    extract_node_features,
    normalize_roles,
    prelabel_roles,
    resolve_role_conflicts,
    run_strict_semantic_naming,
    validate_and_autofix_roles,
)


def _frame(w: float = 1080, h: float = 540) -> dict:
    return {
        "id": "root",
        "type": "frame",
        "bounds": {"x": 0, "y": 0, "width": w, "height": h},
        "mid_parent_ids": [],
        "mid_child_ids": [],
    }


def _rect(nid: str, x: float, y: float, w: float, h: float, fills: list, parents: list | None = None) -> dict:
    return {
        "id": nid,
        "type": "rectangle",
        "bounds": {"x": x, "y": y, "width": w, "height": h},
        "fills": fills,
        "mid_parent_ids": parents or ["root"],
        "mid_child_ids": [],
    }


def _text(nid: str, chars: str, font_size: float, x: float = 48, y: float = 200) -> dict:
    return {
        "id": nid,
        "type": "text",
        "characters": chars,
        "fontSize": font_size,
        "fontName": {"family": "YS Geo", "style": "Black"},
        "bounds": {"x": x, "y": y, "width": 400, "height": 120},
        "fills": [{"type": "SOLID", "color": {"r": 1, "g": 1, "b": 1}}],
        "mid_parent_ids": ["root"],
        "mid_child_ids": [],
    }


def _gradient_rect(nid: str) -> dict:
    return _rect(
        nid,
        0,
        0,
        1080,
        540,
        [{"type": "GRADIENT_LINEAR", "visible": True}],
    )


def _hero_rect(nid: str = "hero") -> dict:
    return _rect(
        nid,
        400,
        0,
        700,
        540,
        [
            {
                "type": "IMAGE",
                "imageHash": "abc123",
                "scaleMode": "FILL",
            }
        ],
    )


def _cherry_like_mid() -> list[dict]:
    root = _frame()
    root["mid_child_ids"] = ["bg", "hero", "brand_g", "hg", "legal", "age"]
    bg = _rect("bg", 0, 0, 1080, 540, [{"type": "SOLID", "color": {"r": 0, "g": 0.5, "b": 1}}])
    hero = _hero_rect("hero")
    brand_g = {
        "id": "brand_g",
        "type": "group",
        "bounds": {"x": 48, "y": 48, "width": 300, "height": 80},
        "fills": [],
        "mid_parent_ids": ["root"],
        "mid_child_ids": ["bn1", "bn2", "logo"],
    }
    bn1 = {
        "id": "bn1",
        "type": "vector",
        "bounds": {"x": 48, "y": 48, "width": 80, "height": 40},
        "fills": [{"type": "SOLID"}],
        "mid_parent_ids": ["root", "brand_g"],
        "mid_child_ids": [],
    }
    bn2 = {**bn1, "id": "bn2", "bounds": {"x": 140, "y": 48, "width": 100, "height": 40}}
    logo = {**bn1, "id": "logo", "bounds": {"x": 260, "y": 40, "width": 60, "height": 60}}
    hg = {
        "id": "hg",
        "type": "group",
        "bounds": {"x": 48, "y": 140, "width": 500, "height": 200},
        "fills": [],
        "mid_parent_ids": ["root"],
        "mid_child_ids": ["hl", "sub"],
    }
    hl = _text("hl", "Праздничная вишня", 72)
    hl["mid_parent_ids"] = ["root", "hg"]
    sub = _text("sub", "с доставкой от 15 минут", 24, y=280)
    sub["mid_parent_ids"] = ["root", "hg"]
    legal = _text("legal", "ООО «Яндекс» …", 12, y=480)
    age = _text("age", "0+", 18, x=900, y=480)
    return [root, bg, hero, brand_g, bn1, bn2, logo, hg, hl, sub, legal, age]


class TestFigmaSemanticStrict(unittest.TestCase):
    def test_product_image_alias_normalizes_to_hero_image(self) -> None:
        cherry_like_mid = _cherry_like_mid()
        features = extract_node_features(cherry_like_mid)
        names = prelabel_roles(cherry_like_mid, features)
        names["hero"] = "product_image"
        normalized = normalize_roles(cherry_like_mid, features, names)
        self.assertEqual(normalized["hero"], "hero_image")
        self.assertNotIn("product_image", normalized.values())

    def test_gradient_not_background_shape(self) -> None:
        cherry_like_mid = _cherry_like_mid()
        mid = copy.deepcopy(cherry_like_mid)
        grad = _gradient_rect("grad")
        grad["mid_parent_ids"] = ["root"]
        mid.append(grad)
        root = mid[0]
        root["mid_child_ids"] = list(root["mid_child_ids"]) + ["grad"]
        result = run_strict_semantic_naming(mid)
        self.assertIn(result.names["grad"], ("background_gradient_1", "background_gradient_2"))
        self.assertNotEqual(result.names["grad"], "background_shape")

    def test_age_badge_stays_age_badge(self) -> None:
        result = run_strict_semantic_naming(_cherry_like_mid())
        self.assertEqual(result.names["age"], "age_badge")

    def test_legal_text_stays_legal_text(self) -> None:
        result = run_strict_semantic_naming(_cherry_like_mid())
        self.assertEqual(result.names["legal"], "legal_text")

    def test_decoration_group_with_hero_child_promotes_hero(self) -> None:
        root = _frame()
        wrapper = {
            "id": "wrap",
            "type": "group",
            "bounds": {"x": 0, "y": 0, "width": 800, "height": 500},
            "fills": [],
            "mid_parent_ids": ["root"],
            "mid_child_ids": ["hero"],
        }
        hero = _hero_rect("hero")
        hero["mid_parent_ids"] = ["root", "wrap"]
        root["mid_child_ids"] = ["wrap"]
        mid = [root, wrapper, hero]
        features = extract_node_features(mid)
        names = {"root": "banner_root", "wrap": "decoration_group", "hero": "unassigned"}
        resolved, fixes = resolve_role_conflicts(mid, features, names)
        self.assertEqual(resolved["wrap"], "hero_image")
        self.assertEqual(resolved["hero"], "unassigned")
        self.assertTrue(
            any(f["reason"] in ("hero_wrapper_is_hero", "hero_inner_image_leaf", "promote_largest_hero") for f in fixes)
        )

    def test_hero_mislabeled_unassigned_gets_fixed(self) -> None:
        root = _frame()
        hero = _hero_rect("hero")
        root["mid_child_ids"] = ["hero"]
        mid = [root, hero]
        features = extract_node_features(mid)
        names = {"root": "banner_root", "hero": "unassigned"}
        resolved, _ = resolve_role_conflicts(mid, features, names)
        self.assertEqual(resolved["hero"], "hero_image")

    def test_forbidden_names_absent_in_final_pipeline(self) -> None:
        result = run_strict_semantic_naming(_cherry_like_mid())
        allowed = {
            "banner_root",
            "hero_image",
            "background_shape",
            "background_gradient_1",
            "background_gradient_2",
            "brand_group",
            "brand_name_first_part_1",
            "brand_name_first_part_2",
            "brand_name_second",
            "logo",
            "logo_back",
            "logo_fore",
            "headline_group",
            "headline",
            "subheadline_delivery_time",
            "legal_text",
            "age_badge",
            "star_decoration_1",
            "star_decoration_2",
            "offer_group",
            "price_text",
            "old_price_text",
            "discount_badge",
            "product_label",
            "unassigned",
        }
        for role in result.names.values():
            self.assertNotIn(role, FORBIDDEN_OUTPUT_NAMES)
            self.assertIn(role, allowed)

    def test_rich_metadata_preserved_in_tree(self) -> None:
        cherry_like_mid = _cherry_like_mid()
        result = run_strict_semantic_naming(cherry_like_mid)
        warnings: list[str] = []
        tree = build_semantic_json_from_strict_names(cherry_like_mid, result.names, warnings)

        def find(node: dict, role: str) -> dict | None:
            if node.get("name") == role:
                return node
            for ch in node.get("children") or []:
                if isinstance(ch, dict):
                    hit = find(ch, role)
                    if hit:
                        return hit
            return None

        hero_node = find(tree, "hero_image")
        self.assertIsNotNone(hero_node)
        fills = hero_node.get("fills") or []
        self.assertTrue(any(f.get("type") == "IMAGE" for f in fills))
        self.assertEqual(fills[0].get("imageHash"), "abc123")

        hl_node = find(tree, "headline")
        self.assertIsNotNone(hl_node)
        self.assertEqual(hl_node.get("fontSize"), 72)
        self.assertTrue(hl_node.get("characters"))

    def test_brand_row_logo_and_wordmarks_by_position(self) -> None:
        """Lavka-style row: word parts left of logo; logo_back/logo_fore inside logo boolean."""
        root = _frame(500, 300)
        brand = {
            "id": "brand",
            "type": "frame",
            "bounds": {"x": 48, "y": 151, "width": 395, "height": 57},
            "fills": [],
            "mid_parent_ids": ["root"],
            "mid_child_ids": ["part1", "part2", "logo", "second"],
        }
        part1 = {
            "id": "part1",
            "type": "vector",
            "bounds": {"x": 49, "y": 158, "width": 28, "height": 43},
            "fills": [{"type": "SOLID"}],
            "mid_parent_ids": ["root", "brand"],
            "mid_child_ids": [],
        }
        part2 = {
            "id": "part2",
            "type": "vector",
            "bounds": {"x": 83, "y": 169, "width": 137, "height": 38},
            "fills": [{"type": "SOLID"}],
            "mid_parent_ids": ["root", "brand"],
            "mid_child_ids": [],
        }
        logo = {
            "id": "logo",
            "type": "boolean operation",
            "bounds": {"x": 230, "y": 152, "width": 57, "height": 56},
            "fills": [{"type": "SOLID"}],
            "mid_parent_ids": ["root", "brand"],
            "mid_child_ids": ["logo_back", "logo_wrap"],
        }
        logo_back = {
            "id": "logo_back",
            "type": "vector",
            "bounds": {"x": 230, "y": 152, "width": 57, "height": 56},
            "fills": [{"type": "SOLID"}],
            "mid_parent_ids": ["root", "brand", "logo"],
            "mid_child_ids": [],
        }
        logo_wrap = {
            "id": "logo_wrap",
            "type": "boolean operation",
            "bounds": {"x": 241, "y": 166, "width": 34, "height": 31},
            "fills": [{"type": "SOLID"}],
            "mid_parent_ids": ["root", "brand", "logo"],
            "mid_child_ids": ["logo_fore"],
        }
        logo_fore = {
            "id": "logo_fore",
            "type": "vector",
            "bounds": {"x": 241, "y": 166, "width": 34, "height": 31},
            "fills": [{"type": "SOLID"}],
            "mid_parent_ids": ["root", "brand", "logo", "logo_wrap"],
            "mid_child_ids": [],
        }
        second = {
            "id": "second",
            "type": "vector",
            "bounds": {"x": 297, "y": 158, "width": 146, "height": 43},
            "fills": [{"type": "SOLID"}],
            "mid_parent_ids": ["root", "brand"],
            "mid_child_ids": [],
        }
        root["mid_child_ids"] = ["brand"]
        mid = [root, brand, part1, part2, logo, logo_back, logo_wrap, logo_fore, second]
        features = extract_node_features(mid)
        names = {b["id"]: "unassigned" for b in mid}
        names["root"] = "banner_root"
        names["brand"] = "brand_group"
        out, _ = assign_brand_row_child_roles(mid, features, names, [])
        self.assertEqual(out["part1"], "brand_name_first_part_1")
        self.assertEqual(out["part2"], "brand_name_first_part_2")
        self.assertEqual(out["logo"], "logo")
        self.assertEqual(out["logo_back"], "logo_back")
        self.assertEqual(out["logo_wrap"], "logo_fore")
        self.assertEqual(out["logo_fore"], "logo_fore")
        self.assertEqual(out["second"], "brand_name_second")

    def test_star_nodes_numbered_ltr_ttb(self) -> None:
        root = _frame(1000, 500)
        left = {
            "id": "s_left",
            "type": "star",
            "bounds": {"x": 50, "y": 400, "width": 40, "height": 40},
            "fills": [{"type": "SOLID"}],
            "mid_parent_ids": ["root"],
            "mid_child_ids": [],
        }
        right = {
            "id": "s_right",
            "type": "star",
            "bounds": {"x": 800, "y": 100, "width": 40, "height": 40},
            "fills": [{"type": "SOLID"}],
            "mid_parent_ids": ["root"],
            "mid_child_ids": [],
        }
        root["mid_child_ids"] = ["s_left", "s_right"]
        mid = [root, left, right]
        result = run_strict_semantic_naming(mid)
        self.assertEqual(result.names["s_left"], "star_decoration_1")
        self.assertEqual(result.names["s_right"], "star_decoration_2")

    def test_legal_text_not_promoted_over_headline(self) -> None:
        """Long legal copy with medium font must stay legal_text; largest promo line stays headline."""
        root = _frame(1080, 540)
        legal = _text(
            "legal",
            "ООО «Яндекс.Лавка» ОГРН 1187746479250 " + "x" * 60,
            48,
            y=400,
        )
        headline = _text("hl", "Готовая еда на любой вкус", 120, y=100)
        root["mid_child_ids"] = ["legal", "hl"]
        mid = [root, legal, headline]
        result = run_strict_semantic_naming(mid)
        self.assertEqual(result.names["legal"], "legal_text")
        self.assertEqual(result.names["hl"], "headline")
        self.assertEqual(result.names["root"], "banner_root")

    def test_validator_autofixes_missing_hero(self) -> None:
        root = _frame()
        hero = _hero_rect("hero")
        root["mid_child_ids"] = ["hero"]
        mid = [root, hero]
        features = extract_node_features(mid)
        names = {"root": "banner_root", "hero": "unassigned"}
        final, validation = validate_and_autofix_roles(mid, features, names)
        self.assertEqual(final["hero"], "hero_image")
        self.assertTrue(any("hero_image_missing" in w for w in validation.warnings))


if __name__ == "__main__":
    unittest.main()
