"""Draw a simple rectangle preview for semantic role bounds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .extract import flatten_semantic_nodes, get_canvas_size
from .roles import CHILD_ROLE_SET, FLOATING_ROLE_SET, TRAIN_ROLE_SET


COLORS = [
    "#e6194b",
    "#3cb44b",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#46f0f0",
    "#f032e6",
    "#bcf60c",
    "#fabebe",
    "#008080",
    "#e6beff",
    "#9a6324",
    "#fffac8",
    "#800000",
    "#aaffc3",
    "#808000",
    "#ffd8b1",
    "#000075",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    with args.json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    frame = data[0] if isinstance(data, list) else data
    if not isinstance(frame, dict):
        raise ValueError("preview JSON must contain a frame object or list of frame objects")

    width, height = get_canvas_size(frame)
    image = Image.new("RGB", (max(1, round(width)), max(1, round(height))), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for index, (role, node) in enumerate(flatten_semantic_nodes(frame).items()):
        bounds = node.get("bounds") or {}
        x = float(bounds.get("x") or 0)
        y = float(bounds.get("y") or 0)
        w = float(bounds.get("width") or 0)
        h = float(bounds.get("height") or 0)
        color = COLORS[index % len(COLORS)]
        if role in FLOATING_ROLE_SET:
            _draw_dashed_rectangle(draw, [x, y, x + w, y + h], color, width=3)
            label_offset = 3
        elif role in CHILD_ROLE_SET:
            draw.rectangle([x, y, x + w, y + h], outline=color, width=1)
            label_offset = 2
        elif role in TRAIN_ROLE_SET:
            draw.rectangle([x, y, x + w, y + h], outline=color, width=4)
            label_offset = 5
        else:
            draw.rectangle([x, y, x + w, y + h], outline=color, width=2)
            label_offset = 3
        label = role
        label_box = draw.textbbox((x + label_offset, y + label_offset), label, font=font)
        draw.rectangle(label_box, fill="white", outline=color)
        draw.text((x + label_offset, y + label_offset), label, fill=color, font=font)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.out)
    print(f"Wrote: {args.out}")


def _draw_dashed_rectangle(
    draw: ImageDraw.ImageDraw,
    box: list[float],
    color: str,
    width: int = 2,
    dash: int = 14,
    gap: int = 8,
) -> None:
    x1, y1, x2, y2 = box
    _draw_dashed_line(draw, x1, y1, x2, y1, color, width, dash, gap)
    _draw_dashed_line(draw, x2, y1, x2, y2, color, width, dash, gap)
    _draw_dashed_line(draw, x2, y2, x1, y2, color, width, dash, gap)
    _draw_dashed_line(draw, x1, y2, x1, y1, color, width, dash, gap)


def _draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    color: str,
    width: int,
    dash: int,
    gap: int,
) -> None:
    length = max(1.0, ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
    dx = (x2 - x1) / length
    dy = (y2 - y1) / length
    pos = 0.0
    while pos < length:
        end = min(length, pos + dash)
        draw.line(
            [
                (x1 + dx * pos, y1 + dy * pos),
                (x1 + dx * end, y1 + dy * end),
            ],
            fill=color,
            width=width,
        )
        pos += dash + gap


if __name__ == "__main__":
    main()
