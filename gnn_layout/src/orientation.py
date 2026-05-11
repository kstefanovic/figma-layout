"""Orientation helpers shared by training and inference."""

ORIENTATIONS = ["portrait", "balanced", "landscape", "super_wide"]
ORIENTATION_TO_IDX = {name: idx for idx, name in enumerate(ORIENTATIONS)}


def get_orientation(width: float, height: float) -> str:
    """Classify an aspect ratio into the layout buckets used by this pipeline."""
    w = float(width)
    h = float(height)
    if h <= 0:
        raise ValueError("height must be positive")
    aspect = w / h
    if aspect < 0.75:
        return "portrait"
    if aspect < 1.4:
        return "balanced"
    if aspect < 2.5:
        return "landscape"
    return "super_wide"


def orientation_to_onehot(orientation: str) -> list[float]:
    """Return a stable one-hot vector for an orientation name."""
    if orientation not in ORIENTATION_TO_IDX:
        raise ValueError(f"unknown orientation {orientation!r}; expected one of {ORIENTATIONS}")
    out = [0.0] * len(ORIENTATIONS)
    out[ORIENTATION_TO_IDX[orientation]] = 1.0
    return out
