def get_orientation(width: float, height: float) -> str:
    aspect = width / max(height, 1)

    if aspect < 0.75:
        return "portrait"
    if aspect < 1.4:
        return "balanced"
    if aspect < 2.5:
        return "landscape"
    return "super_wide"
