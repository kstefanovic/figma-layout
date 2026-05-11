from .default_priors import DEFAULT_PRIORS


def prior_to_box(prior: dict, target_w: int, target_h: int) -> dict:
    return {
        "x": round(prior["x"] * target_w, 2),
        "y": round(prior["y"] * target_h, 2),
        "width": round(prior["w"] * target_w, 2),
        "height": round(prior["h"] * target_h, 2),
    }


def fallback_boxes(orientation: str, target_w: int, target_h: int) -> dict:
    priors = DEFAULT_PRIORS[orientation]
    return {
        role: prior_to_box(prior, target_w, target_h)
        for role, prior in priors.items()
    }
