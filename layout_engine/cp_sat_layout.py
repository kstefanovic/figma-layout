from ortools.sat.python import cp_model

from .default_priors import DEFAULT_PRIORS
from .fallback_layout import fallback_boxes


SCALE = 1000


def abs_diff(model, value, target, name):
    d = model.NewIntVar(0, SCALE * 3, f"abs_{name}")
    model.AddAbsEquality(d, value - target)
    return d


def make_vars(model, role: str, allow_overflow: bool = False):
    if allow_overflow:
        x = model.NewIntVar(-250, 1250, f"{role}_x")
        y = model.NewIntVar(-250, 1250, f"{role}_y")
        w = model.NewIntVar(20, 1500, f"{role}_w")
        h = model.NewIntVar(20, 1500, f"{role}_h")
    else:
        x = model.NewIntVar(0, SCALE, f"{role}_x")
        y = model.NewIntVar(0, SCALE, f"{role}_y")
        w = model.NewIntVar(20, SCALE, f"{role}_w")
        h = model.NewIntVar(20, SCALE, f"{role}_h")

    return {"x": x, "y": y, "w": w, "h": h}


def add_inside_constraint(model, box):
    model.Add(box["x"] >= 0)
    model.Add(box["y"] >= 0)
    model.Add(box["x"] + box["w"] <= SCALE)
    model.Add(box["y"] + box["h"] <= SCALE)


def add_above_constraint(model, upper, lower, margin=20):
    model.Add(upper["y"] + upper["h"] + margin <= lower["y"])


def clamp_int(v, lo, hi):
    return max(lo, min(hi, int(v)))


def add_prior_objective_terms(model, role, box, prior):
    tx = clamp_int(prior["x"] * SCALE, -250, 1250)
    ty = clamp_int(prior["y"] * SCALE, -250, 1250)
    tw = clamp_int(prior["w"] * SCALE, 20, 1500)
    th = clamp_int(prior["h"] * SCALE, 20, 1500)

    return [
        abs_diff(model, box["x"], tx, f"{role}_x"),
        abs_diff(model, box["y"], ty, f"{role}_y"),
        abs_diff(model, box["w"], tw, f"{role}_w"),
        abs_diff(model, box["h"], th, f"{role}_h"),
    ]


def solve_layout(
    orientation: str,
    target_w: int,
    target_h: int,
    available_roles: set,
):
    model = cp_model.CpModel()
    priors = DEFAULT_PRIORS[orientation]

    role_order = [
        "background",
        "hero_image",
        "brand_group",
        "headline_group",
        "legal_text",
        "age_badge",
        "decoration_group",
    ]

    roles = [r for r in role_order if r in available_roles]

    vars_by_role = {}

    for role in roles:
        allow_overflow = role in {"background", "hero_image", "decoration_group"}
        vars_by_role[role] = make_vars(model, role, allow_overflow=allow_overflow)

    # Hard constraints
    for role, box in vars_by_role.items():
        if role not in {"background", "hero_image", "decoration_group"}:
            add_inside_constraint(model, box)

    if "background" in vars_by_role:
        b = vars_by_role["background"]
        model.Add(b["x"] <= 0)
        model.Add(b["y"] <= 0)
        model.Add(b["w"] >= SCALE)
        model.Add(b["h"] >= SCALE)

    if "hero_image" in vars_by_role:
        hero = vars_by_role["hero_image"]
        model.Add(hero["w"] >= 250)
        model.Add(hero["h"] >= 250)

    if "legal_text" in vars_by_role:
        legal = vars_by_role["legal_text"]
        model.Add(legal["y"] >= 720)
        model.Add(legal["h"] >= 20)

    if "age_badge" in vars_by_role:
        age = vars_by_role["age_badge"]
        model.Add(age["w"] >= 25)
        model.Add(age["h"] >= 20)

    if "headline_group" in vars_by_role:
        headline = vars_by_role["headline_group"]
        model.Add(headline["w"] >= 220)
        model.Add(headline["h"] >= 70)

    if "brand_group" in vars_by_role:
        brand = vars_by_role["brand_group"]
        model.Add(brand["w"] >= 80)
        model.Add(brand["h"] >= 25)

    # Prevent important vertical collisions
    if "headline_group" in vars_by_role and "legal_text" in vars_by_role:
        add_above_constraint(
            model,
            vars_by_role["headline_group"],
            vars_by_role["legal_text"],
            margin=20,
        )

    # In portrait/landscape, brand is usually above headline.
    # In super_wide, brand and headline may be side-by-side, so do not force this.
    if orientation != "super_wide":
        if "brand_group" in vars_by_role and "headline_group" in vars_by_role:
            add_above_constraint(
                model,
                vars_by_role["brand_group"],
                vars_by_role["headline_group"],
                margin=10,
            )

    # Objective: stay close to priors
    objective_terms = []

    for role, box in vars_by_role.items():
        prior = priors.get(role)
        if not prior:
            continue
        objective_terms.extend(add_prior_objective_terms(model, role, box, prior))

    if objective_terms:
        model.Minimize(sum(objective_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 1.0
    solver.parameters.num_search_workers = 8

    status = solver.Solve(model)

    if status not in {cp_model.OPTIMAL, cp_model.FEASIBLE}:
        return fallback_boxes(orientation, target_w, target_h), "fallback"

    result = {}

    for role, box in vars_by_role.items():
        result[role] = {
            "x": round(solver.Value(box["x"]) / SCALE * target_w, 2),
            "y": round(solver.Value(box["y"]) / SCALE * target_h, 2),
            "width": round(solver.Value(box["w"]) / SCALE * target_w, 2),
            "height": round(solver.Value(box["h"]) / SCALE * target_h, 2),
        }

    return result, "solver"
