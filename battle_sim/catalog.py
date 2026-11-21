"""
LLM Battle-Cars — Parts Catalog, Raw Physical Sensors & DIY Strategy Linter (v3.1).

Defines:
- Fair-play tournament limits (max functional parts, build points, active weapons, simulation quota)
- Unrestricted vehicle Length x Width x Height (governed naturally by 3D volume x density mass & inertia!)
- 13 physical functional part types + 7 decoration primitives
- Complete raw physical sensors & static AST compiler/linter for 100% model-authored wheel & weapon control!
"""
from __future__ import annotations

import difflib
import math
import re
from typing import Any, Dict, List, Set, Tuple

from .expr import BASE_EVAL_ENV, VALID_CONST_NAMES, VALID_FUNC_NAMES, validate_and_compile_expr

# ============================================================================
# Core Fair-Play & Tournament Constants
# ============================================================================
MAX_FUNCTIONAL_PARTS = 14
MAX_BUILD_POINTS = 100
MAX_ACTIVE_WEAPONS = 4
MAX_ENERGY_CANNONS = 2
MAX_DECO_PARTS = 10
MAX_SIM_CREDITS = 120    # 1 single stage = 1 credit; 1 full BO3 = 3 credits

MAX_STRATEGY_VARS = 20
MAX_STRATEGY_LET = 28

# Minimum non-degenerate physical dimensions for core chassis [Length_X, Width_Y, Height_Z] in meters.
# NO upper Length/Width/Height cap! Larger dimensions naturally add mass & rotational inertia.
CHASSIS_MIN_DIMS = (0.06, 0.06, 0.05)

# Physics & Match Timing (Exact 360 Hz -> 12 substeps = native 30.000 FPS)
PHYSICS_DT = 1.0 / 360.0
SUBSTEPS = 12
FPS = 30
MATCH_DURATION_SEC = 15.0
MATCH_FRAMES = int(MATCH_DURATION_SEC * FPS)  # 450 frames
POST_KO_FRAMES = 39                           # +1.30s fall/impact tail

MOUNT_ANCHORS: Dict[str, Tuple[Tuple[float, float, float], float]] = {
    "center":      ((0.0,  0.0,  0.0),   0.0),
    "front":       ((1.0,  0.0,  0.0),   0.0),
    "back":        ((-1.0, 0.0,  0.0), 180.0),
    "left":        ((0.0,  1.0,  0.0),  90.0),
    "right":       ((0.0, -1.0,  0.0), -90.0),
    "top":         ((0.0,  0.0,  1.0),   0.0),
    "bottom":      ((0.0,  0.0, -1.0),   0.0),
    "front_left":  ((0.78,  1.0, -0.15),  0.0),
    "front_right": ((0.78, -1.0, -0.15),  0.0),
    "mid_left":    ((0.0,   1.0, -0.15),  0.0),
    "mid_right":   ((0.0,  -1.0, -0.15),  0.0),
    "back_left":   ((-0.78, 1.0, -0.15),  0.0),
    "back_right":  ((-0.78,-1.0, -0.15),  0.0),
    "top_front":   ((0.55,  0.0,  1.0),   0.0),
    "top_back":    ((-0.55, 0.0,  1.0),   0.0),
}

VALID_MATERIALS = ("glossy", "matte", "metallic", "neon", "wood", "glass")
DECO_SHAPES = ("box", "cylinder", "sphere", "cone", "fin", "ring", "eye")

ACTIVE_WEAPON_TYPES = {
    "spinner_horiz",
    "spinner_vert",
    "flipper",
    "hammer",
    "piston_punch",
    "energy_cannon",
    "thruster",
}

CANNON_MODES: Dict[str, Dict[str, float]] = {
    "rapid":    {"cooldown": 0.70, "speed": 17.0, "impulse": 19.5, "recoil": 3.2, "damage": 8.5},
    "standard": {"cooldown": 1.02, "speed": 15.0, "impulse": 30.0, "recoil": 5.0, "damage": 13.0},
    "heavy":    {"cooldown": 1.45, "speed": 13.0, "impulse": 44.0, "recoil": 7.5, "damage": 18.5},
}

# Raw Physical Sensors at 120Hz (ZERO pre-baked tactical maneuvers!)
VALID_NUMERIC_SENSORS = frozenset({
    # Self pose & motion
    "self_x",
    "self_y",
    "self_z",
    "self_yaw_deg",
    "self_speed",
    "self_forward_speed",
    "self_yaw_rate_deg",
    "tilt_deg",
    "stalled_sec",
    # Opponent pose, velocity & relative geometry
    "enemy_x",
    "enemy_y",
    "enemy_z",
    "enemy_yaw_deg",
    "enemy_vx",
    "enemy_vy",
    "enemy_speed",
    "enemy_closing_speed",
    "enemy_dist",
    "enemy_angle_deg",
    "abs_enemy_angle_deg",
    "enemy_edge_dist",
    # Arena cliff & center geometry
    "edge_dist",
    "edge_angle_deg",
    "abs_edge_angle_deg",
    "center_dist",
    "center_angle_deg",
    "abs_center_angle_deg",
    # Nearest hinged floor catapult to self & to enemy (9.9 if none on stage)
    "spring_x",
    "spring_y",
    "spring_dist",
    "spring_angle_deg",
    "abs_spring_angle_deg",
    "spring_fling_yaw_deg",
    "enemy_spring_x",
    "enemy_spring_y",
    "enemy_spring_dist",
    "enemy_spring_fling_yaw_deg",
    # Nearest quantum portal to self & to enemy (Stage 3 portal_arena; 9.9 if none on stage)
    "portal_x",
    "portal_y",
    "portal_dist",
    "portal_angle_deg",
    "abs_portal_angle_deg",
    "portal_out_x",
    "portal_out_y",
    "portal_out_yaw_deg",
    "enemy_portal_x",
    "enemy_portal_y",
    "enemy_portal_dist",
    "enemy_portal_out_yaw_deg",
    # Incoming projectile, fuel & clock
    "incoming_bolt_dist",
    "incoming_bolt_angle_deg",
    "abs_incoming_bolt_angle_deg",
    "incoming_bolt_vx",
    "incoming_bolt_vy",
    "thruster_fuel",
    "time_sec",
    "dt",
    "stage_id",
})

VALID_BOOL_SENSORS = frozenset({
    "is_inverted",
    "enemy_is_tilted",
    "has_line_of_sight",
    "cannon_ready",
    "spring_ready",
    "enemy_spring_ready",
    "portal_ready",
    "enemy_portal_ready",
})

VALID_ALL_SENSORS = VALID_NUMERIC_SENSORS | VALID_BOOL_SENSORS

# ONLY direct physical hardware actuators — NO pre-baked 'drive' autopilot modes!
VALID_ACTION_KEYS = frozenset({
    "left",
    "right",
    "center",
    "spinner",
    "flipper",
    "hammer",
    "piston",
    "cannon",
    "fire_cannon",
    "thruster",
})


# ============================================================================
# 13 Functional Combat, Ranged, Defense & Mobility Parts Catalog
# ============================================================================
PARTS_CATALOG: Dict[str, Dict[str, Any]] = {
    "frame": {
        "cost": 4,
        "category": "structure",
        "name_zh": "扩展结构梁",
        "summary": "Structural beam/outrigger of arbitrary size that extends the chassis in 3D and can host child parts.",
        "defaults": {"size": [0.36, 0.10, 0.08], "material": "metallic"},
    },
    "wheel": {
        "cost": 6,
        "category": "mobility",
        "name_zh": "橡胶驱动轮",
        "summary": "Motorized high-grip drive wheel (or free caster). Supports speed (high RPM), balanced, or torque (high push) gearing.",
        "defaults": {"radius": 0.18, "width": 0.055, "channel": "auto", "gear": "balanced"},
    },
    "monster_wheel": {
        "cost": 10,
        "category": "mobility",
        "name_zh": "越野大脚轮",
        "summary": "Large wide tire for slope climbing & invertible driving when radius > half chassis height.",
        "defaults": {"radius": 0.31, "width": 0.105, "channel": "auto", "gear": "torque"},
    },
    "wedge": {
        "cost": 7,
        "category": "control",
        "name_zh": "精钢斜铲",
        "summary": "Low-friction angled steel ramp that scoops under enemy tires to break traction and deflects 30% of cannon bolt impulse.",
        "defaults": {"width": 0.44, "length": 0.28, "slope_deg": 26.0},
    },
    "armor_shield": {
        "cost": 6,
        "category": "defense",
        "name_zh": "实体防爆护甲板",
        "summary": "High-density physical armor plate to guard wheels against melee spinners and reduce cannon bolt knockback & damage by 30%.",
        "defaults": {"width": 0.42, "height": 0.22, "thickness": 0.045, "tilt_deg": 10.0},
    },
    "gyro_stabilizer": {
        "cost": 7,
        "category": "defense",
        "name_zh": "底盘重型配重块",
        "summary": "Ultra-dense underbelly ballast disc that lowers Center of Mass to resist flipping and knockback (adds mass, lowering top speed).",
        "defaults": {"radius": 0.16, "thickness": 0.05},
    },
    "spinner_horiz": {
        "cost": 18,
        "category": "weapon",
        "name_zh": "水平旋转刀梁",
        "summary": "High-RPM horizontal spinning bar with dual heavy impact teeth for strong lateral knockback.",
        "defaults": {"radius": 0.38, "height_offset": 0.04, "spin_dir": 1},
    },
    "spinner_vert": {
        "cost": 20,
        "category": "weapon",
        "name_zh": "垂直上勾滚筒",
        "summary": "High-RPM vertical uppercut drum that launches opponents upward on close-range contact.",
        "defaults": {"radius": 0.24, "width": 0.09, "spin_dir": 1},
    },
    "flipper": {
        "cost": 16,
        "category": "weapon",
        "name_zh": "气动弹射铲",
        "summary": "Hinged pneumatic launcher (72-deg upward snap, 1.3s cooldown) that flips enemies or self-rights.",
        "defaults": {"length": 0.40, "width": 0.28},
    },
    "hammer": {
        "cost": 15,
        "category": "weapon",
        "name_zh": "过顶动力战锤",
        "summary": "Overhead pneumatic battle-hammer (115-deg downward slam, 1.15s cooldown) that strikes enemy tops or self-rights.",
        "defaults": {"arm_length": 0.46, "head_size": 0.12},
    },
    "piston_punch": {
        "cost": 15,
        "category": "weapon",
        "name_zh": "直线伸缩冲拳",
        "summary": "Telescoping linear ram (1.1s cooldown) that punches up to 0.56m outward at high melee impulse.",
        "defaults": {"stroke": 0.42, "head_width": 0.18},
    },
    "energy_cannon": {
        "cost": 18,
        "category": "weapon",
        "name_zh": "远程能量炮",
        "summary": "Ranged twin-rail plasma cannon (rapid/standard/heavy modes, max 2 per car). Fires 3D energy bolts with physical knockback & recoil; 30% reduced by armor_shield/wedge.",
        "defaults": {"mode": "standard", "pitch_deg": 0.0},
    },
    "thruster": {
        "cost": 14,
        "category": "weapon",
        "name_zh": "推进喷射器",
        "summary": "95N booster nozzle with regenerating 1.8s fuel tank for burst acceleration, slope jumps, or cliff saves.",
        "defaults": {"aim": "push_forward", "power": 1.0},
    },
}

HEX_COLOR_RE = re.compile(r"^#?[0-9a-fA-F]{6}$")
IDENT_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def parse_hex_color(hex_str: Any, fallback: Tuple[float, float, float] = (0.20, 0.52, 0.92)) -> Tuple[float, float, float]:
    """Convert '#RRGGBB' string to [0..1] float RGB tuple."""
    if not isinstance(hex_str, str) or not HEX_COLOR_RE.match(hex_str.strip()):
        return fallback
    s = hex_str.strip().lstrip("#")
    r = int(s[0:2], 16) / 255.0
    g = int(s[2:4], 16) / 255.0
    b = int(s[4:6], 16) / 255.0
    return (round(r, 4), round(g, 4), round(b, 4))


def clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(val)))


def validate_car_dict(spec: Dict[str, Any]) -> Tuple[bool, List[str], List[str], Dict[str, Any]]:
    """
    Validate raw JSON dict against schema, part caps, point budget, and DIY strategy expression compiler.
    Returns: (ok, errors, warnings, summary_info)
    """
    errors: List[str] = []
    warnings: List[str] = []

    if not isinstance(spec, dict):
        return False, ["Root JSON value must be an object {...}."], [], {}

    meta = spec.get("meta", {})
    if not isinstance(meta, dict):
        errors.append("'meta' must be an object containing model_name, car_name, etc.")
        meta = {}

    model_name = str(meta.get("model_name", "")).strip()
    car_name = str(meta.get("car_name", "")).strip()
    if not model_name:
        errors.append("'meta.model_name' is required.")
    if not car_name:
        errors.append("'meta.car_name' is required.")

    for cname in ("primary_color", "secondary_color", "accent_color"):
        if cname in meta and not HEX_COLOR_RE.match(str(meta[cname]).strip()):
            errors.append(f"'meta.{cname}' must be a 6-digit hex color like '#3B82F6', got {meta[cname]!r}.")

    chassis = spec.get("chassis", {})
    if not isinstance(chassis, dict):
        errors.append("'chassis' must be an object with 'size': [length_x, width_y, height_z].")
        chassis = {}
    csize = chassis.get("size", [0.52, 0.38, 0.18])
    if not (
        isinstance(csize, (list, tuple))
        and len(csize) == 3
        and all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x)) for x in csize)
    ):
        errors.append("'chassis.size' must be a list of 3 finite numbers [length_x, width_y, height_z] in meters.")
    else:
        for idx, (val, lo, axis_lbl) in enumerate(zip(csize, CHASSIS_MIN_DIMS, ("length_x", "width_y", "height_z"))):
            if float(val) < lo:
                errors.append(f"'chassis.size[{idx}]' ({axis_lbl}={val}) must be >= {lo}m (no upper size limit!).")

    parts = spec.get("parts", [])
    if not isinstance(parts, list):
        errors.append("'parts' must be a list of functional part objects.")
        parts = []

    if len(parts) == 0:
        errors.append("Car must have at least 1 functional part in 'parts'.")
    if len(parts) > MAX_FUNCTIONAL_PARTS:
        errors.append(f"Too many functional parts: {len(parts)} > MAX_FUNCTIONAL_PARTS ({MAX_FUNCTIONAL_PARTS}).")

    seen_ids = {"chassis"}
    frame_ids = {"chassis"}
    total_points = 0
    active_weapons = 0
    wheel_count = 0
    energy_cannon_count = 0

    for i, p in enumerate(parts):
        if not isinstance(p, dict):
            errors.append(f"parts[{i}] must be an object.")
            continue
        pid = str(p.get("id", f"part_{i}")).strip()
        if not pid or not re.match(r"^[a-zA-Z0-9_\-]+$", pid):
            errors.append(f"parts[{i}].id ({pid!r}) must be alphanumeric/underscore.")
        if pid in seen_ids:
            errors.append(f"Duplicate part id {pid!r} at parts[{i}]. Every part id must be unique.")
        seen_ids.add(pid)

        ptype = str(p.get("type", "")).strip()
        if ptype == "energy_shield":
            errors.append(
                f"parts[{i}] ({pid}): 'energy_shield' has been removed! "
                f"Use physical 'armor_shield'/'wedge' (30% bolt reduction) or write lateral wheel dodge logic in 'strategy'."
            )
            continue
        if ptype not in PARTS_CATALOG:
            errors.append(f"parts[{i}] ({pid}): unknown part type {ptype!r}. Valid types: {sorted(PARTS_CATALOG.keys())}")
            continue

        total_points += int(PARTS_CATALOG[ptype]["cost"])
        if ptype in ACTIVE_WEAPON_TYPES:
            active_weapons += 1
        if ptype in ("wheel", "monster_wheel"):
            wheel_count += 1
        if ptype == "energy_cannon":
            energy_cannon_count += 1
            params = p.get("params") or {}
            if isinstance(params, dict) and "mode" in params:
                cmode = str(params["mode"]).strip().lower()
                if cmode not in CANNON_MODES:
                    errors.append(
                        f"parts[{i}] ({pid}): invalid energy_cannon mode {params['mode']!r}. "
                        f"Choose from {list(CANNON_MODES.keys())}."
                    )

        parent = str(p.get("parent", "chassis")).strip()
        if parent not in frame_ids:
            errors.append(
                f"parts[{i}] ({pid}): parent {parent!r} must be 'chassis' or the 'id' of an EARLIER 'frame' part."
            )
        if ptype == "frame":
            frame_ids.add(pid)

        mount = p.get("mount", None)
        if mount is not None and str(mount) not in MOUNT_ANCHORS:
            errors.append(
                f"parts[{i}] ({pid}): invalid mount {mount!r}. Valid mounts: {sorted(MOUNT_ANCHORS.keys())}"
            )
        for vec_key in ("pos", "offset", "euler"):
            if vec_key in p and p[vec_key] is not None:
                v = p[vec_key]
                if not (isinstance(v, (list, tuple)) and len(v) == 3 and all(isinstance(x, (int, float)) and math.isfinite(x) for x in v)):
                    errors.append(f"parts[{i}] ({pid}): '{vec_key}' must be a list of 3 finite numbers.")

        if "color" in p and p["color"] is not None and not HEX_COLOR_RE.match(str(p["color"]).strip()):
            errors.append(f"parts[{i}] ({pid}): 'color' must be a 6-digit hex color like '#00E5FF'.")

    if energy_cannon_count > MAX_ENERGY_CANNONS:
        errors.append(
            f"Too many 'energy_cannon' parts ({energy_cannon_count} > {MAX_ENERGY_CANNONS}). "
            f"At most {MAX_ENERGY_CANNONS} energy cannons may be installed per vehicle."
        )
    if total_points > MAX_BUILD_POINTS:
        errors.append(f"Build point budget exceeded: {total_points} pts > MAX_BUILD_POINTS ({MAX_BUILD_POINTS} pts).")
    if active_weapons > MAX_ACTIVE_WEAPONS:
        errors.append(
            f"Too many active powered weapons ({active_weapons} > {MAX_ACTIVE_WEAPONS}). "
            f"Active weapon types are: {sorted(ACTIVE_WEAPON_TYPES)}."
        )
    if wheel_count == 0:
        warnings.append("Car has 0 wheels! Unless it moves purely via thrusters, it won't be able to drive.")
    elif wheel_count == 1:
        warnings.append("Car only has 1 wheel — differential steering requires at least left & right wheels.")

    decos = spec.get("decorations", [])
    if not isinstance(decos, list):
        errors.append("'decorations' must be a list of styling objects (or omitted).")
        decos = []
    if len(decos) > MAX_DECO_PARTS:
        errors.append(f"Too many decorations: {len(decos)} > MAX_DECO_PARTS ({MAX_DECO_PARTS}).")

    for j, d in enumerate(decos):
        if not isinstance(d, dict):
            errors.append(f"decorations[{j}] must be an object.")
            continue
        shape = str(d.get("shape", "box")).strip()
        if shape not in DECO_SHAPES:
            errors.append(f"decorations[{j}]: unknown shape {shape!r}. Valid shapes: {list(DECO_SHAPES)}")
        dparent = str(d.get("parent", "chassis")).strip()
        if dparent not in seen_ids:
            errors.append(f"decorations[{j}]: parent {dparent!r} does not match 'chassis' or any defined part id.")
        mat = str(d.get("material", "glossy")).strip()
        if mat not in VALID_MATERIALS:
            errors.append(f"decorations[{j}]: unknown material {mat!r}. Valid materials: {list(VALID_MATERIALS)}")
        if "color" in d and d["color"] is not None and not HEX_COLOR_RE.match(str(d["color"]).strip()):
            errors.append(f"decorations[{j}]: 'color' must be a 6-digit hex color like '#FFD700'.")

    strategy = spec.get("strategy")
    if not isinstance(strategy, dict):
        errors.append("'strategy' is required and must be an object defining your custom wheel & weapon control logic.")
    else:
        installed_part_types = {
            str(p.get("type", "")).strip()
            for p in parts
            if isinstance(p, dict) and str(p.get("type", "")).strip() in PARTS_CATALOG
        }
        _lint_strategy_dict(strategy, installed_part_types, wheel_count, errors, warnings)

    summary = {
        "model_name": model_name or "Unknown",
        "car_name": car_name or "Unnamed",
        "functional_parts": len(parts),
        "max_functional_parts": MAX_FUNCTIONAL_PARTS,
        "build_points": total_points,
        "max_build_points": MAX_BUILD_POINTS,
        "active_weapons": active_weapons,
        "energy_cannons": energy_cannon_count,
        "wheel_count": wheel_count,
        "decorations": len(decos),
        "max_decorations": MAX_DECO_PARTS,
    }
    return (len(errors) == 0), errors, warnings, summary


_VALID_STAGE_OVERRIDE_KEYS = frozenset({
    "1", "2", "3",
    "s1", "s2", "s3",
    "stage1", "stage2", "stage3",
    "stage_1", "stage_2", "stage_3",
    "spring_arena", "camelback_ridge", "portal_arena", "turbo_bowl",
    "vortex_crown", "twin_drawbridge", "forge_colosseum",
    "cyclone_ring", "pendulum_bridge", "crusher_pit",
})

_VALID_STRATEGY_TOP_KEYS = frozenset({
    "vars",
    "let",
    "rules",
    "default_action",
    "stage_overrides",
})


def _make_sample_sensor_env() -> Dict[str, Any]:
    env = dict(BASE_EVAL_ENV)
    for k in VALID_NUMERIC_SENSORS:
        env[k] = 1.0
    env["stage_id"] = 1
    env["dt"] = 1.0 / 120.0
    env["enemy_angle_deg"] = 18.0
    env["abs_enemy_angle_deg"] = 18.0
    env["enemy_dist"] = 2.4
    env["edge_dist"] = 2.1
    env["incoming_bolt_dist"] = 9.9
    for kb in VALID_BOOL_SENSORS:
        env[kb] = False
    env["has_line_of_sight"] = True
    env["cannon_ready"] = True
    env["spring_ready"] = True
    env["enemy_spring_ready"] = True
    return env


def _lint_when_clause(when: Any, ctx: str, allowed_vars: Set[str], sample_env: Dict[str, Any], errors: List[str]) -> None:
    if isinstance(when, (str, bool)):
        try:
            fn, _ = validate_and_compile_expr(when, allowed_vars, f"{ctx}.when")
            fn(sample_env)
        except Exception as exc:
            errors.append(str(exc))
        return

    if not isinstance(when, dict):
        errors.append(f"{ctx}.when must be a boolean formula string (e.g. \"edge_dist < 0.9 and abs_edge_angle_deg < 75\") or condition object, got {type(when).__name__}.")
        return

    for ckey, cval in when.items():
        if ckey == "stage_in":
            if not isinstance(cval, (list, tuple)) or not all(x in (1, 2, 3) for x in cval):
                errors.append(f"{ctx}.when['stage_in'] must be a list of stage numbers from [1, 2, 3], got {cval!r}.")
            continue
        if ckey == "expr":
            try:
                fn, _ = validate_and_compile_expr(cval, allowed_vars, f"{ctx}.when['expr']")
                fn(sample_env)
            except Exception as exc:
                errors.append(str(exc))
            continue
        if ckey.endswith("_between"):
            sname = ckey[: -len("_between")]
            if sname not in allowed_vars:
                hint = difflib.get_close_matches(sname, allowed_vars, n=2)
                errors.append(
                    f"{ctx}.when[{ckey!r}]: unknown numeric sensor/var {sname!r}."
                    + (f" Did you mean {hint}?" if hint else "")
                )
            elif not (
                isinstance(cval, (list, tuple))
                and len(cval) == 2
                and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in cval)
                and float(cval[0]) <= float(cval[1])
            ):
                errors.append(f"{ctx}.when[{ckey!r}] must be [min_val, max_val] with min_val <= max_val, got {cval!r}.")
            continue

        matched_num_suf = False
        for suf in ("_lte", "_lt", "_gte", "_gt"):
            if ckey.endswith(suf):
                matched_num_suf = True
                sname = ckey[: -len(suf)]
                if sname in VALID_BOOL_SENSORS:
                    errors.append(
                        f"{ctx}.when[{ckey!r}]: {sname!r} is a boolean sensor — write '{sname}': true/false or a formula string."
                    )
                elif sname not in allowed_vars:
                    hint = difflib.get_close_matches(sname, allowed_vars, n=2)
                    errors.append(
                        f"{ctx}.when[{ckey!r}]: unknown sensor/var {sname!r}."
                        + (f" Did you mean {hint}?" if hint else "")
                    )
                elif not (isinstance(cval, (int, float)) and not isinstance(cval, bool) and math.isfinite(float(cval))):
                    errors.append(f"{ctx}.when[{ckey!r}] must be a finite number, got {cval!r}.")
                break
        if matched_num_suf:
            continue

        if ckey not in allowed_vars:
            hint = difflib.get_close_matches(ckey, allowed_vars, n=2)
            errors.append(
                f"{ctx}.when[{ckey!r}]: unknown condition {ckey!r} (use a formula string or _lt/_gt/_lte/_gte/_between)."
                + (f" Closest names: {hint}." if hint else "")
            )
        elif ckey in VALID_BOOL_SENSORS and not isinstance(cval, bool):
            errors.append(f"{ctx}.when[{ckey!r}]: boolean sensor expects true or false, got {cval!r}.")


def _lint_action_dict(
    act: Any,
    ctx: str,
    allowed_vars: Set[str],
    sample_env: Dict[str, Any],
    installed_types: Set[str],
    used_triggers: Set[str],
    wheel_keys_seen: Set[str],
    errors: List[str],
    warnings: List[str],
    is_explicit_rule: bool,
) -> None:
    if not isinstance(act, dict):
        errors.append(f"{ctx} must be an action object controlling 'left', 'right', and weapons, got {type(act).__name__}.")
        return
    if "drive" in act:
        errors.append(
            f"{ctx}['drive']: pre-built 'drive' autopilot modes ({act['drive']!r}) do NOT exist! "
            f"Design and write your own differential wheel control directly using 'left' and 'right' "
            f"(-1.0..+1.0 numbers or math formulas over sensors, e.g. \"clamp(0.85 - enemy_angle_deg / 60, -1, 1)\")."
        )

    for akey, aval in act.items():
        if akey == "drive":
            continue
        if akey not in VALID_ACTION_KEYS:
            hint = difflib.get_close_matches(akey, VALID_ACTION_KEYS, n=2)
            errors.append(
                f"{ctx}[{akey!r}]: unknown actuator key {akey!r}."
                + (f" Did you mean {hint}?" if hint else f" Valid actuator keys: {sorted(VALID_ACTION_KEYS)}")
            )
            continue
        try:
            fn, _ = validate_and_compile_expr(aval, allowed_vars, f"{ctx}[{akey!r}]")
            fn(sample_env)
        except Exception as exc:
            errors.append(str(exc))

        if akey in ("left", "right", "center"):
            wheel_keys_seen.add(akey)

    trigger_map = (
        ("flipper", "flipper"),
        ("hammer", "hammer"),
        ("piston", "piston_punch"),
        ("thruster", "thruster"),
    )
    for act_k, req_part in trigger_map:
        if act_k in act and act[act_k] is not False and str(act[act_k]).strip().lower() not in ("false", "0", "0.0"):
            used_triggers.add(req_part)
            if is_explicit_rule and req_part not in installed_types:
                warnings.append(f"{ctx} controls '{act_k}', but this car has NO '{req_part}' part installed.")

    for c_k in ("cannon", "fire_cannon"):
        if c_k in act and act[c_k] is not False and str(act[c_k]).strip().lower() not in ("false", "0", "0.0"):
            used_triggers.add("energy_cannon")
            if is_explicit_rule and "energy_cannon" not in installed_types:
                warnings.append(f"{ctx} controls '{c_k}', but this car has NO 'energy_cannon' part installed.")


def _lint_strategy_block(
    block: Dict[str, Any],
    ctx_prefix: str,
    base_vars: Dict[str, Any],
    base_let: Dict[str, Any],
    installed_types: Set[str],
    used_triggers: Set[str],
    wheel_keys_seen: Set[str],
    errors: List[str],
    warnings: List[str],
    require_default: bool,
) -> None:
    for k in block.keys():
        if k not in _VALID_STRATEGY_TOP_KEYS:
            errors.append(
                f"{ctx_prefix}[{k!r}]: unrecognized strategy key {k!r}. Valid keys: {sorted(_VALID_STRATEGY_TOP_KEYS)}."
            )

    merged_vars: Dict[str, Any] = dict(base_vars)
    if "vars" in block and block["vars"] is not None:
        if not isinstance(block["vars"], dict):
            errors.append(f"{ctx_prefix}.vars must be an object of initial state variables (numbers/booleans).")
        elif len(block["vars"]) > MAX_STRATEGY_VARS:
            errors.append(f"{ctx_prefix}.vars exceeds max {MAX_STRATEGY_VARS} state variables.")
        else:
            for vk, vv in block["vars"].items():
                if not IDENT_RE.match(str(vk)) or vk in VALID_ALL_SENSORS or vk in VALID_FUNC_NAMES or vk in VALID_CONST_NAMES:
                    errors.append(
                        f"{ctx_prefix}.vars[{vk!r}]: invalid variable name (must be identifier and cannot shadow built-in sensor/function)."
                    )
                elif not isinstance(vv, (int, float, bool)) or (isinstance(vv, (int, float)) and not math.isfinite(float(vv))):
                    errors.append(f"{ctx_prefix}.vars[{vk!r}] initial value must be a finite number or boolean, got {vv!r}.")
                else:
                    merged_vars[vk] = vv

    sample_env = _make_sample_sensor_env()
    sample_env.update(merged_vars)
    allowed_vars: Set[str] = set(VALID_ALL_SENSORS) | set(merged_vars.keys())
    declared_state_vars: Set[str] = set(merged_vars.keys())

    merged_let: Dict[str, Any] = dict(base_let)
    if "let" in block and block["let"] is not None:
        if not isinstance(block["let"], dict):
            errors.append(f"{ctx_prefix}.let must be an ordered object mapping helper variable names to formulas.")
        elif len(block["let"]) > MAX_STRATEGY_LET:
            errors.append(f"{ctx_prefix}.let exceeds max {MAX_STRATEGY_LET} bindings.")
        else:
            merged_let.update(block["let"])

    for lk, lexpr in merged_let.items():
        if not IDENT_RE.match(str(lk)) or lk in VALID_ALL_SENSORS or lk in VALID_FUNC_NAMES or lk in VALID_CONST_NAMES:
            errors.append(
                f"{ctx_prefix}.let[{lk!r}]: invalid helper name (must be identifier and cannot shadow built-in sensor/function)."
            )
            continue
        try:
            fn, _ = validate_and_compile_expr(lexpr, allowed_vars, f"{ctx_prefix}.let[{lk!r}]")
            sample_env[lk] = fn(sample_env)
            allowed_vars.add(lk)
        except Exception as exc:
            errors.append(str(exc))
            sample_env[lk] = 0.0
            allowed_vars.add(lk)

    if "rules" in block and block["rules"] is not None:
        rules = block["rules"]
        if not isinstance(rules, list):
            errors.append(f"{ctx_prefix}.rules must be a list of rule objects.")
        else:
            seen_rnames: Set[str] = set()
            for idx, r in enumerate(rules):
                rctx = f"{ctx_prefix}.rules[{idx}]"
                if not isinstance(r, dict):
                    errors.append(f"{rctx} must be an object with 'name', 'when', and 'action'.")
                    continue
                rname = str(r.get("name", f"rule_{idx}")).strip()
                rctx_named = f"{rctx} ({rname!r})"
                if rname in seen_rnames:
                    warnings.append(f"{rctx_named}: duplicate rule name {rname!r} — use unique names for clearer telemetry.")
                seen_rnames.add(rname)

                for rk in r.keys():
                    if rk not in ("name", "when", "set_vars", "action"):
                        errors.append(f"{rctx_named}: unknown rule key {rk!r} (allowed: 'name', 'when', 'set_vars', 'action').")

                when = r.get("when", {})
                _lint_when_clause(when, rctx_named, allowed_vars, sample_env, errors)
                is_unconditional = (
                    (isinstance(when, dict) and len(when) == 0)
                    or when is True
                    or (isinstance(when, str) and when.strip().lower() in ("true", "1"))
                )
                if is_unconditional and idx < len(rules) - 1:
                    warnings.append(
                        f"{rctx_named} has an unconditional 'when' (always true) and permanently shadows "
                        f"{len(rules) - 1 - idx} subsequent rule(s)!"
                    )

                local_allowed = set(allowed_vars)
                local_env = dict(sample_env)
                if "set_vars" in r and r["set_vars"] is not None:
                    sv = r["set_vars"]
                    if not isinstance(sv, dict):
                        errors.append(f"{rctx_named}.set_vars must be an object mapping variable names to formulas.")
                    else:
                        for svk, sv_expr in sv.items():
                            if not IDENT_RE.match(str(svk)) or svk in VALID_ALL_SENSORS or svk in VALID_FUNC_NAMES or svk in VALID_CONST_NAMES:
                                errors.append(f"{rctx_named}.set_vars[{svk!r}]: invalid variable name.")
                                continue
                            if svk not in declared_state_vars and svk not in merged_let:
                                warnings.append(
                                    f"{rctx_named}.set_vars[{svk!r}] assigns a variable not initialized in 'strategy.vars' "
                                    f"(consider declaring initial value in 'vars' if used across ticks)."
                                )
                            try:
                                fn_sv, _ = validate_and_compile_expr(sv_expr, local_allowed, f"{rctx_named}.set_vars[{svk!r}]")
                                local_env[svk] = fn_sv(local_env)
                                local_allowed.add(svk)
                            except Exception as exc:
                                errors.append(str(exc))
                                local_env[svk] = 0.0
                                local_allowed.add(svk)

                if "action" not in r:
                    errors.append(f"{rctx_named} is missing required 'action' object.")
                else:
                    _lint_action_dict(
                        r["action"], f"{rctx_named}.action", local_allowed, local_env,
                        installed_types, used_triggers, wheel_keys_seen, errors, warnings, is_explicit_rule=True,
                    )

    if "default_action" in block and block["default_action"] is not None:
        _lint_action_dict(
            block["default_action"], f"{ctx_prefix}.default_action", allowed_vars, sample_env,
            installed_types, used_triggers, wheel_keys_seen, errors, warnings, is_explicit_rule=False,
        )
    elif require_default:
        errors.append(
            f"{ctx_prefix}.default_action is required (must define fallback 'left' and 'right' wheel commands when no rule matches)."
        )


def _lint_strategy_dict(
    strategy: Dict[str, Any],
    installed_types: Set[str],
    wheel_count: int,
    errors: List[str],
    warnings: List[str],
) -> None:
    used_triggers: Set[str] = set()
    wheel_keys_seen: Set[str] = set()

    base_vars = strategy.get("vars") if isinstance(strategy.get("vars"), dict) else {}
    base_let = strategy.get("let") if isinstance(strategy.get("let"), dict) else {}

    _lint_strategy_block(
        strategy, "strategy", {}, {}, installed_types, used_triggers, wheel_keys_seen, errors, warnings, require_default=True
    )

    ov_map = strategy.get("stage_overrides")
    if ov_map is not None:
        if not isinstance(ov_map, dict):
            errors.append("'strategy.stage_overrides' must be an object mapping stage keys ('1'/'2'/'3' or stage names) to override objects.")
        else:
            for sk, sv in ov_map.items():
                sk_clean = str(sk).strip().lower()
                if sk_clean not in _VALID_STAGE_OVERRIDE_KEYS:
                    errors.append(
                        f"strategy.stage_overrides[{sk!r}]: unrecognized stage key {sk!r}. "
                        f"Use '1', '2', '3' or 'spring_arena', 'camelback_ridge', 'portal_arena'."
                    )
                    continue
                if not isinstance(sv, dict):
                    errors.append(f"strategy.stage_overrides[{sk!r}] must be an object.")
                    continue
                _lint_strategy_block(
                    sv, f"strategy.stage_overrides[{sk!r}]", base_vars, base_let,
                    installed_types, used_triggers, wheel_keys_seen, errors, warnings, require_default=False,
                )

    if wheel_count >= 2 and not ({"left", "right"} <= wheel_keys_seen):
        warnings.append(
            "Car has differential wheels, yet strategy never sets both 'left' and 'right' motor outputs — "
            "remember you must write both 'left' and 'right' wheel commands yourself!"
        )

    part_to_flag = (
        ("flipper", "'flipper'"),
        ("hammer", "'hammer'"),
        ("piston_punch", "'piston'"),
        ("thruster", "'thruster'"),
        ("energy_cannon", "'cannon'"),
    )
    for req_part, flag_desc in part_to_flag:
        if req_part in installed_types and req_part not in used_triggers:
            warnings.append(
                f"Car installs '{req_part}', but NO strategy rule or default_action ever activates {flag_desc}!"
            )
