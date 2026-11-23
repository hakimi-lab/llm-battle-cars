"""
Three Distinct Diorama-Grade 3D BO3 Arenas for LLM Battle-Cars (v4.0 — Cyber Catapults, Emerald Mudlands & Glacier Ice).

Design Philosophy (v4.0):
  - ZERO vertical snag pillars, ZERO narrow bridges, ZERO unused teleport portals, and ZERO mechanical repetition!
  - Rich, multi-layered 3D visual art direction (`contype="0" conaffinity="0"` architectural inlays, reactor cores,
    sandstone flagstones, glossy mud swirls, crystalline ice cracks, and under-stage supports) that renders identically
    in `preview-arena`, WebGL, Playblast, and Blender 4.2 Cycles OptiX!
  - Every BO3 stage has a 100% unique biome aesthetic AND passive/reactive surface physics that engage cars every match:
      1. Stage 1 ('spring_arena') — 霓虹四象·赛博弹射台 (Cyber-Neon Quad-Catapult Ring):
           6.90m circular gunmetal-neon colosseum (flat r=2.45m core + seamless 32-sector -7.5° cambered danger slope)
           with 4 Edge-Sensitive Solar-Amber Hinged Diagonal Catapult Plates!
      2. Stage 2 ('mud_arena')    — 翡翠湿地·泥沼陷车台 (Emerald Bayou & Viscous Mudlands):
           7.20m x 5.80m emerald basalt & sandstone causeway island (-7.0° North/South wet mossy cliff banks) featuring
           3 Viscous Peat Mud Bogs (1 large central basin across the direct rush lane + 2 NW/SE corner pools).
           All vehicles entering mud suffer a uniform ~50% speed reduction (never dead-stalling!), rewarding flanking
           along dry stone causeways and ramming/blasting enemies into the mud!
      3. Stage 3 ('ice_ridge')    — 极地冰川·溜冰飞坡台 (Glacier Overpass & Mirror-Ice Rink):
           7.40m x 5.80m alpine arena bisected by a high-grip +0.26m Central Glacier Ski-Jump Flyover Ramp (14.2° slopes)
           and bordered on all 4 sides by ultra-low-friction Mirror-Ice Skating Sheets (mu=0.10) leading into -8.0°
           East/West Icy Cliff Chutes — cars knocked or braking onto the ice slide helplessly toward the abyss!
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import mujoco
import numpy as np

FLOOR_Z = -1.85
DECK_HALF_THICK = 0.25

ARENA_ORDER = ("spring_arena", "mud_arena", "ice_ridge")

STAGE_ALIASES: Dict[str, str] = {
    "1": "spring_arena",
    "s1": "spring_arena",
    "stage1": "spring_arena",
    "stage_1": "spring_arena",
    "spring_arena": "spring_arena",
    "cyber_springs": "spring_arena",
    "vortex_crown": "spring_arena",
    "cyclone_ring": "spring_arena",

    "2": "mud_arena",
    "s2": "mud_arena",
    "stage2": "mud_arena",
    "stage_2": "mud_arena",
    "mud_arena": "mud_arena",
    "emerald_mudlands": "mud_arena",
    # Legacy Stage 2 aliases mapped cleanly
    "camelback_ridge": "mud_arena",
    "twin_drawbridge": "mud_arena",
    "pendulum_bridge": "mud_arena",

    "3": "ice_ridge",
    "s3": "ice_ridge",
    "stage3": "ice_ridge",
    "stage_3": "ice_ridge",
    "ice_ridge": "ice_ridge",
    "glacier_overpass": "ice_ridge",
    # Legacy Stage 3 aliases mapped cleanly
    "portal_arena": "ice_ridge",
    "turbo_bowl": "ice_ridge",
    "forge_colosseum": "ice_ridge",
    "crusher_pit": "ice_ridge",
}

# Hinged Diagonal Catapult Plate definitions (Stage 1 ONLY — zero repetition across stages!):
CATAPULT_HX = 0.48
CATAPULT_HY = 0.44
CATAPULT_HZ = 0.028
CATAPULT_MAX_DEG = 56.0
CATAPULT_MAX_RAD = math.radians(CATAPULT_MAX_DEG)

STAGE_SPRINGS: Dict[str, List[Dict[str, Any]]] = {
    "spring_arena": [
        {
            "tag": "ne", "x": 1.18, "y": 1.18, "z": 0.0,
            "slope_deg": 0.0, "fling_yaw": 45.0,
            "axis": (0.50, 0.50, 0.707), "max_rad": CATAPULT_MAX_RAD, "cooldown": 2.10, "arm_t": 0.30,
        },
        {
            "tag": "nw", "x": -1.18, "y": 1.18, "z": 0.0,
            "slope_deg": 0.0, "fling_yaw": 135.0,
            "axis": (-0.50, 0.50, 0.707), "max_rad": CATAPULT_MAX_RAD, "cooldown": 2.10, "arm_t": 0.30,
        },
        {
            "tag": "sw", "x": -1.18, "y": -1.18, "z": 0.0,
            "slope_deg": 0.0, "fling_yaw": -135.0,
            "axis": (-0.50, -0.50, 0.707), "max_rad": CATAPULT_MAX_RAD, "cooldown": 2.10, "arm_t": 0.30,
        },
        {
            "tag": "se", "x": 1.18, "y": -1.18, "z": 0.0,
            "slope_deg": 0.0, "fling_yaw": -45.0,
            "axis": (0.50, -0.50, 0.707), "max_rad": CATAPULT_MAX_RAD, "cooldown": 2.10, "arm_t": 0.30,
        },
    ],
    "mud_arena": [],
    "ice_ridge": [],
}

# Elliptical Viscous Mud Bog zones (Stage 2 `mud_arena` — uniform ~50% speed reduction for ALL vehicles!):
STAGE_MUD_ZONES: Dict[str, List[Dict[str, Any]]] = {
    "spring_arena": [],
    "mud_arena": [
        {"tag": "mud_core", "x": 0.0, "y": 0.0, "rx": 1.18, "ry": 0.86},
        {"tag": "mud_nw",   "x": -1.78, "y": 1.50, "rx": 0.86, "ry": 0.56},
        {"tag": "mud_se",   "x": 1.78, "y": -1.50, "rx": 0.86, "ry": 0.56},
    ],
    "ice_ridge": [],
}

# Rectangular Mirror-Ice Skating zones (Stage 3 `ice_ridge` — priority=2 ultra-low tire friction mu≈0.10!):
# Any vehicle with |x| >= 2.12 or |y| >= 2.08 on `ice_ridge` is on gleaming mirror ice!
ICE_INNER_HX = 2.12
ICE_INNER_HY = 2.08
ICE_OUTER_HX = 3.70
ICE_OUTER_HY = 2.90

# Kept empty for backward-compatibility with legacy helpers
STAGE_PORTALS: Dict[str, List[Dict[str, Any]]] = {
    "spring_arena": [],
    "mud_arena": [],
    "ice_ridge": [],
}

ARENA_META: Dict[str, Dict[str, Any]] = {
    "spring_arena": {
        "stage_id": 1,
        "key": "spring_arena",
        "name_zh": "霓虹四象·赛博弹射台",
        "name_en": "Cyber-Neon Quad-Catapult Ring",
        "spawn_a": (-1.95, 0.0),
        "spawn_b": (1.95, 0.0),
        "yaw_a": 0.0,
        "yaw_b": 180.0,
        "description": (
            "6.90m dark-titanium & cyan-neon circular colosseum: flat inner deck (r=0..2.45m) ringed by a seamless "
            "32-sector -7.5° downward danger slope (r=2.45..3.45m) and 4 Edge-Sensitive Hinged Diagonal Catapult "
            "Plates at (±1.18, ±1.18) that snap up 56° the instant any wheel or wedge touches their edge!"
        ),
    },
    "mud_arena": {
        "stage_id": 2,
        "key": "mud_arena",
        "name_zh": "翡翠湿地·泥沼陷车台",
        "name_en": "Emerald Bayou & Viscous Mudlands",
        "spawn_a": (-2.55, 0.0),
        "spawn_b": (2.55, 0.0),
        "yaw_a": 0.0,
        "yaw_b": 180.0,
        "description": (
            "7.20m x 5.80m emerald basalt & sandstone causeway plateau (-7.0° North/South mossy cliff slopes, |y|>2.20m) "
            "with 3 Viscous Peat Mud Bogs: a large Central Mud Basin at (0, 0, rx=1.18m, ry=0.86m) blocking blind "
            "straight-line rushes plus 2 corner mud pools at (-1.78, +1.50) & (+1.78, -1.50). All cars in mud slow by ~50%!"
        ),
    },
    "ice_ridge": {
        "stage_id": 3,
        "key": "ice_ridge",
        "name_zh": "极地冰川·溜冰飞坡台",
        "name_en": "Glacier Overpass & Mirror-Ice Rink",
        "spawn_a": (-1.80, 0.0),
        "spawn_b": (1.80, 0.0),
        "yaw_a": 0.0,
        "yaw_b": 180.0,
        "description": (
            "7.40m x 5.80m alpine glacier arena bisected by a high-grip +0.26m Central Glacier Ski-Jump Flyover Ramp "
            "(|x|<=1.45m, 14.2° slopes) and surrounded on all 4 sides (|x|>=2.12m or |y|>=2.08m) by ultra-low-friction "
            "Mirror-Ice Skating Sheets (mu=0.10) leading into -8.0° East/West Icy Cliff Chutes (|x|>=2.75m)!"
        ),
    },
}


def resolve_stage_key(stage: str) -> str:
    s = str(stage).strip().lower()
    if s not in STAGE_ALIASES:
        raise ValueError(
            f"Unknown stage {stage!r}. Choose 1/spring_arena, 2/mud_arena, or 3/ice_ridge."
        )
    return STAGE_ALIASES[s]


def _slope_quat_and_center(
    x0: float,
    y0: float,
    z0: float,
    horiz_run: float,
    yaw_deg: float,
    slope_up_deg: float,
    hz: float,
) -> Tuple[Tuple[float, float, float], str, float]:
    phi = math.radians(yaw_deg)
    alp = math.radians(slope_up_deg)
    c_p, s_p = math.cos(phi), math.sin(phi)
    c_a, s_a = math.cos(alp), math.sin(alp)

    ux = np.array([c_a * c_p, c_a * s_p, s_a], dtype=np.float64)
    uy = np.array([-s_p, c_p, 0.0], dtype=np.float64)
    uz = np.array([-s_a * c_p, -s_a * s_p, c_a], dtype=np.float64)
    R = np.column_stack([ux, uy, uz])

    q = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(q, R.reshape(-1))
    qstr = f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}"

    hx = 0.5 * horiz_run / max(1e-6, c_a)
    top_mid = np.array([
        x0 + 0.5 * horiz_run * c_p,
        y0 + 0.5 * horiz_run * s_p,
        z0 + 0.5 * horiz_run * math.tan(alp),
    ], dtype=np.float64)
    center = top_mid - hz * uz
    return (float(center[0]), float(center[1]), float(center[2])), qstr, hx


def _build_flush_spring_xml(sp: Dict[str, Any], env_ct: str, vis_ct: str) -> Tuple[str, str, str]:
    """Build a cyber-styled flush-mounted Hinged Diagonal Catapult Plate (`铰链斜向弹射板`)."""
    tag = sp["tag"]
    sx, sy, sz = float(sp["x"]), float(sp["y"]), float(sp["z"])
    alp = math.radians(float(sp["slope_deg"]))
    phi = math.radians(float(sp["fling_yaw"]))
    c_p, s_p = math.cos(phi), math.sin(phi)
    c_a, s_a = math.cos(alp), math.sin(alp)

    ux = np.array([c_a * c_p, c_a * s_p, s_a], dtype=np.float64)
    uy = np.array([-s_p, c_p, 0.0], dtype=np.float64)
    uz = np.array([-s_a * c_p, -s_a * s_p, c_a], dtype=np.float64)
    R_pad = np.column_stack([ux, uy, uz])
    q_pad = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(q_pad, R_pad.reshape(-1))
    qstr = f"{q_pad[0]:.6f} {q_pad[1]:.6f} {q_pad[2]:.6f} {q_pad[3]:.6f}"

    R_hinge = np.column_stack([ux, -uz, uy])
    q_hinge = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(q_hinge, R_hinge.reshape(-1))
    qh_str = f"{q_hinge[0]:.6f} {q_hinge[1]:.6f} {q_hinge[2]:.6f} {q_hinge[3]:.6f}"

    hx, hy, hz = CATAPULT_HX, CATAPULT_HY, CATAPULT_HZ
    recess_center = np.array([sx, sy, sz], dtype=np.float64) - (hz + 0.0025) * uz
    frame_center = np.array([sx, sy, sz], dtype=np.float64) + 0.0022 * uz
    hinge_world = np.array([sx, sy, sz], dtype=np.float64) + hx * ux + 0.014 * uz
    max_rad = float(sp["max_rad"])

    pad_rgba = "0.95 0.54 0.09 1"
    cap_rgba = "0.22 0.25 0.32 1"
    frame_rgba = "0.09 0.11 0.15 1"
    trim_rgba = "1.00 0.86 0.22 1"
    hinge_rgba = "0.80 0.88 0.96 1"

    aname = f"act_trap_spring_{tag}"
    jname = f"j_trap_spring_{tag}"
    body_xml = (
        f'    <geom name="env_spring_frame_{tag}" type="box" '
        f'pos="{frame_center[0]:.4f} {frame_center[1]:.4f} {frame_center[2]:.4f}" quat="{qstr}" '
        f'size="{hx + 0.052:.4f} {hy + 0.052:.4f} 0.0022" rgba="{frame_rgba}" {vis_ct}/>\n'
        f'    <geom name="env_spring_hinge_{tag}" type="cylinder" '
        f'pos="{hinge_world[0]:.4f} {hinge_world[1]:.4f} {hinge_world[2]:.4f}" quat="{qh_str}" '
        f'size="0.025 {hy + 0.040:.4f}" rgba="{hinge_rgba}" {vis_ct}/>\n'
        f'    <body name="trap_spring_{tag}" pos="{recess_center[0]:.4f} {recess_center[1]:.4f} {recess_center[2]:.4f}" quat="{qstr}">\n'
        f'      <joint name="{jname}" type="hinge" pos="{hx:.4f} 0 {hz:.4f}" axis="0 1 0" '
        f'range="0 {CATAPULT_MAX_DEG:.1f}" damping="8.0" armature="0.25"/>\n'
        f'      <geom name="trap_spring_{tag}_pad" type="box" pos="0 0 0" size="{hx:.4f} {hy:.4f} {hz:.4f}" '
        f'mass="14.0" rgba="{pad_rgba}" friction="0.55 0.005 0.002" {env_ct}/>\n'
        f'      <geom name="trap_spring_{tag}_inlay" type="box" pos="-0.02 0 {hz + 0.0065:.4f}" '
        f'size="{hx - 0.065:.4f} {hy - 0.065:.4f} 0.0024" mass="0" rgba="{cap_rgba}" {vis_ct}/>\n'
        f'      <geom name="trap_spring_{tag}_chev1_l" type="box" pos="0.12 0.11 {hz + 0.0105:.4f}" euler="0 0 -32" '
        f'size="0.15 0.030 0.0022" mass="0" rgba="{trim_rgba}" {vis_ct}/>\n'
        f'      <geom name="trap_spring_{tag}_chev1_r" type="box" pos="0.12 -0.11 {hz + 0.0105:.4f}" euler="0 0 32" '
        f'size="0.15 0.030 0.0022" mass="0" rgba="{trim_rgba}" {vis_ct}/>\n'
        f'      <geom name="trap_spring_{tag}_chev2_l" type="box" pos="-0.12 0.11 {hz + 0.0105:.4f}" euler="0 0 -32" '
        f'size="0.13 0.024 0.0022" mass="0" rgba="{pad_rgba}" {vis_ct}/>\n'
        f'      <geom name="trap_spring_{tag}_chev2_r" type="box" pos="-0.12 -0.11 {hz + 0.0105:.4f}" euler="0 0 32" '
        f'size="0.13 0.024 0.0022" mass="0" rgba="{pad_rgba}" {vis_ct}/>\n'
        f'      <geom name="trap_spring_{tag}_coil_l" type="cylinder" pos="{-0.55 * hx:.4f} {0.52 * hy:.4f} -0.16" '
        f'size="0.068 0.14" mass="0.5" rgba="{trim_rgba}" {vis_ct}/>\n'
        f'      <geom name="trap_spring_{tag}_coil_r" type="cylinder" pos="{-0.55 * hx:.4f} {-0.52 * hy:.4f} -0.16" '
        f'size="0.068 0.14" mass="0.5" rgba="{trim_rgba}" {vis_ct}/>\n'
        f'    </body>'
    )
    act_xml = (
        f'    <position name="{aname}" joint="{jname}" kp="1350" kv="26" '
        f'ctrllimited="true" ctrlrange="0 {max_rad:.5f}" forcelimited="true" forcerange="-140 620"/>'
    )
    return body_xml, act_xml, aname


def _build_ring_segments_xml(
    prefix_name: str,
    radius: float,
    z_pos: float,
    radial_half_w: float,
    half_h: float,
    n_seg: int,
    rgba: str,
    vis_ct: str,
    alt_rgba: Optional[str] = None,
) -> List[str]:
    """Build a crisp 3D polygonal/circular ring out of flush visual box segments."""
    out: List[str] = []
    half_tang = radius * math.tan(math.pi / n_seg) * 1.02
    for k in range(n_seg):
        yaw = k * (360.0 / n_seg)
        rad = math.radians(yaw)
        cx, cy = radius * math.cos(rad), radius * math.sin(rad)
        col = (alt_rgba if (alt_rgba and k % 2 == 1) else rgba)
        out.append(
            f'    <geom name="{prefix_name}_{k}" type="box" '
            f'pos="{cx:.4f} {cy:.4f} {z_pos:.4f}" euler="0 0 {yaw:.2f}" '
            f'size="{radial_half_w:.4f} {half_tang:.4f} {half_h:.4f}" rgba="{col}" {vis_ct}/>'
        )
    return out


def _build_spawn_pad_xml(
    name_prefix: str,
    x: float,
    y: float,
    z: float,
    yaw_deg: float,
    glow_rgba: str,
    vis_ct: str,
) -> str:
    """Build a sleek flush holographic team starting pad with directional chevron."""
    return (
        f'    <body name="{name_prefix}_pad" pos="{x:.4f} {y:.4f} {z:.4f}" euler="0 0 {yaw_deg:.2f}">\n'
        f'      <geom name="{name_prefix}_base" type="cylinder" pos="0 0 0.0018" size="0.44 0.0016" rgba="0.09 0.12 0.17 1" {vis_ct}/>\n'
        f'      <geom name="{name_prefix}_ring" type="cylinder" pos="0 0 0.0032" size="0.38 0.0014" rgba="{glow_rgba}" {vis_ct}/>\n'
        f'      <geom name="{name_prefix}_core" type="cylinder" pos="0 0 0.0046" size="0.31 0.0014" rgba="0.13 0.17 0.24 1" {vis_ct}/>\n'
        f'      <geom name="{name_prefix}_chev_l" type="box" pos="0.08 0.085 0.0062" euler="0 0 -34" size="0.12 0.024 0.0015" rgba="{glow_rgba}" {vis_ct}/>\n'
        f'      <geom name="{name_prefix}_chev_r" type="box" pos="0.08 -0.085 0.0062" euler="0 0 34" size="0.12 0.024 0.0015" rgba="{glow_rgba}" {vis_ct}/>\n'
        f'    </body>'
    )


def _build_mud_bog_visual_xml(mz: Dict[str, Any], vis_ct: str) -> str:
    """
    Build a lush, multi-layered 3D wetland mud bog visual assembly (zero-collision flush layers so
    wedges and wheels never snag on a hard vertical step, while looking rich and organic in 3D!).
    """
    tag = mz["tag"]
    mx, my = float(mz["x"]), float(mz["y"])
    rx, ry = float(mz["rx"]), float(mz["ry"])

    # Outer mossy bank -> dark peat mud -> glossy wet chocolate inner pool -> mud bubble rings & stepping pebbles
    lines = [
        f'    <body name="env_mud_vis_{tag}" pos="{mx:.4f} {my:.4f} 0.0">',
        f'      <geom name="env_mud_moss_{tag}" type="ellipsoid" pos="0 0 -0.004" size="{rx + 0.12:.4f} {ry + 0.12:.4f} 0.007" rgba="0.24 0.42 0.25 1" {vis_ct}/>',
        f'      <geom name="env_mud_rim_{tag}" type="ellipsoid" pos="0 0 -0.002" size="{rx + 0.03:.4f} {ry + 0.03:.4f} 0.007" rgba="0.33 0.24 0.15 1" {vis_ct}/>',
        f'      <geom name="env_mud_peat_{tag}" type="ellipsoid" pos="0 0 0.000" size="{rx:.4f} {ry:.4f} 0.0075" rgba="0.21 0.13 0.08 1" {vis_ct}/>',
        f'      <geom name="env_mud_wet1_{tag}" type="ellipsoid" pos="{0.12 * rx:.4f} {0.08 * ry:.4f} 0.0022" size="{0.74 * rx:.4f} {0.70 * ry:.4f} 0.007" rgba="0.28 0.17 0.10 1" {vis_ct}/>',
        f'      <geom name="env_mud_wet2_{tag}" type="ellipsoid" pos="{-0.18 * rx:.4f} {-0.14 * ry:.4f} 0.0035" size="{0.46 * rx:.4f} {0.44 * ry:.4f} 0.0065" rgba="0.35 0.22 0.13 1" {vis_ct}/>',
    ]
    # Decorative swamp mud bubbles & shore pebbles
    bubbles = [
        (-0.36 * rx, 0.28 * ry, 0.075),
        (0.38 * rx, -0.24 * ry, 0.065),
        (0.05 * rx, 0.36 * ry, 0.055),
        (-0.22 * rx, -0.35 * ry, 0.060),
    ]
    for bi, (bx, by, br) in enumerate(bubbles):
        lines.append(
            f'      <geom name="env_mud_bub_{tag}_{bi}" type="cylinder" pos="{bx:.4f} {by:.4f} 0.0085" '
            f'size="{br:.4f} 0.002" rgba="0.44 0.29 0.16 1" {vis_ct}/>'
        )
    pebbles = [
        (0.96 * rx, 0.22 * ry, 0.09, 0.065, 18.0),
        (-0.94 * rx, -0.26 * ry, 0.085, 0.060, -25.0),
        (0.18 * rx, -0.97 * ry, 0.095, 0.065, 42.0),
        (-0.24 * rx, 0.96 * ry, 0.080, 0.055, -12.0),
    ]
    for pi, (px, py, sx_p, sy_p, yaw_p) in enumerate(pebbles):
        lines.append(
            f'      <geom name="env_mud_rock_{tag}_{pi}" type="ellipsoid" pos="{px:.4f} {py:.4f} 0.006" '
            f'euler="0 0 {yaw_p:.1f}" size="{sx_p:.4f} {sy_p:.4f} 0.014" rgba="0.40 0.44 0.39 1" {vis_ct}/>'
        )
    lines.append("    </body>")
    return "\n".join(lines)


def build_arena_mjcf(stage_key: str) -> Tuple[str, str, List[str]]:
    """
    Build MuJoCo <worldbody> and <actuator> XML snippets for the chosen v4.0 diorama arena.
    Environment solid geoms use contype='1' conaffinity='6' (collides with CarA=2 and CarB=4, zero self-collision).
    Returns: (worldbody_xml, actuator_xml, trap_actuator_names)
    """
    env_ct = 'contype="1" conaffinity="6"'
    vis_ct = 'contype="0" conaffinity="0"'

    common_floor = f"""
    <light name="sun_key" pos="4.8 -6.0 9.5" dir="-0.35 0.46 -0.81" diffuse="0.96 0.93 0.88" specular="0.42 0.42 0.45"/>
    <light name="fill_cool" pos="-5.2 5.0 7.0" dir="0.40 -0.40 -0.82" diffuse="0.44 0.52 0.65" specular="0.18 0.20 0.25"/>
    <geom name="env_catch_floor" type="plane" pos="0 0 {FLOOR_Z}" size="24 24 0.1" rgba="0.08 0.10 0.15 1" friction="0.9 0.01 0.003" {env_ct}/>
    """

    spring_bodies: List[str] = []
    spring_acts: List[str] = []
    act_names: List[str] = []
    for sp in STAGE_SPRINGS[stage_key]:
        b_xml, a_xml, anm = _build_flush_spring_xml(sp, env_ct, vis_ct)
        spring_bodies.append(b_xml)
        spring_acts.append(a_xml)
        act_names.append(anm)

    # ========================================================================
    # STAGE 1: CYBER-NEON QUAD-CATAPULT RING (霓虹四象·赛博弹射台)
    # ========================================================================
    if stage_key == "spring_arena":
        # 8 analytical rotated boxes for exact glitch-free Box-vs-Cylinder physics on the flat inner deck
        n_phys_deck = 16
        r_in = 2.45
        r_out = 3.45
        hy_in = r_in * math.tan(math.pi / n_phys_deck) * 1.02
        center_boxes: List[str] = []
        for k in range(n_phys_deck // 2):
            yaw_c = k * (360.0 / n_phys_deck)
            center_boxes.append(
                f'    <geom name="env_s1_center_deck_{k}" type="box" '
                f'pos="0 0 -{DECK_HALF_THICK}" euler="0 0 {yaw_c:.3f}" '
                f'size="{r_in:.5f} {hy_in:.5f} {DECK_HALF_THICK}" '
                f'rgba="0.14 0.17 0.23 1" friction="1.05 0.010 0.003" {env_ct}/>'
            )

        # 32-sector seamless -7.5° cambered danger slope (midpoint tangential width -> <2.4mm seam precision!)
        n_slope = 32
        horiz_run = r_out - r_in
        slope_deg = -7.5
        slab_hz = 0.20
        hy_mid = 0.5 * (r_in + r_out) * math.tan(math.pi / n_slope) * 1.04
        slope_slabs: List[str] = []
        for k in range(n_slope):
            yaw = k * (360.0 / n_slope)
            rad = math.radians(yaw)
            x0, y0 = r_in * math.cos(rad), r_in * math.sin(rad)
            c_xyz, qstr, hx = _slope_quat_and_center(x0, y0, -0.001, horiz_run, yaw, slope_deg, slab_hz)
            col = "0.12 0.14 0.20 1" if k % 2 == 0 else "0.26 0.12 0.16 1"
            slope_slabs.append(
                f'    <geom name="env_s1_slope_{k}" type="box" '
                f'pos="{c_xyz[0]:.5f} {c_xyz[1]:.5f} {c_xyz[2]:.5f}" quat="{qstr}" '
                f'size="{hx:.5f} {hy_mid:.5f} {slab_hz:.4f}" rgba="{col}" friction="0.95 0.010 0.003" {env_ct}/>'
            )

        # Cyber-stadium visual surface inlays, compass target, hazard rings & under-stage reactor struts
        vis_layers: List[str] = [
            f'    <geom name="env_s1_vis_skin" type="cylinder" pos="0 0 0.0010" size="{r_in - 0.01:.3f} 0.0010" rgba="0.15 0.18 0.25 1" {vis_ct}/>',
            f'    <geom name="env_s1_vis_inner_plate" type="cylinder" pos="0 0 0.0022" size="1.72 0.0010" rgba="0.18 0.22 0.30 1" {vis_ct}/>',
            f'    <geom name="env_s1_vis_core_ring" type="cylinder" pos="0 0 0.0036" size="0.62 0.0012" rgba="0.12 0.82 0.96 1" {vis_ct}/>',
            f'    <geom name="env_s1_vis_core_dark" type="cylinder" pos="0 0 0.0050" size="0.55 0.0012" rgba="0.11 0.14 0.20 1" {vis_ct}/>',
            f'    <geom name="env_s1_vis_core_bullseye" type="cylinder" pos="0 0 0.0064" size="0.18 0.0012" rgba="0.98 0.72 0.16 1" {vis_ct}/>',
            f'    <geom name="env_s1_vis_cross_ns" type="box" pos="0 0 0.0032" size="0.022 2.32 0.0010" rgba="0.16 0.56 0.74 1" {vis_ct}/>',
            f'    <geom name="env_s1_vis_cross_ew" type="box" pos="0 0 0.0032" size="1.42 0.022 0.0010" rgba="0.16 0.56 0.74 1" {vis_ct}/>',
            _build_spawn_pad_xml("env_s1_spawn_a", -1.95, 0.0, 0.002, 0.0, "0.18 0.84 0.98 1", vis_ct),
            _build_spawn_pad_xml("env_s1_spawn_b", 1.95, 0.0, 0.002, 180.0, "0.98 0.38 0.50 1", vis_ct),
            # Under-stage sci-fi reactor pedestal & glowing plasma rings
            f'    <geom name="env_s1_pedestal_core" type="cylinder" pos="0 0 -1.05" size="1.28 0.78" rgba="0.11 0.14 0.20 1" {vis_ct}/>',
            f'    <geom name="env_s1_reactor_glow1" type="cylinder" pos="0 0 -0.58" size="1.42 0.035" rgba="0.12 0.82 0.98 1" {vis_ct}/>',
            f'    <geom name="env_s1_reactor_glow2" type="cylinder" pos="0 0 -0.92" size="1.36 0.030" rgba="0.96 0.58 0.12 1" {vis_ct}/>',
        ]
        # Amber transition hazard ring at r=2.44m & crimson rim warning ring at r=3.41m
        vis_layers.extend(
            _build_ring_segments_xml(
                "env_s1_hazard_bevel", 2.435, 0.0038, 0.038, 0.0020, 32,
                "0.98 0.72 0.14 1", vis_ct, alt_rgba="0.16 0.18 0.24 1"
            )
        )
        rim_z = -(horiz_run - 0.04) * math.tan(math.radians(7.5)) + 0.005
        vis_layers.extend(
            _build_ring_segments_xml(
                "env_s1_cliff_rim", 3.41, rim_z, 0.032, 0.0025, 32,
                "0.96 0.24 0.32 1", vis_ct, alt_rgba="0.22 0.12 0.16 1"
            )
        )
        # 8 diagonal under-deck steel ribs
        for k_rib in range(8):
            ryaw = k_rib * 45.0 + 22.5
            rrad = math.radians(ryaw)
            rx, ry = 2.05 * math.cos(rrad), 2.05 * math.sin(rrad)
            vis_layers.append(
                f'    <geom name="env_s1_under_rib_{k_rib}" type="box" '
                f'pos="{rx:.4f} {ry:.4f} -0.56" euler="0 22 {ryaw:.2f}" '
                f'size="0.88 0.08 0.09" rgba="0.18 0.22 0.30 1" {vis_ct}/>'
            )

        world_xml = (
            common_floor
            + "\n"
            + "\n".join(center_boxes)
            + "\n"
            + "\n".join(slope_slabs)
            + "\n"
            + "\n".join(vis_layers)
            + "\n"
            + "\n".join(spring_bodies)
            + "\n"
        )
        return world_xml, "\n".join(spring_acts), act_names

    # ========================================================================
    # STAGE 2: EMERALD BAYOU & VISCOUS MUDLANDS (翡翠湿地·泥沼陷车台)
    # ========================================================================
    elif stage_key == "mud_arena":
        half_x = 3.60
        core_hy = 2.20
        bank_run = 0.70  # North & South mossy cliff banks: |y| in [2.20, 2.90]
        bank_deg = -7.0
        slab_hz = 0.22

        cn_xyz, qn_str, hxn = _slope_quat_and_center(0.0, core_hy, 0.0, bank_run, 90.0, bank_deg, slab_hz)
        cs_xyz, qs_str, hxs = _slope_quat_and_center(0.0, -core_hy, 0.0, bank_run, -90.0, bank_deg, slab_hz)

        # Curved warm sandstone flagstone stepping tiles along North & South dry causeways (zero mud overlap!)
        stone_tiles: List[str] = []
        causeway_pts = [
            # South dry causeway (sweeps cleanly south of mud_core, then threads between mud_core and mud_se!)
            (-2.12, -0.76, 0.34, 0.25, -26.0),
            (-1.52, -1.34, 0.36, 0.26, -15.0),
            (-0.78, -1.62, 0.38, 0.26, -4.0),
            (0.02, -1.64, 0.38, 0.26, 8.0),
            (0.80, -1.36, 0.35, 0.24, 28.0),
            (1.48, -0.74, 0.34, 0.24, 34.0),
            # North dry causeway (mirror-symmetric clear dry stone road!)
            (2.12, 0.76, 0.34, 0.25, -26.0),
            (1.52, 1.34, 0.36, 0.26, -15.0),
            (0.78, 1.62, 0.38, 0.26, -4.0),
            (-0.02, 1.64, 0.38, 0.26, 8.0),
            (-0.80, 1.36, 0.35, 0.24, 28.0),
            (-1.48, 0.74, 0.34, 0.24, 34.0),
        ]
        for ti, (tx, ty, sx_t, sy_t, tyaw) in enumerate(causeway_pts):
            tcol = "0.56 0.51 0.40 1" if ti % 2 == 0 else "0.49 0.45 0.36 1"
            stone_tiles.append(
                f'    <geom name="env_s2_flagstone_{ti}" type="box" '
                f'pos="{tx:.4f} {ty:.4f} 0.0024" euler="0 0 {tyaw:.1f}" '
                f'size="{sx_t:.4f} {sy_t:.4f} 0.0022" rgba="{tcol}" {vis_ct}/>'
            )

        # Lush moss patches & glowing swamp-amber corner boundary beacons
        moss_and_pillars: List[str] = [
            f'    <geom name="env_s2_spawn_plaza_w" type="cylinder" pos="-2.55 0 0.0016" size="0.68 0.0016" rgba="0.46 0.43 0.35 1" {vis_ct}/>',
            f'    <geom name="env_s2_spawn_plaza_e" type="cylinder" pos="2.55 0 0.0016" size="0.68 0.0016" rgba="0.46 0.43 0.35 1" {vis_ct}/>',
            _build_spawn_pad_xml("env_s2_spawn_a", -2.55, 0.0, 0.003, 0.0, "0.22 0.88 0.96 1", vis_ct),
            _build_spawn_pad_xml("env_s2_spawn_b", 2.55, 0.0, 0.003, 180.0, "0.98 0.42 0.50 1", vis_ct),
            # West & East basalt cliff rim trims
            f'    <geom name="env_s2_rim_w" type="box" pos="-3.56 0 0.0035" size="0.036 {core_hy:.3f} 0.0035" rgba="0.86 0.64 0.20 1" {vis_ct}/>',
            f'    <geom name="env_s2_rim_e" type="box" pos="3.56 0 0.0035" size="0.036 {core_hy:.3f} 0.0035" rgba="0.86 0.64 0.20 1" {vis_ct}/>',
            # Stepped basalt rock columns underneath the island
            f'    <geom name="env_s2_basalt_c" type="cylinder" pos="0 0 -1.06" size="1.55 0.78" rgba="0.14 0.19 0.17 1" {vis_ct}/>',
            f'    <geom name="env_s2_basalt_w" type="cylinder" pos="-2.25 -0.25 -1.10" size="1.05 0.74" rgba="0.16 0.22 0.19 1" {vis_ct}/>',
            f'    <geom name="env_s2_basalt_e" type="cylinder" pos="2.25 0.25 -1.10" size="1.05 0.74" rgba="0.16 0.22 0.19 1" {vis_ct}/>',
        ]
        for mi, (mx_p, my_p, rx_p, ry_p) in enumerate([
            (-3.05, 1.55, 0.42, 0.30),
            (3.05, -1.55, 0.42, 0.30),
            (-2.95, -1.68, 0.48, 0.32),
            (2.95, 1.68, 0.48, 0.32),
        ]):
            moss_and_pillars.append(
                f'    <geom name="env_s2_moss_patch_{mi}" type="ellipsoid" pos="{mx_p:.3f} {my_p:.3f} 0.0018" '
                f'size="{rx_p:.3f} {ry_p:.3f} 0.003" rgba="0.23 0.46 0.28 1" {vis_ct}/>'
            )

        mud_visuals = [_build_mud_bog_visual_xml(mz, vis_ct) for mz in STAGE_MUD_ZONES["mud_arena"]]

        world_xml = (
            common_floor
            + f"""
    <!-- Solid Emerald-Slate Basalt Plateau + -7.0° North/South Wet Mossy Cliff Banks -->
    <geom name="env_s2_main_deck" type="box" pos="0 0 -{DECK_HALF_THICK}" size="{half_x:.3f} {core_hy:.3f} {DECK_HALF_THICK}" rgba="0.17 0.27 0.22 1" friction="1.06 0.010 0.003" {env_ct}/>
    <geom name="env_s2_bank_north" type="box" pos="{cn_xyz[0]:.5f} {cn_xyz[1]:.5f} {cn_xyz[2]:.5f}" quat="{qn_str}" size="{hxn:.5f} {half_x - 0.004:.4f} {slab_hz:.4f}" rgba="0.22 0.34 0.24 1" friction="0.82 0.010 0.003" {env_ct}/>
    <geom name="env_s2_bank_south" type="box" pos="{cs_xyz[0]:.5f} {cs_xyz[1]:.5f} {cs_xyz[2]:.5f}" quat="{qs_str}" size="{hxs:.5f} {half_x - 0.004:.4f} {slab_hz:.4f}" rgba="0.22 0.34 0.24 1" friction="0.82 0.010 0.003" {env_ct}/>
{chr(10).join(stone_tiles)}
{chr(10).join(moss_and_pillars)}
{chr(10).join(mud_visuals)}
    """
        )
        return world_xml, "", []

    # ========================================================================
    # STAGE 3: GLACIER OVERPASS & MIRROR-ICE RINK (极地冰川·溜冰飞坡台)
    # ========================================================================
    else:
        ridge_h = 0.26
        crest_hx = 0.42
        ramp_x0 = 1.45
        ramp_run = ramp_x0 - crest_hx  # 1.03m -> 14.17 deg smooth ski-jump!
        ramp_deg = math.degrees(math.atan2(ridge_h, ramp_run))
        slab_hz = 0.22

        rw_xyz, qw_r, hxr = _slope_quat_and_center(-ramp_x0, 0.0, 0.0, ramp_run, 0.0, ramp_deg, slab_hz)
        re_xyz, qe_r, _ = _slope_quat_and_center(ramp_x0, 0.0, 0.0, ramp_run, 180.0, ramp_deg, slab_hz)

        chute_x0 = 2.75
        chute_run = ICE_OUTER_HX - chute_x0  # 0.95m -> -8.0 deg icy cliff slide!
        chute_deg = -8.0
        iw_xyz, qw_i, hxi = _slope_quat_and_center(-chute_x0, 0.0, 0.0, chute_run, 180.0, chute_deg, slab_hz)
        ie_xyz, qe_i, _ = _slope_quat_and_center(chute_x0, 0.0, 0.0, chute_run, 0.0, chute_deg, slab_hz)

        ice_attr = f'priority="2" friction="0.10 0.002 0.0005" {env_ct}'
        ice_chute_attr = f'priority="2" friction="0.085 0.002 0.0005" {env_ct}'

        # Crystalline frost crack lines, ski-jump glowing chevrons & under-stage glacier crystal spires
        glacier_vis: List[str] = [
            _build_spawn_pad_xml("env_s3_spawn_a", -1.80, 0.0, 0.002, 0.0, "0.20 0.86 1.00 1", vis_ct),
            _build_spawn_pad_xml("env_s3_spawn_b", 1.80, 0.0, 0.002, 180.0, "0.98 0.40 0.54 1", vis_ct),
            # Frost-white snow cap, center compass medallion & glowing cyan runway stripes on the +0.26m Glacier Crest
            f'    <geom name="env_s3_crest_snow" type="box" pos="0 0 {ridge_h + 0.0018:.4f}" size="{crest_hx - 0.03:.3f} {ICE_INNER_HY - 0.04:.3f} 0.0018" rgba="0.88 0.94 1.00 1" {vis_ct}/>',
            f'    <geom name="env_s3_crest_line_w" type="box" pos="-{crest_hx - 0.04:.3f} 0 {ridge_h + 0.0038:.4f}" size="0.025 {ICE_INNER_HY - 0.05:.3f} 0.0018" rgba="0.14 0.78 0.98 1" {vis_ct}/>',
            f'    <geom name="env_s3_crest_line_e" type="box" pos="{crest_hx - 0.04:.3f} 0 {ridge_h + 0.0038:.4f}" size="0.025 {ICE_INNER_HY - 0.05:.3f} 0.0018" rgba="0.14 0.78 0.98 1" {vis_ct}/>',
            f'    <geom name="env_s3_crest_emblem" type="cylinder" pos="0 0 {ridge_h + 0.0048:.4f}" size="0.28 0.0015" rgba="0.18 0.78 0.98 1" {vis_ct}/>',
            f'    <geom name="env_s3_crest_emblem_in" type="cylinder" pos="0 0 {ridge_h + 0.0064:.4f}" size="0.18 0.0015" rgba="0.92 0.97 1.00 1" {vis_ct}/>',
            # Crisp frost boundary line marking where high-grip snow ends and Mirror-Ice begins!
            f'    <geom name="env_s3_ice_border_n" type="box" pos="0 {ICE_INNER_HY:.3f} 0.003" size="{ICE_INNER_HX:.3f} 0.026 0.0024" rgba="0.90 0.98 1.00 1" {vis_ct}/>',
            f'    <geom name="env_s3_ice_border_s" type="box" pos="0 -{ICE_INNER_HY:.3f} 0.003" size="{ICE_INNER_HX:.3f} 0.026 0.0024" rgba="0.90 0.98 1.00 1" {vis_ct}/>',
            f'    <geom name="env_s3_ice_border_w" type="box" pos="-{ICE_INNER_HX:.3f} 0 0.003" size="0.026 {ICE_INNER_HY:.3f} 0.0024" rgba="0.90 0.98 1.00 1" {vis_ct}/>',
            f'    <geom name="env_s3_ice_border_e" type="box" pos="{ICE_INNER_HX:.3f} 0 0.003" size="0.026 {ICE_INNER_HY:.3f} 0.0024" rgba="0.90 0.98 1.00 1" {vis_ct}/>',
            # Under-stage arctic glacier crystal pillars
            f'    <geom name="env_s3_glacier_core" type="cylinder" pos="0 0 -1.05" size="1.48 0.78" rgba="0.22 0.44 0.64 1" {vis_ct}/>',
            f'    <geom name="env_s3_crystal_1" type="box" pos="-1.95 -1.15 -0.92" euler="14 -18 25" size="0.42 0.42 0.68" rgba="0.34 0.68 0.90 1" {vis_ct}/>',
            f'    <geom name="env_s3_crystal_2" type="box" pos="1.95 1.15 -0.92" euler="-14 18 25" size="0.42 0.42 0.68" rgba="0.34 0.68 0.90 1" {vis_ct}/>',
            f'    <geom name="env_s3_crystal_3" type="box" pos="-1.85 1.25 -0.98" euler="-12 -15 -30" size="0.35 0.35 0.60" rgba="0.28 0.58 0.82 1" {vis_ct}/>',
            f'    <geom name="env_s3_crystal_4" type="box" pos="1.85 -1.25 -0.98" euler="12 15 -30" size="0.35 0.35 0.60" rgba="0.28 0.58 0.82 1" {vis_ct}/>',
        ]
        # Ski-jump glowing directional chevrons mounted flush along both West & East 14.2° ramps!
        for ri, y_lane in enumerate((-1.05, 0.0, 1.05)):
            z_mid = 0.5 * ridge_h + 0.008
            glacier_vis.append(
                f'    <body name="env_s3_ramp_chev_w_{ri}" pos="-0.94 {y_lane:.2f} {z_mid:.4f}" euler="0 -{ramp_deg:.2f} 0">\n'
                f'      <geom name="env_s3_rchevw_l_{ri}" type="box" pos="0 0.11 0" euler="0 0 -34" size="0.16 0.028 0.002" rgba="0.22 0.86 1.00 1" {vis_ct}/>\n'
                f'      <geom name="env_s3_rchevw_r_{ri}" type="box" pos="0 -0.11 0" euler="0 0 34" size="0.16 0.028 0.002" rgba="0.22 0.86 1.00 1" {vis_ct}/>\n'
                f'    </body>'
            )
            glacier_vis.append(
                f'    <body name="env_s3_ramp_chev_e_{ri}" pos="0.94 {y_lane:.2f} {z_mid:.4f}" euler="0 {ramp_deg:.2f} 180">\n'
                f'      <geom name="env_s3_rcheve_l_{ri}" type="box" pos="0 0.11 0" euler="0 0 -34" size="0.16 0.028 0.002" rgba="0.22 0.86 1.00 1" {vis_ct}/>\n'
                f'      <geom name="env_s3_rcheve_r_{ri}" type="box" pos="0 -0.11 0" euler="0 0 34" size="0.16 0.028 0.002" rgba="0.22 0.86 1.00 1" {vis_ct}/>\n'
                f'    </body>'
            )

        # Decorative crystalline frost cracks & translucent ice facet plates across the mirror-ice sheets
        cracks = [
            (-2.44, 1.15, 0.36, 28.0),
            (-2.44, -1.15, 0.38, -36.0),
            (2.44, 1.15, 0.36, -24.0),
            (2.44, -1.15, 0.36, 32.0),
            (-1.15, 2.48, 0.44, 16.0),
            (1.15, 2.48, 0.44, -19.0),
            (-1.15, -2.48, 0.44, -22.0),
            (1.15, -2.48, 0.44, 18.0),
            (0.0, 2.52, 0.32, 62.0),
            (0.0, -2.52, 0.32, -58.0),
        ]
        for ci, (cx_i, cy_i, clen, cyaw) in enumerate(cracks):
            glacier_vis.append(
                f'    <geom name="env_s3_frost_crack_{ci}" type="box" pos="{cx_i:.3f} {cy_i:.3f} 0.0028" '
                f'euler="0 0 {cyaw:.1f}" size="{clen:.3f} 0.015 0.0016" rgba="0.90 0.98 1.00 1" {vis_ct}/>'
            )
        # Rim冰晶棱柱点缀 (outside driving rim at the 4 cliff tips, contype=0 zero collision!)
        for ki, (kx, ky, kz, keul) in enumerate([
            (-3.55, 2.72, -0.06, "18 -14 35"),
            (3.55, 2.72, -0.06, "18 14 -35"),
            (-3.55, -2.72, -0.06, "-18 -14 -35"),
            (3.55, -2.72, -0.06, "-18 14 35"),
        ]):
            glacier_vis.append(
                f'    <geom name="env_s3_rim_crystal_{ki}" type="box" pos="{kx:.2f} {ky:.2f} {kz:.2f}" '
                f'euler="{keul}" size="0.16 0.16 0.28" rgba="0.68 0.92 1.00 1" {vis_ct}/>'
            )

        ns_band_cy = 0.5 * (ICE_INNER_HY + ICE_OUTER_HY)
        ns_band_hy = 0.5 * (ICE_OUTER_HY - ICE_INNER_HY)
        ew_band_cx = 0.5 * (ICE_INNER_HX + chute_x0)
        ew_band_hx = 0.5 * (chute_x0 - ICE_INNER_HX)

        world_xml = (
            common_floor
            + f"""
    <!-- High-Grip Frosted Alpine Core & Spawn Launch Strips -->
    <geom name="env_s3_snow_core" type="box" pos="0 0 -{DECK_HALF_THICK}" size="{ICE_INNER_HX:.3f} {ICE_INNER_HY:.3f} {DECK_HALF_THICK}" rgba="0.30 0.38 0.50 1" friction="1.08 0.010 0.003" {env_ct}/>

    <!-- Central +0.26m Glacier Ski-Jump Flyover Ramp -->
    <geom name="env_s3_ramp_west" type="box" pos="{rw_xyz[0]:.5f} {rw_xyz[1]:.5f} {rw_xyz[2]:.5f}" quat="{qw_r}" size="{hxr:.5f} {ICE_INNER_HY - 0.004:.4f} {slab_hz:.4f}" rgba="0.46 0.58 0.74 1" friction="1.08 0.010 0.003" {env_ct}/>
    <geom name="env_s3_ramp_east" type="box" pos="{re_xyz[0]:.5f} {re_xyz[1]:.5f} {re_xyz[2]:.5f}" quat="{qe_r}" size="{hxr:.5f} {ICE_INNER_HY - 0.004:.4f} {slab_hz:.4f}" rgba="0.46 0.58 0.74 1" friction="1.08 0.010 0.003" {env_ct}/>
    <geom name="env_s3_ridge_crest" type="box" pos="0 0 {ridge_h - slab_hz:.5f}" size="{crest_hx:.4f} {ICE_INNER_HY - 0.004:.4f} {slab_hz:.4f}" rgba="0.62 0.76 0.90 1" friction="1.08 0.010 0.003" {env_ct}/>

    <!-- Ultra-Low-Friction Mirror-Ice Skating Border Rink (priority=2, mu=0.10) -->
    <geom name="env_s3_ice_north" type="box" pos="0 {ns_band_cy:.4f} -{DECK_HALF_THICK}" size="{chute_x0:.4f} {ns_band_hy:.4f} {DECK_HALF_THICK}" rgba="0.54 0.85 0.97 1" {ice_attr}/>
    <geom name="env_s3_ice_south" type="box" pos="0 -{ns_band_cy:.4f} -{DECK_HALF_THICK}" size="{chute_x0:.4f} {ns_band_hy:.4f} {DECK_HALF_THICK}" rgba="0.54 0.85 0.97 1" {ice_attr}/>
    <geom name="env_s3_ice_west" type="box" pos="-{ew_band_cx:.4f} 0 -{DECK_HALF_THICK}" size="{ew_band_hx:.4f} {ICE_INNER_HY:.4f} {DECK_HALF_THICK}" rgba="0.58 0.88 0.98 1" {ice_attr}/>
    <geom name="env_s3_ice_east" type="box" pos="{ew_band_cx:.4f} 0 -{DECK_HALF_THICK}" size="{ew_band_hx:.4f} {ICE_INNER_HY:.4f} {DECK_HALF_THICK}" rgba="0.58 0.88 0.98 1" {ice_attr}/>

    <!-- -8.0° Downward Mirror-Ice Cliff Chutes on West & East (priority=2, mu=0.085) -->
    <geom name="env_s3_ice_chute_west" type="box" pos="{iw_xyz[0]:.5f} {iw_xyz[1]:.5f} {iw_xyz[2]:.5f}" quat="{qw_i}" size="{hxi:.5f} {ICE_OUTER_HY - 0.003:.4f} {slab_hz:.4f}" rgba="0.38 0.73 0.92 1" {ice_chute_attr}/>
    <geom name="env_s3_ice_chute_east" type="box" pos="{ie_xyz[0]:.5f} {ie_xyz[1]:.5f} {ie_xyz[2]:.5f}" quat="{qe_i}" size="{hxi:.5f} {ICE_OUTER_HY - 0.003:.4f} {slab_hz:.4f}" rgba="0.38 0.73 0.92 1" {ice_chute_attr}/>
{chr(10).join(glacier_vis)}
    """
        )
        return world_xml, "", []


def create_hazard_runtime(stage_key: str) -> Dict[str, Any]:
    """Initialize per-spring cooldown/runtime state for a match."""
    state: Dict[str, Any] = {}
    for sp in STAGE_SPRINGS[stage_key]:
        state[sp["tag"]] = {
            "ready_t": float(sp["arm_t"]),
            "fire_start_t": -9.0,
            "fire_until_t": -9.0,
        }
    return state


def point_mud_factor(stage_key: str, x: float, y: float) -> float:
    """Return smooth mud immersion in [0.0, 1.0] at world (x, y) on `mud_arena` (0.0 elsewhere)."""
    zones = STAGE_MUD_ZONES.get(stage_key, [])
    if not zones:
        return 0.0
    best = 0.0
    for mz in zones:
        dx = (x - float(mz["x"])) / float(mz["rx"])
        dy = (y - float(mz["y"])) / float(mz["ry"])
        q = math.hypot(dx, dy)
        if q <= 0.90:
            return 1.0
        if q < 1.08:
            f = (1.08 - q) / 0.18
            if f > best:
                best = f
    return best


def compute_car_mud_factor(
    stage_key: str,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    root_body_id: int,
    solid_gids: List[int],
) -> float:
    """
    Compute effective mud immersion [0.0, 1.0] for a vehicle across its root body and grounded geoms.
    Treats all vehicle/tire types identically: stepping wheels or hull into a mud bog smoothly ramps to 1.0!
    """
    zones = STAGE_MUD_ZONES.get(stage_key, [])
    if not zones:
        return 0.0
    rpos = data.xpos[root_body_id]
    if float(rpos[2]) > 0.52 or float(rpos[2]) < -0.30:
        return 0.0

    best = point_mud_factor(stage_key, float(rpos[0]), float(rpos[1]))
    if best >= 0.99:
        return 1.0
    for gid in solid_gids:
        gp = data.geom_xpos[gid]
        if float(gp[2]) <= 0.24:
            f = point_mud_factor(stage_key, float(gp[0]), float(gp[1]))
            if f > best:
                best = f
                if best >= 0.99:
                    return 1.0
    return best


def point_on_ice(stage_key: str, x: float, y: float) -> bool:
    """Return True if world (x, y) lies on Stage 3's Mirror-Ice Skating surface."""
    if stage_key != "ice_ridge":
        return False
    if abs(x) > ICE_OUTER_HX or abs(y) > ICE_OUTER_HY:
        return False
    return (abs(x) >= ICE_INNER_HX) or (abs(y) >= ICE_INNER_HY)


def _nearest_mud_query(stage_key: str, x: float, y: float) -> Tuple[float, float, float, Tuple[float, float]]:
    """Return (mud_x, mud_y, signed_or_zero_edge_dist, unit_dir_to_mud) for nearest mud zone."""
    zones = STAGE_MUD_ZONES.get(stage_key, [])
    if not zones:
        return 0.0, 0.0, 9.9, (1.0, 0.0)
    best_dist = 9.9
    best_xy = (0.0, 0.0)
    best_dir = (1.0, 0.0)
    for mz in zones:
        mx, my = float(mz["x"]), float(mz["y"])
        rx, ry = float(mz["rx"]), float(mz["ry"])
        dx, dy = mx - x, my - y
        center_d = math.hypot(dx, dy)
        nx = (x - mx) / rx
        ny = (y - my) / ry
        qn = math.hypot(nx, ny)
        if qn <= 1.0:
            edge_d = 0.0
        else:
            r_eff = center_d / qn
            edge_d = max(0.0, center_d - r_eff)
        if edge_d < best_dist:
            best_dist = edge_d
            best_xy = (mx, my)
            if center_d > 1e-5:
                best_dir = (dx / center_d, dy / center_d)
    return best_xy[0], best_xy[1], best_dist, best_dir


def _nearest_ice_query(stage_key: str, x: float, y: float) -> Tuple[float, Tuple[float, float]]:
    """Return (dist_to_ice_sheet, unit_dir_toward_nearest_ice) on `ice_ridge` (9.9 elsewhere)."""
    if stage_key != "ice_ridge":
        return 9.9, (1.0, 0.0)
    if abs(x) >= ICE_INNER_HX or abs(y) >= ICE_INNER_HY:
        # Already on ice: point outward toward whichever ice band we are on
        if abs(x) - ICE_INNER_HX >= abs(y) - ICE_INNER_HY:
            return 0.0, (1.0 if x >= 0.0 else -1.0, 0.0)
        return 0.0, (0.0, 1.0 if y >= 0.0 else -1.0)

    dw = x - (-ICE_INNER_HX)
    de = ICE_INNER_HX - x
    ds = y - (-ICE_INNER_HY)
    dn = ICE_INNER_HY - y
    dmin = min(dw, de, ds, dn)
    if dmin == dw:
        return dmin, (-1.0, 0.0)
    if dmin == de:
        return dmin, (1.0, 0.0)
    if dmin == ds:
        return dmin, (0.0, -1.0)
    return dmin, (0.0, 1.0)


def _car_overlaps_spring_pad(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    gids: List[int],
    sx: float,
    sy: float,
    sz: float,
    cp: float,
    sp_sin: float,
) -> bool:
    """
    Return True as soon as ANY wheel, wedge, armor, or chassis geom of the car presses onto
    ANY edge or interior region of the 0.96m x 0.88m physical spring plate!
    """
    box_t = int(mujoco.mjtGeom.mjGEOM_BOX)
    cyl_t = int(mujoco.mjtGeom.mjGEOM_CYLINDER)
    ux_x, ux_y = cp, sp_sin
    uy_x, uy_y = -sp_sin, cp

    for gid in gids:
        gp = data.geom_xpos[gid]
        dx = float(gp[0]) - sx
        dy = float(gp[1]) - sy
        if dx * dx + dy * dy > 1.35:
            continue

        gtype = int(model.geom_type[gid])
        gsz = model.geom_size[gid]
        if gtype == box_t:
            hx, hy, hz = float(gsz[0]), float(gsz[1]), float(gsz[2])
        elif gtype == cyl_t:
            hx, hy, hz = float(gsz[0]), float(gsz[0]), float(gsz[1])
        else:
            r = float(gsz[0])
            hx, hy, hz = r, r, r

        gR = data.geom_xmat[gid].reshape(3, 3)
        wz = abs(float(gR[2, 0])) * hx + abs(float(gR[2, 1])) * hy + abs(float(gR[2, 2])) * hz
        z_bot = float(gp[2]) - wz
        z_top = float(gp[2]) + wz
        if z_bot > sz + 0.085 or z_top < sz - 0.06:
            continue

        c0x = abs(ux_x * float(gR[0, 0]) + ux_y * float(gR[1, 0]))
        c1x = abs(ux_x * float(gR[0, 1]) + ux_y * float(gR[1, 1]))
        c2x = abs(ux_x * float(gR[0, 2]) + ux_y * float(gR[1, 2]))
        rx_loc = c0x * hx + c1x * hy + c2x * hz

        c0y = abs(uy_x * float(gR[0, 0]) + uy_y * float(gR[1, 0]))
        c1y = abs(uy_x * float(gR[0, 1]) + uy_y * float(gR[1, 1]))
        c2y = abs(uy_x * float(gR[0, 2]) + uy_y * float(gR[1, 2]))
        ry_loc = c0y * hx + c1y * hy + c2y * hz

        x_loc = dx * ux_x + dy * ux_y
        y_loc = dx * uy_x + dy * uy_y

        if (x_loc + rx_loc >= -0.465) and (x_loc - rx_loc <= 0.41) and (abs(y_loc) - ry_loc <= 0.425):
            return True
    return False


def step_arena_hazards(
    stage_key: str,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    t_sec: float,
    trap_act_ids: List[int],
    car_positions: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = None,
    hazard_state: Optional[Dict[str, Any]] = None,
    car_geom_groups: Optional[Tuple[List[int], List[int]]] = None,
) -> List[Tuple[str, str]]:
    """
    Update tactical hinged diagonal catapult plates deterministically (Stage 1 `spring_arena`).
    Edge-Sensitive Trigger & Zero-Pinch Retraction Guarantee.
    """
    springs = STAGE_SPRINGS[stage_key]
    triggered_events: List[Tuple[str, str]] = []
    if not springs:
        return triggered_events

    if car_positions is None or hazard_state is None:
        for idx, sp in enumerate(springs):
            if idx >= len(trap_act_ids) or trap_act_ids[idx] < 0:
                continue
            ph = (t_sec + idx * 0.75) % 2.4
            ext = float(sp["max_rad"]) if ph < 0.35 else 0.0
            data.ctrl[trap_act_ids[idx]] = ext
        return triggered_events

    pa, pb = car_positions
    for idx, sp in enumerate(springs):
        if idx >= len(trap_act_ids) or trap_act_ids[idx] < 0:
            continue
        tag = sp["tag"]
        st = hazard_state[tag]
        if "pad_gid" not in st:
            st["pad_gid"] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"trap_spring_{tag}_pad")
            j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"j_trap_spring_{tag}")
            st["j_qposadr"] = int(model.jnt_qposadr[j_id]) if j_id >= 0 else -1

        pad_gid = int(st["pad_gid"])
        j_qadr = int(st["j_qposadr"])
        cur_qpos = float(data.qpos[j_qadr]) if j_qadr >= 0 else 0.0

        sx, sy, sz = float(sp["x"]), float(sp["y"]), float(sp["z"])
        phi = math.radians(float(sp["fling_yaw"]))
        cp, sp_sin = math.cos(phi), math.sin(phi)
        max_rad = float(sp["max_rad"])

        if t_sec >= st["ready_t"] and cur_qpos <= 0.035:
            dax, day = pa[0] - sx, pa[1] - sy
            dbx, dby = pb[0] - sx, pb[1] - sy
            if car_geom_groups is not None:
                hit_a = _car_overlaps_spring_pad(model, data, car_geom_groups[0], sx, sy, sz, cp, sp_sin)
                hit_b = _car_overlaps_spring_pad(model, data, car_geom_groups[1], sx, sy, sz, cp, sp_sin)
            else:
                ax_loc = dax * cp + day * sp_sin
                ay_loc = -dax * sp_sin + day * cp
                bx_loc = dbx * cp + dby * cp
                by_loc = -dbx * sp_sin + dby * cp
                hit_a = (-0.47 <= ax_loc <= 0.41) and (abs(ay_loc) <= 0.43) and (-0.20 <= (pa[2] - sz) <= 0.44)
                hit_b = (-0.47 <= bx_loc <= 0.41) and (abs(by_loc) <= 0.43) and (-0.20 <= (pb[2] - sz) <= 0.44)

            if hit_a or hit_b:
                st["fire_start_t"] = t_sec
                st["fire_until_t"] = t_sec + 0.22
                st["ready_t"] = t_sec + float(sp["cooldown"])
                da = math.hypot(dax, day)
                db = math.hypot(dbx, dby)
                if hit_a and (not hit_b or da <= db):
                    triggered_events.append(("A", f"trap_spring_{tag}"))
                if hit_b and (not hit_a or db < da):
                    triggered_events.append(("B", f"trap_spring_{tag}"))

        is_launching_up = (t_sec < st["fire_until_t"])
        if pad_gid >= 0:
            model.geom_contype[pad_gid] = 1 if is_launching_up else 0
            model.geom_conaffinity[pad_gid] = 6 if is_launching_up else 0

        if is_launching_up:
            ctrl_val = max_rad
        elif t_sec < st["fire_until_t"] + 0.34:
            frac = 1.0 - (t_sec - st["fire_until_t"]) / 0.34
            ctrl_val = max(0.0, max_rad * frac)
        else:
            ctrl_val = 0.0
        data.ctrl[trap_act_ids[idx]] = ctrl_val

    return triggered_events


def step_arena_portals(*_args: Any, **_kwargs: Any) -> List[Tuple[str, str, str]]:
    """No-op stub retained for legacy API compatibility (portals removed in v4.0)."""
    return []


def try_warp_bolt(*_args: Any, **_kwargs: Any) -> Optional[Tuple[List[float], List[float]]]:
    """No-op stub retained for legacy API compatibility (portals removed in v4.0)."""
    return None


def query_arena_sensors(
    stage_key: str,
    x: float,
    y: float,
    t_sec: float,
    data: mujoco.MjData,
    hazard_geom_ids: List[int],
    enemy_xy: Optional[Tuple[float, float]] = None,
    hazard_state: Optional[Dict[str, Any]] = None,
    self_mud_factor: Optional[float] = None,
    enemy_mud_factor: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Compute raw physical cliff, hinged-catapult (`spring_*`), mud-bog (`mud_*`),
    and mirror-ice (`ice_*`) sensors for a vehicle at world (x, y).
    """
    if stage_key == "spring_arena":
        r = math.hypot(x, y)
        edge_dist = 3.45 - r
        edge_dir = (x / r, y / r) if r > 1e-4 else (1.0, 0.0)
        safe_pt = (0.0, 0.0)

    elif stage_key == "mud_arena":
        # 7.20m x 5.80m: x in [-3.60, +3.60], y in [-2.90, +2.90]
        mx_w = x - (-3.60)
        mx_e = 3.60 - x
        my_s = y - (-2.90)
        my_n = 2.90 - y
        edge_dist = min(mx_w, mx_e, my_s, my_n)
        if edge_dist == mx_w:
            edge_dir = (-1.0, 0.0)
        elif edge_dist == mx_e:
            edge_dir = (1.0, 0.0)
        elif edge_dist == my_s:
            edge_dir = (0.0, -1.0)
        else:
            edge_dir = (0.0, 1.0)
        safe_pt = (0.0, 0.0)

    else:  # ice_ridge (7.40m x 5.80m: x in [-3.70, +3.70], y in [-2.90, +2.90])
        mx_w = x - (-3.70)
        mx_e = 3.70 - x
        my_s = y - (-2.90)
        my_n = 2.90 - y
        edge_dist = min(mx_w, mx_e, my_s, my_n)
        if edge_dist == mx_w:
            edge_dir = (-1.0, 0.0)
        elif edge_dist == mx_e:
            edge_dir = (1.0, 0.0)
        elif edge_dist == my_s:
            edge_dir = (0.0, -1.0)
        else:
            edge_dir = (0.0, 1.0)
        safe_pt = (0.0, 0.0)

    # 1. Nearest Hinged Catapult (`spring_*`, active on Stage 1; 9.9 on Stages 2 & 3)
    springs = STAGE_SPRINGS.get(stage_key, [])
    sp_dist = 9.9
    sp_xy = (0.0, 0.0)
    sp_fling_yaw = 0.0
    sp_dir = (1.0, 0.0)
    sp_ready = False
    for sp in springs:
        sx, sy = float(sp["x"]), float(sp["y"])
        d = math.hypot(sx - x, sy - y)
        if d < sp_dist:
            sp_dist = d
            sp_xy = (sx, sy)
            sp_fling_yaw = float(sp["fling_yaw"])
            if d > 1e-5:
                sp_dir = ((sx - x) / d, (sy - y) / d)
            if hazard_state is not None and sp["tag"] in hazard_state:
                sp_ready = (t_sec >= float(hazard_state[sp["tag"]]["ready_t"]))
            else:
                sp_ready = True

    esp_xy = sp_xy
    esp_fling_yaw = sp_fling_yaw
    esp_dist = 9.9
    esp_ready = False
    if enemy_xy is not None and springs:
        ex, ey = enemy_xy
        for sp in springs:
            sx, sy = float(sp["x"]), float(sp["y"])
            ed = math.hypot(sx - ex, sy - ey)
            if ed < esp_dist:
                esp_dist = ed
                esp_xy = (sx, sy)
                esp_fling_yaw = float(sp["fling_yaw"])
                if hazard_state is not None and sp["tag"] in hazard_state:
                    esp_ready = (t_sec >= float(hazard_state[sp["tag"]]["ready_t"]))
                else:
                    esp_ready = True

    # 2. Mud Bog Sensors (`mud_*`, active on Stage 2 `mud_arena`; 9.9 / False elsewhere)
    mud_x, mud_y, mud_dist, mud_dir = _nearest_mud_query(stage_key, x, y)
    m_self = self_mud_factor if self_mud_factor is not None else point_mud_factor(stage_key, x, y)
    if enemy_mud_factor is not None:
        m_enemy = enemy_mud_factor
    elif enemy_xy is not None:
        m_enemy = point_mud_factor(stage_key, enemy_xy[0], enemy_xy[1])
    else:
        m_enemy = 0.0

    # 3. Mirror-Ice Sensors (`ice_*`, active on Stage 3 `ice_ridge`; 9.9 / False elsewhere)
    ice_dist, ice_dir = _nearest_ice_query(stage_key, x, y)
    on_ice_self = point_on_ice(stage_key, x, y)
    on_ice_enemy = point_on_ice(stage_key, enemy_xy[0], enemy_xy[1]) if enemy_xy is not None else False

    return {
        "edge_dist": edge_dist,
        "edge_dir": edge_dir,
        "safe_pt": safe_pt,
        "spring_x": sp_xy[0],
        "spring_y": sp_xy[1],
        "spring_dist": sp_dist,
        "spring_dir": sp_dir,
        "spring_fling_yaw_deg": sp_fling_yaw,
        "spring_ready": sp_ready,
        "enemy_spring_x": esp_xy[0],
        "enemy_spring_y": esp_xy[1],
        "enemy_spring_dist": esp_dist,
        "enemy_spring_fling_yaw_deg": esp_fling_yaw,
        "enemy_spring_ready": esp_ready,
        # Mud sensors (Stage 2)
        "in_mud": bool(m_self >= 0.25),
        "enemy_in_mud": bool(m_enemy >= 0.25),
        "mud_x": mud_x,
        "mud_y": mud_y,
        "mud_dist": mud_dist,
        "mud_dir": mud_dir,
        # Ice sensors (Stage 3)
        "on_ice": on_ice_self,
        "enemy_on_ice": on_ice_enemy,
        "ice_dist": ice_dist,
        "ice_dir": ice_dir,
        # Silent legacy defaults so archival v3 strategies never crash
        "portal_x": 0.0,
        "portal_y": 0.0,
        "portal_dist": 9.9,
        "portal_dir": (1.0, 0.0),
        "portal_out_x": 0.0,
        "portal_out_y": 0.0,
        "portal_out_yaw_deg": 0.0,
        "portal_ready": False,
        "enemy_portal_x": 0.0,
        "enemy_portal_y": 0.0,
        "enemy_portal_dist": 9.9,
        "enemy_portal_out_yaw_deg": 0.0,
        "enemy_portal_ready": False,
    }
