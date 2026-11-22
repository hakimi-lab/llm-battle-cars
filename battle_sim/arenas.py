"""
Three Snag-Free 3D BO3 Arenas for LLM Battle-Cars (v3.1 — Hinged Catapults, Ridge Vaults & Quantum Portals).

Design Philosophy (v3.1):
  - ZERO vertical solid pillars, ZERO narrow bridges, ZERO rotating cross-walls, and ZERO overhead pendulums!
  - Every BO3 stage has a 100% distinct geometry and signature interactive mechanism:
      1. Stage 1 ('spring_arena')    — 四象弹射·斜坡擂台:
           6.9m circular ring (flat center + -7.5° downward danger skirt) + 4 Hinged Diagonal Catapult Plates!
      2. Stage 2 ('camelback_ridge') — 双峰飞坡·弹力竞技场:
           7.2m x 5.9m plateau bisected by a +0.24m Central Tabletop Ridge Hill + 2 Ridge Catapults + 2 Vault Catapults!
      3. Stage 3 ('portal_arena')    — 双生折跃·传送门擂台:
           7.3m x 5.7m slipway deck (flat core + East/West -8.0° downward cliff aprons) + 2 Linked Bidirectional
           Pairs of Quantum Portals (Cyan Flank-Ambush Gates & Magenta Cliff-Slingshot Gates) that teleport BOTH
           vehicles (preserving & boosting speed along exit yaw!) AND fired Energy Cannon plasma bolts!
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import mujoco
import numpy as np

FLOOR_Z = -1.85
DECK_HALF_THICK = 0.25

ARENA_ORDER = ("spring_arena", "camelback_ridge", "portal_arena")

STAGE_ALIASES: Dict[str, str] = {
    "1": "spring_arena",
    "s1": "spring_arena",
    "stage1": "spring_arena",
    "stage_1": "spring_arena",
    "spring_arena": "spring_arena",
    "vortex_crown": "spring_arena",
    "cyclone_ring": "spring_arena",

    "2": "camelback_ridge",
    "s2": "camelback_ridge",
    "stage2": "camelback_ridge",
    "stage_2": "camelback_ridge",
    "camelback_ridge": "camelback_ridge",
    "twin_drawbridge": "camelback_ridge",
    "pendulum_bridge": "camelback_ridge",

    "3": "portal_arena",
    "s3": "portal_arena",
    "stage3": "portal_arena",
    "stage_3": "portal_arena",
    "portal_arena": "portal_arena",
    # Backward-compatible legacy aliases for Stage 3
    "turbo_bowl": "portal_arena",
    "forge_colosseum": "portal_arena",
    "crusher_pit": "portal_arena",
}

# Hinged Diagonal Catapult Plate definitions (Stages 1 & 2):
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
    "camelback_ridge": [
        {
            "tag": "ridge_n", "x": 0.0, "y": 1.18, "z": 0.24,
            "slope_deg": 0.0, "fling_yaw": 90.0,
            "axis": (0.0, 0.707, 0.707), "max_rad": CATAPULT_MAX_RAD, "cooldown": 2.10, "arm_t": 0.30,
        },
        {
            "tag": "ridge_s", "x": 0.0, "y": -1.18, "z": 0.24,
            "slope_deg": 0.0, "fling_yaw": -90.0,
            "axis": (0.0, -0.707, 0.707), "max_rad": CATAPULT_MAX_RAD, "cooldown": 2.10, "arm_t": 0.30,
        },
        {
            "tag": "vault_w", "x": -2.28, "y": -1.35, "z": 0.0,
            "slope_deg": 0.0, "fling_yaw": 18.0,
            "axis": (0.672, 0.218, 0.707), "max_rad": CATAPULT_MAX_RAD, "cooldown": 2.00, "arm_t": 0.30,
        },
        {
            "tag": "vault_e", "x": 2.28, "y": 1.35, "z": 0.0,
            "slope_deg": 0.0, "fling_yaw": -162.0,
            "axis": (-0.672, -0.218, 0.707), "max_rad": CATAPULT_MAX_RAD, "cooldown": 2.00, "arm_t": 0.30,
        },
    ],
    "portal_arena": [],
}

# Linked Bidirectional Quantum Portal definitions (Stage 3 'portal_arena'):
# - Pair 'cyan' (`cyan_nw` <-> `cyan_se`): Cross-arena flank/escape portals that eject INWARD across the deck!
# - Pair 'magenta' (`mag_w` <-> `mag_e`): Brawl-zone slingshot portals that eject OUTWARD toward the -8° West/East cliffs!
PORTAL_TRIGGER_R = 0.44

STAGE_PORTALS: Dict[str, List[Dict[str, Any]]] = {
    "spring_arena": [],
    "camelback_ridge": [],
    "portal_arena": [
        {
            "tag": "cyan_nw", "pair": "cyan",
            "x": -1.75, "y": 1.62, "z": 0.0, "gate_yaw": -40.0,
            "dest_tag": "cyan_se", "dest_x": 1.75, "dest_y": -1.62, "dest_z": 0.0, "dest_exit_yaw": 140.0,
            "cooldown": 0.65, "arm_t": 0.25,
        },
        {
            "tag": "cyan_se", "pair": "cyan",
            "x": 1.75, "y": -1.62, "z": 0.0, "gate_yaw": 140.0,
            "dest_tag": "cyan_nw", "dest_x": -1.75, "dest_y": 1.62, "dest_z": 0.0, "dest_exit_yaw": -40.0,
            "cooldown": 0.65, "arm_t": 0.25,
        },
        {
            "tag": "mag_sw", "pair": "magenta",
            "x": -1.12, "y": -1.08, "z": 0.0, "gate_yaw": 180.0,
            "dest_tag": "mag_ne", "dest_x": 1.12, "dest_y": 1.08, "dest_z": 0.0, "dest_exit_yaw": 0.0,
            "cooldown": 0.75, "arm_t": 0.35,
        },
        {
            "tag": "mag_ne", "pair": "magenta",
            "x": 1.12, "y": 1.08, "z": 0.0, "gate_yaw": 0.0,
            "dest_tag": "mag_sw", "dest_x": -1.12, "dest_y": -1.08, "dest_z": 0.0, "dest_exit_yaw": 180.0,
            "cooldown": 0.75, "arm_t": 0.35,
        },
    ],
}

ARENA_META: Dict[str, Dict[str, Any]] = {
    "spring_arena": {
        "stage_id": 1,
        "key": "spring_arena",
        "name_zh": "四象弹射·斜坡擂台",
        "name_en": "Quad-Catapult Cambered Ring",
        "spawn_a": (-1.95, 0.0),
        "spawn_b": (1.95, 0.0),
        "yaw_a": 0.0,
        "yaw_b": 180.0,
        "description": (
            "6.9m snag-free circular arena: flat center deck (r=0..2.45m) bordered by a smooth 7.5-deg downward-cambered "
            "danger slope (r=2.45..3.45m) and 360-deg cliff drop, plus 4 flush Hinged Diagonal Catapult Plates "
            "at (±1.18, ±1.18) whose outer edge is hinged on the floor and snaps up 56-deg to hurl cars diagonally!"
        ),
    },
    "camelback_ridge": {
        "stage_id": 2,
        "key": "camelback_ridge",
        "name_zh": "双峰飞坡·弹力竞技场",
        "name_en": "Tabletop Ridge & Vault-Catapult Plateau",
        "spawn_a": (-2.45, 0.0),
        "spawn_b": (2.45, 0.0),
        "yaw_a": 0.0,
        "yaw_b": 180.0,
        "description": (
            "Wide 7.2m x 5.9m snag-free open plateau with a full-width +0.24m Central Tabletop Ridge Hill "
            "(smooth 11-deg West & East slopes), 2 Hilltop Cliff-Ejector Hinged Catapults at (0, ±1.18), and 2 Slope-Foot "
            "Vault Hinged Catapults at (-2.28, -1.35) & (+2.28, +1.35) that launch cars diagonally over the ridge!"
        ),
    },
    "portal_arena": {
        "stage_id": 3,
        "key": "portal_arena",
        "name_zh": "双生折跃·传送门擂台",
        "name_en": "Quantum Slipway & Linked Portals",
        "spawn_a": (-2.10, 0.0),
        "spawn_b": (2.10, 0.0),
        "yaw_a": 0.0,
        "yaw_b": 180.0,
        "description": (
            "7.3m x 5.7m snag-free Quantum Slipway (flat 5.2m core + smooth -8.0-deg East & West downward cliff aprons) "
            "with 2 X-Linked Pairs of Portals: Cyan Flank Gates (-1.75,+1.62 <-> +1.75,-1.62, inward ambush!) and "
            "Magenta Slingshot Gates (-1.12,-1.08 <-> +1.12,+1.08, ejecting outward toward the East/West cliffs!). "
            "Teleports BOTH vehicles (with warp speed boost) AND Energy Cannon bolts!"
        ),
    },
}


def resolve_stage_key(stage: str) -> str:
    s = str(stage).strip().lower()
    if s not in STAGE_ALIASES:
        raise ValueError(
            f"Unknown stage {stage!r}. Choose 1/spring_arena, 2/camelback_ridge, or 3/portal_arena."
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
    """Build a flush-mounted Hinged Diagonal Catapult Plate (`铰链斜向弹射板`)."""
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
    frame_center = np.array([sx, sy, sz], dtype=np.float64) + 0.0020 * uz
    hinge_world = np.array([sx, sy, sz], dtype=np.float64) + hx * ux + 0.014 * uz
    max_rad = float(sp["max_rad"])

    is_vault = "vault" in tag
    pad_rgba = "0.12 0.78 0.88 1" if is_vault else "0.94 0.56 0.10 1"
    frame_rgba = "0.07 0.30 0.38 1" if is_vault else "0.36 0.18 0.08 1"
    trim_rgba = "0.62 0.98 1.00 1" if is_vault else "1.00 0.86 0.25 1"
    hinge_rgba = "0.78 0.84 0.92 1"

    aname = f"act_trap_spring_{tag}"
    jname = f"j_trap_spring_{tag}"
    body_xml = (
        f'    <geom name="env_spring_frame_{tag}" type="box" '
        f'pos="{frame_center[0]:.4f} {frame_center[1]:.4f} {frame_center[2]:.4f}" quat="{qstr}" '
        f'size="{hx + 0.045:.4f} {hy + 0.045:.4f} 0.0022" rgba="{frame_rgba}" {vis_ct}/>\n'
        f'    <geom name="env_spring_hinge_{tag}" type="cylinder" '
        f'pos="{hinge_world[0]:.4f} {hinge_world[1]:.4f} {hinge_world[2]:.4f}" quat="{qh_str}" '
        f'size="0.024 {hy + 0.035:.4f}" rgba="{hinge_rgba}" {vis_ct}/>\n'
        f'    <body name="trap_spring_{tag}" pos="{recess_center[0]:.4f} {recess_center[1]:.4f} {recess_center[2]:.4f}" quat="{qstr}">\n'
        f'      <joint name="{jname}" type="hinge" pos="{hx:.4f} 0 {hz:.4f}" axis="0 1 0" '
        f'range="0 {CATAPULT_MAX_DEG:.1f}" damping="8.0" armature="0.25"/>\n'
        f'      <geom name="trap_spring_{tag}_pad" type="box" pos="0 0 0" size="{hx:.4f} {hy:.4f} {hz:.4f}" '
        f'mass="14.0" rgba="{pad_rgba}" friction="0.55 0.005 0.002" {env_ct}/>\n'
        f'      <geom name="trap_spring_{tag}_cap" type="box" pos="-0.01 0 {hz + 0.0075:.4f}" '
        f'size="{hx - 0.025:.4f} {hy - 0.025:.4f} 0.0028" mass="0" rgba="{pad_rgba}" {vis_ct}/>\n'
        f'      <geom name="trap_spring_{tag}_chev_l" type="box" pos="0.06 0.11 {hz + 0.0115:.4f}" euler="0 0 -32" '
        f'size="0.16 0.032 0.0022" mass="0" rgba="{trim_rgba}" {vis_ct}/>\n'
        f'      <geom name="trap_spring_{tag}_chev_r" type="box" pos="0.06 -0.11 {hz + 0.0115:.4f}" euler="0 0 32" '
        f'size="0.16 0.032 0.0022" mass="0" rgba="{trim_rgba}" {vis_ct}/>\n'
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


def _build_portal_xml(pt: Dict[str, Any], vis_ct: str) -> str:
    """
    Build a zero-collision 3D Quantum Portal Gate (`contype="0" conaffinity="0"` so cars of ANY width/height
    never snag on the arch!) consisting of:
      - Flush dark base pad + glowing neon vortex core + concentric inner energy ring
      - Directional ejection chevron showing `gate_yaw` (direction objects shoot OUT of this gate)
      - 3D vertical holographic portal archway (`0.88m` wide x `0.58m` tall)
    """
    tag = pt["tag"]
    px, py, pz = float(pt["x"]), float(pt["y"]), float(pt["z"])
    yaw = float(pt["gate_yaw"])
    is_cyan = (pt["pair"] == "cyan")

    base_rgba = "0.06 0.22 0.32 1" if is_cyan else "0.26 0.07 0.28 1"
    glow_rgba = "0.10 0.88 1.00 1" if is_cyan else "0.94 0.24 0.96 1"
    core_rgba = "0.72 0.98 1.00 1" if is_cyan else "1.00 0.72 0.98 1"

    return (
        f'    <body name="env_portal_{tag}" pos="{px:.4f} {py:.4f} {pz:.4f}" euler="0 0 {yaw:.2f}">\n'
        f'      <geom name="env_portal_base_{tag}" type="cylinder" pos="0 0 0.003" size="0.46 0.003" rgba="{base_rgba}" {vis_ct}/>\n'
        f'      <geom name="env_portal_glow_{tag}" type="cylinder" pos="0 0 0.007" size="0.38 0.003" rgba="{glow_rgba}" {vis_ct}/>\n'
        f'      <geom name="env_portal_core_{tag}" type="cylinder" pos="0 0 0.011" size="0.21 0.0025" rgba="{core_rgba}" {vis_ct}/>\n'
        f'      <geom name="env_portal_chev_l_{tag}" type="box" pos="0.32 0.11 0.013" euler="0 0 -32" size="0.14 0.028 0.0025" rgba="{core_rgba}" {vis_ct}/>\n'
        f'      <geom name="env_portal_chev_r_{tag}" type="box" pos="0.32 -0.11 0.013" euler="0 0 32" size="0.14 0.028 0.0025" rgba="{core_rgba}" {vis_ct}/>\n'
        f'      <geom name="env_portal_arch_l_{tag}" type="box" pos="0 0.42 0.29" size="0.032 0.032 0.29" rgba="{glow_rgba}" {vis_ct}/>\n'
        f'      <geom name="env_portal_arch_r_{tag}" type="box" pos="0 -0.42 0.29" size="0.032 0.032 0.29" rgba="{glow_rgba}" {vis_ct}/>\n'
        f'      <geom name="env_portal_arch_top_{tag}" type="box" pos="0 0 0.58" size="0.036 0.45 0.034" rgba="{core_rgba}" {vis_ct}/>\n'
        f'    </body>'
    )


def build_arena_mjcf(stage_key: str) -> Tuple[str, str, List[str]]:
    """
    Build MuJoCo <worldbody> and <actuator> XML snippets for the chosen v3.1 arena.
    Environment solid geoms use contype='1' conaffinity='6' (collides with CarA=2 and CarB=4, zero self-collision).
    Returns: (worldbody_xml, actuator_xml, trap_actuator_names)
    """
    env_ct = 'contype="1" conaffinity="6"'
    vis_ct = 'contype="0" conaffinity="0"'

    common_floor = f"""
    <light name="sun_key" pos="4.5 -5.8 9.0" dir="-0.35 0.48 -0.80" diffuse="0.96 0.92 0.85" specular="0.35 0.35 0.35"/>
    <light name="fill_cool" pos="-4.8 4.8 6.5" dir="0.40 -0.40 -0.82" diffuse="0.44 0.50 0.60" specular="0.14 0.14 0.18"/>
    <geom name="env_catch_floor" type="plane" pos="0 0 {FLOOR_Z}" size="24 24 0.1" rgba="0.11 0.14 0.19 1" friction="0.9 0.01 0.003" {env_ct}/>
    """

    spring_bodies: List[str] = []
    spring_acts: List[str] = []
    act_names: List[str] = []
    for sp in STAGE_SPRINGS[stage_key]:
        b_xml, a_xml, anm = _build_flush_spring_xml(sp, env_ct, vis_ct)
        spring_bodies.append(b_xml)
        spring_acts.append(a_xml)
        act_names.append(anm)

    portal_bodies: List[str] = [_build_portal_xml(pt, vis_ct) for pt in STAGE_PORTALS[stage_key]]

    # ========================================================================
    # STAGE 1: SPRING ARENA (四象弹射·斜坡擂台)
    # ========================================================================
    if stage_key == "spring_arena":
        n_sec = 16
        r_in = 2.45
        r_out = 3.45
        horiz_run = r_out - r_in
        slope_deg = -7.5
        slab_hz = 0.22
        hy_tang = r_out * math.tan(math.pi / n_sec) * 1.04
        hy_in = r_in * math.tan(math.pi / n_sec) * 1.03

        center_boxes: List[str] = []
        for k in range(n_sec // 2):
            yaw_c = k * (360.0 / n_sec)
            center_boxes.append(
                f'    <geom name="env_s1_center_deck_{k}" type="box" '
                f'pos="0 0 -{DECK_HALF_THICK}" euler="0 0 {yaw_c:.3f}" '
                f'size="{r_in + 0.01:.5f} {hy_in:.5f} {DECK_HALF_THICK}" '
                f'rgba="0.74 0.56 0.36 1" friction="1.05 0.010 0.003" {env_ct}/>'
            )

        slope_slabs: List[str] = []
        for k in range(n_sec):
            yaw = k * (360.0 / n_sec)
            rad = math.radians(yaw)
            x0, y0 = r_in * math.cos(rad), r_in * math.sin(rad)
            c_xyz, qstr, hx = _slope_quat_and_center(x0, y0, 0.0, horiz_run, yaw, slope_deg, slab_hz)
            col = "0.46 0.24 0.24 1" if k % 2 == 0 else "0.54 0.28 0.26 1"
            slope_slabs.append(
                f'    <geom name="env_s1_slope_{k}" type="box" '
                f'pos="{c_xyz[0]:.5f} {c_xyz[1]:.5f} {c_xyz[2]:.5f}" quat="{qstr}" '
                f'size="{hx:.5f} {hy_tang:.5f} {slab_hz:.4f}" rgba="{col}" friction="0.95 0.010 0.003" {env_ct}/>'
            )

        world_xml = (
            common_floor
            + f"""
{chr(10).join(center_boxes)}
    <geom name="env_s1_pedestal" type="cylinder" pos="0 0 -1.18" size="1.45 0.68" rgba="0.15 0.18 0.23 1" {vis_ct}/>
{chr(10).join(slope_slabs)}
{chr(10).join(spring_bodies)}
    """
        )
        return world_xml, "\n".join(spring_acts), act_names

    # ========================================================================
    # STAGE 2: CAMELBACK RIDGE (双峰飞坡·弹力竞技场)
    # ========================================================================
    elif stage_key == "camelback_ridge":
        ridge_h = 0.24
        ramp_run = 1.23
        slope_deg = math.degrees(math.atan2(ridge_h, ramp_run))
        slab_hz = 0.22
        half_y = 2.95
        ridge_hy = half_y - 0.003

        cw_xyz, qw_str, hxw = _slope_quat_and_center(-1.75, 0.0, 0.0, ramp_run, 0.0, slope_deg, slab_hz)
        ce_xyz, qe_str, hxe = _slope_quat_and_center(1.75, 0.0, 0.0, ramp_run, 180.0, slope_deg, slab_hz)

        world_xml = (
            common_floor
            + f"""
    <geom name="env_s2_main_deck" type="box" pos="0 0 -{DECK_HALF_THICK}" size="3.60 {half_y} {DECK_HALF_THICK}" rgba="0.72 0.55 0.35 1" friction="1.05 0.010 0.003" {env_ct}/>
    <geom name="env_s2_west_pillar" type="cylinder" pos="-2.30 0 -1.18" size="1.05 0.68" rgba="0.15 0.18 0.23 1" {vis_ct}/>
    <geom name="env_s2_east_pillar" type="cylinder" pos="2.30 0 -1.18" size="1.05 0.68" rgba="0.15 0.18 0.23 1" {vis_ct}/>

    <geom name="env_s2_slope_west" type="box" pos="{cw_xyz[0]:.5f} {cw_xyz[1]:.5f} {cw_xyz[2]:.5f}" quat="{qw_str}" size="{hxw:.5f} {ridge_hy:.4f} {slab_hz:.4f}" rgba="0.30 0.39 0.52 1" friction="1.05 0.010 0.003" {env_ct}/>
    <geom name="env_s2_slope_east" type="box" pos="{ce_xyz[0]:.5f} {ce_xyz[1]:.5f} {ce_xyz[2]:.5f}" quat="{qe_str}" size="{hxe:.5f} {ridge_hy:.4f} {slab_hz:.4f}" rgba="0.30 0.39 0.52 1" friction="1.05 0.010 0.003" {env_ct}/>
    <geom name="env_s2_ridge_crest" type="box" pos="0 0 {ridge_h - slab_hz:.5f}" size="0.525 {ridge_hy:.4f} {slab_hz:.4f}" rgba="0.24 0.32 0.44 1" friction="1.05 0.010 0.003" {env_ct}/>

    <geom name="env_s2_trim_w" type="box" pos="-3.56 0 0.004" size="0.035 {half_y:.3f} 0.005" rgba="0.92 0.26 0.20 1" {vis_ct}/>
    <geom name="env_s2_trim_e" type="box" pos="3.56 0 0.004" size="0.035 {half_y:.3f} 0.005" rgba="0.92 0.26 0.20 1" {vis_ct}/>

{chr(10).join(spring_bodies)}
    """
        )
        return world_xml, "\n".join(spring_acts), act_names

    # ========================================================================
    # STAGE 3: PORTAL ARENA (双生折跃·传送门擂台)
    # ========================================================================
    else:
        core_hx = 2.60
        apron_run = 1.05  # x in [-3.65, -2.60] and [+2.60, +3.65]
        half_y = 2.85
        apron_hy = half_y - 0.003
        apron_deg = -8.0
        slab_hz = 0.22

        cw_xyz, qw_str, hxw = _slope_quat_and_center(-core_hx, 0.0, 0.0, apron_run, 180.0, apron_deg, slab_hz)
        ce_xyz, qe_str, hxe = _slope_quat_and_center(core_hx, 0.0, 0.0, apron_run, 0.0, apron_deg, slab_hz)

        world_xml = (
            common_floor
            + f"""
    <!-- Flat 5.20m x 5.70m Quantum Core Deck + Smooth -8.0-deg West & East Danger Cliff Aprons -->
    <geom name="env_s3_core_deck" type="box" pos="0 0 -{DECK_HALF_THICK}" size="{core_hx:.3f} {half_y:.3f} {DECK_HALF_THICK}" rgba="0.20 0.26 0.36 1" friction="1.05 0.010 0.003" {env_ct}/>
    <geom name="env_s3_pedestal" type="cylinder" pos="0 0 -1.18" size="1.65 0.68" rgba="0.13 0.16 0.22 1" {vis_ct}/>

    <geom name="env_s3_apron_west" type="box" pos="{cw_xyz[0]:.5f} {cw_xyz[1]:.5f} {cw_xyz[2]:.5f}" quat="{qw_str}" size="{hxw:.5f} {apron_hy:.4f} {slab_hz:.4f}" rgba="0.46 0.22 0.28 1" friction="0.94 0.010 0.003" {env_ct}/>
    <geom name="env_s3_apron_east" type="box" pos="{ce_xyz[0]:.5f} {ce_xyz[1]:.5f} {ce_xyz[2]:.5f}" quat="{qe_str}" size="{hxw:.5f} {apron_hy:.4f} {slab_hz:.4f}" rgba="0.46 0.22 0.28 1" friction="0.94 0.010 0.003" {env_ct}/>

    <!-- North/South Cliff Rim Warning Strips & Center Divider Line -->
    <geom name="env_s3_trim_n" type="box" pos="0 {half_y - 0.035:.3f} 0.004" size="{core_hx:.3f} 0.032 0.004" rgba="0.22 0.82 0.98 1" {vis_ct}/>
    <geom name="env_s3_trim_s" type="box" pos="0 -{half_y - 0.035:.3f} 0.004" size="{core_hx:.3f} 0.032 0.004" rgba="0.22 0.82 0.98 1" {vis_ct}/>
    <geom name="env_s3_center_line" type="box" pos="0 0 0.003" size="0.022 {half_y - 0.12:.3f} 0.002" rgba="0.30 0.40 0.56 1" {vis_ct}/>

    <!-- 2 Linked Bidirectional Quantum Portal Pairs (Cyan Flank Gates & Magenta Cliff-Slingshot Gates) -->
{chr(10).join(portal_bodies)}
    """
        )
        return world_xml, "", []


def create_hazard_runtime(stage_key: str) -> Dict[str, Any]:
    """Initialize per-spring and per-portal cooldown/runtime state for a match."""
    state: Dict[str, Any] = {
        "_portal_car_cd": {"A": 0.0, "B": 0.0},
        "_portals": {},
    }
    for sp in STAGE_SPRINGS[stage_key]:
        state[sp["tag"]] = {
            "ready_t": float(sp["arm_t"]),
            "fire_start_t": -9.0,
            "fire_until_t": -9.0,
        }
    for pt in STAGE_PORTALS[stage_key]:
        state["_portals"][pt["tag"]] = {
            "ready_t": float(pt["arm_t"]),
        }
    return state


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
        # Vertical half-extent in world Z (must be touching/skimming floor height z ≈ sz, not high airborne)
        wz = abs(float(gR[2, 0])) * hx + abs(float(gR[2, 1])) * hy + abs(float(gR[2, 2])) * hz
        z_bot = float(gp[2]) - wz
        z_top = float(gp[2]) + wz
        if z_bot > sz + 0.085 or z_top < sz - 0.06:
            continue

        # Exact projected half-extents along the plate's local X (ux) and local Y (uy) axes
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

        # Plate physical rectangle is [-0.48, +0.48] x [-0.44, +0.44] (+0.48 is fixed ground hinge axle).
        # Trigger immediately when at least ~1.5cm of the tire/wedge steps onto [-0.47, +0.41] x [-0.43, +0.43]!
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
    Update tactical hinged diagonal catapult plates deterministically.
    Edge-Sensitive Trigger & Anti-Pinch Guarantee:
      - Triggers immediately as soon as ANY ground-level tire, wedge, or hull geom steps onto any edge of the plate!
      - Solid pad collision (`contype=1, conaffinity=6`) is ONLY active during the 0.22s upward catapult stroke!
      - The instant the plate begins retracting back into the floor (`t_sec >= fire_until_t`), pad collision
        switches to `0` (`contype=0, conaffinity=0`) while the continuous floor slab underneath supports cars,
        so a retracting plate NEVER pinches, clamps, or traps any wheel/wedge/chassis!
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
                bx_loc = dbx * cp + dby * sp_sin
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


def step_arena_portals(
    stage_key: str,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    t_sec: float,
    body_a: int,
    body_b: int,
    hazard_state: Dict[str, Any],
) -> List[Tuple[str, str, str]]:
    """
    Check if Car A or Car B enters an active Quantum Portal on `portal_arena`.
    When triggered:
      - Teleports the entire articulated vehicle (`<freejoint>` qpos & qvel) cleanly out of `dest_tag`
      - Rotates vehicle heading & redirects horizontal momentum along `dest_exit_yaw` with a warp surge
      - Applies anti-overlap clearance if the other vehicle is near the exit gate!
    Returns list of `(who, entry_portal_tag, dest_portal_tag)` warp events on this step.
    """
    portals = STAGE_PORTALS.get(stage_key, [])
    if not portals:
        return []

    p_states = hazard_state.setdefault("_portals", {})
    car_cd = hazard_state.setdefault("_portal_car_cd", {"A": 0.0, "B": 0.0})
    events: List[Tuple[str, str, str]] = []

    for who, bid, other_bid in (("A", body_a, body_b), ("B", body_b, body_a)):
        if t_sec < float(car_cd.get(who, 0.0)):
            continue
        pos = data.xpos[bid]
        cx, cy, cz = float(pos[0]), float(pos[1]), float(pos[2])

        for pt in portals:
            tag = pt["tag"]
            pst = p_states.get(tag)
            if pst is None or t_sec < float(pst["ready_t"]):
                continue
            px, py, pz = float(pt["x"]), float(pt["y"]), float(pt["z"])
            if math.hypot(cx - px, cy - py) > PORTAL_TRIGGER_R or not (-0.22 <= (cz - pz) <= 0.56):
                continue

            # Trigger Warp!
            j_id = int(model.body_jntadr[bid])
            qadr = int(model.jnt_qposadr[j_id])
            vadr = int(model.jnt_dofadr[j_id])

            out_yaw_rad = math.radians(float(pt["dest_exit_yaw"]))
            ux, uy = math.cos(out_yaw_rad), math.sin(out_yaw_rad)

            dest_x = float(pt["dest_x"]) + 0.54 * ux
            dest_y = float(pt["dest_y"]) + 0.54 * uy
            dest_z = max(float(pt["dest_z"]) + 0.22, cz + 0.03)

            # Anti-overlap check against the other vehicle near the exit gate
            opos = data.xpos[other_bid]
            ox, oy = float(opos[0]), float(opos[1])
            if math.hypot(dest_x - ox, dest_y - oy) < 0.92:
                # Try sliding further forward along exit vector or laterally
                cand1 = (dest_x + 0.65 * ux, dest_y + 0.65 * uy)
                cand2 = (dest_x - 0.55 * uy, dest_y + 0.55 * ux)
                if math.hypot(cand1[0] - ox, cand1[1] - oy) >= math.hypot(cand2[0] - ox, cand2[1] - oy):
                    dest_x, dest_y = cand1
                else:
                    dest_x, dest_y = cand2

            # Rotate vehicle yaw to face `dest_exit_yaw` while preserving roll/pitch attitude
            R_cur = data.xmat[bid].reshape(3, 3)
            cur_yaw = math.atan2(float(R_cur[1, 0]), float(R_cur[0, 0]))
            d_yaw = out_yaw_rad - cur_yaw
            qz = np.array([math.cos(0.5 * d_yaw), 0.0, 0.0, math.sin(0.5 * d_yaw)], dtype=np.float64)
            q_old = data.qpos[qadr + 3 : qadr + 7].copy()
            q_new = np.zeros(4, dtype=np.float64)
            mujoco.mju_mulQuat(q_new, qz, q_old)

            vx_old, vy_old, vz_old = float(data.qvel[vadr]), float(data.qvel[vadr + 1]), float(data.qvel[vadr + 2])
            spd_in = math.hypot(vx_old, vy_old)
            spd_out = max(2.65, min(5.20, spd_in * 1.15 + 0.65))

            data.qpos[qadr] = dest_x
            data.qpos[qadr + 1] = dest_y
            data.qpos[qadr + 2] = dest_z
            data.qpos[qadr + 3 : qadr + 7] = q_new

            data.qvel[vadr] = spd_out * ux
            data.qvel[vadr + 1] = spd_out * uy
            data.qvel[vadr + 2] = max(0.30, min(1.80, vz_old + 0.25))
            data.qvel[vadr + 3 : vadr + 6] *= 0.45

            pst["ready_t"] = t_sec + float(pt["cooldown"])
            dest_st = p_states.get(pt["dest_tag"])
            if dest_st is not None:
                dest_st["ready_t"] = max(float(dest_st["ready_t"]), t_sec + 0.35)
            car_cd[who] = t_sec + 0.90

            mujoco.mj_forward(model, data)
            events.append((who, tag, pt["dest_tag"]))
            break

    return events


def try_warp_bolt(stage_key: str, bolt: Dict[str, Any]) -> Optional[Tuple[List[float], List[float]]]:
    """
    If an Energy Cannon bolt enters a Quantum Portal (and hasn't warped yet), teleport it out of
    the linked destination portal along `dest_exit_yaw` at full speed!
    Returns `(entry_xy, exit_xy)` if warped, else `None`.
    """
    if bolt.get("warped"):
        return None
    portals = STAGE_PORTALS.get(stage_key, [])
    if not portals:
        return None
    bx, by, bz = float(bolt["pos"][0]), float(bolt["pos"][1]), float(bolt["pos"][2])
    for pt in portals:
        px, py, pz = float(pt["x"]), float(pt["y"]), float(pt["z"])
        if math.hypot(bx - px, by - py) <= PORTAL_TRIGGER_R and (-0.10 <= (bz - pz) <= 0.62):
            out_rad = math.radians(float(pt["dest_exit_yaw"]))
            ux, uy = math.cos(out_rad), math.sin(out_rad)
            spd = max(12.0, math.hypot(float(bolt["vel"][0]), float(bolt["vel"][1])))
            nx = float(pt["dest_x"]) + 0.50 * ux
            ny = float(pt["dest_y"]) + 0.50 * uy
            nz = max(float(pt["dest_z"]) + 0.18, bz)
            entry_xy = [bx, by]
            exit_xy = [nx, ny]
            bolt["pos"] = np.array([nx, ny, nz], dtype=np.float64)
            bolt["vel"] = np.array([spd * ux, spd * uy, 0.0], dtype=np.float64)
            bolt["dir"] = np.array([ux, uy, 0.0], dtype=np.float64)
            bolt["warped"] = True
            return (entry_xy, exit_xy)
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
) -> Dict[str, Any]:
    """
    Compute raw physical arena, hinged-catapult, and quantum-portal geometry for a car at world (x, y).
    """
    if stage_key == "spring_arena":
        r = math.hypot(x, y)
        edge_dist = 3.45 - r
        edge_dir = (x / r, y / r) if r > 1e-4 else (1.0, 0.0)
        safe_pt = (0.0, 0.0)

    elif stage_key == "camelback_ridge":
        mx_w = x - (-3.60)
        mx_e = 3.60 - x
        my_s = y - (-2.95)
        my_n = 2.95 - y
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

    else:  # portal_arena (7.30m x 5.70m: x in [-3.65, +3.65], y in [-2.85, +2.85])
        mx_w = x - (-3.65)
        mx_e = 3.65 - x
        my_s = y - (-2.85)
        my_n = 2.85 - y
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

    # 1. Nearest Hinged Catapult (`spring_*`, 9.9 if stage has no catapults)
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

    # 2. Nearest Quantum Portal (`portal_*`, 9.9 if stage has no portals)
    portals = STAGE_PORTALS.get(stage_key, [])
    p_states = (hazard_state or {}).get("_portals", {})
    pt_dist = 9.9
    pt_xy = (0.0, 0.0)
    pt_out_xy = (0.0, 0.0)
    pt_out_yaw = 0.0
    pt_dir = (1.0, 0.0)
    pt_ready = False
    for pt in portals:
        px, py = float(pt["x"]), float(pt["y"])
        d = math.hypot(px - x, py - y)
        if d < pt_dist:
            pt_dist = d
            pt_xy = (px, py)
            pt_out_xy = (float(pt["dest_x"]), float(pt["dest_y"]))
            pt_out_yaw = float(pt["dest_exit_yaw"])
            if d > 1e-5:
                pt_dir = ((px - x) / d, (py - y) / d)
            if pt["tag"] in p_states:
                pt_ready = (t_sec >= float(p_states[pt["tag"]]["ready_t"]))
            else:
                pt_ready = True

    ept_xy = pt_xy
    ept_out_yaw = pt_out_yaw
    ept_dist = 9.9
    ept_ready = False
    if enemy_xy is not None and portals:
        ex, ey = enemy_xy
        for pt in portals:
            px, py = float(pt["x"]), float(pt["y"])
            ed = math.hypot(px - ex, py - ey)
            if ed < ept_dist:
                ept_dist = ed
                ept_xy = (px, py)
                ept_out_yaw = float(pt["dest_exit_yaw"])
                if pt["tag"] in p_states:
                    ept_ready = (t_sec >= float(p_states[pt["tag"]]["ready_t"]))
                else:
                    ept_ready = True

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
        "portal_x": pt_xy[0],
        "portal_y": pt_xy[1],
        "portal_dist": pt_dist,
        "portal_dir": pt_dir,
        "portal_out_x": pt_out_xy[0],
        "portal_out_y": pt_out_xy[1],
        "portal_out_yaw_deg": pt_out_yaw,
        "portal_ready": pt_ready,
        "enemy_portal_x": ept_xy[0],
        "enemy_portal_y": ept_xy[1],
        "enemy_portal_dist": ept_dist,
        "enemy_portal_out_yaw_deg": ept_out_yaw,
        "enemy_portal_ready": ept_ready,
    }
