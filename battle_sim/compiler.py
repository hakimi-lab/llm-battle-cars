"""
MuJoCo 3D Procedural Vehicle & Match Scene Compiler for LLM Battle-Cars (v3.0).

Key v3.0 Upgrades:
  - Removed `energy_shield` completely so ranged combat is fast, active, and dodge-driven
  - Removed ALL upper Length x Width x Height bounding-box & part dimension limits (governed naturally by 3D volume x density mass & inertia!)
  - Realistic electric drivetrain gearing (`speed` / `balanced` / `torque`) + shared onboard motor power budget + grounded tire load drag
  - Reports real MuJoCo flat-sprint `est_top_speed_mps` and `weight_class` during 0-credit `validate`
"""
from __future__ import annotations

import math
import xml.sax.saxutils as saxutils
from typing import Any, Dict, List, Optional, Tuple

import mujoco
import numpy as np

from .arenas import ARENA_META, build_arena_mjcf, resolve_stage_key
from .catalog import (
    CANNON_MODES,
    CHASSIS_MIN_DIMS,
    MOUNT_ANCHORS,
    PARTS_CATALOG,
    PHYSICS_DT,
    clamp,
    parse_hex_color,
    validate_car_dict,
)

BOLTS_PER_CANNON = 3

# Tuned physical tire deformation rolling resistance coefficients under vertical mass load
LOAD_DRAG_BASE_KG = 12.0
LOAD_DRAG_EXP = 1.18
LOAD_ROLL_COEF = 0.88
LOAD_VISC_COEF = 0.85


def apply_grounded_mass_drag(
    data: mujoco.MjData,
    root_body_id: int,
    vehicle_mass_kg: float,
    is_grounded: bool,
) -> None:
    """
    Apply realistic grounded tire rolling resistance proportional to vehicle mass load.
    Airborne vehicles (`is_grounded=False`) experience zero ground rolling drag, preserving pure ballistic arcs!
    """
    data.xfrc_applied[root_body_id, :] = 0.0
    if not is_grounded:
        return

    vel6 = data.cvel[root_body_id]
    vx, vy = float(vel6[3]), float(vel6[4])
    spd = math.hypot(vx, vy)

    excess = max(0.0, float(vehicle_mass_kg) - LOAD_DRAG_BASE_KG)
    if excess <= 1e-4 or spd <= 1e-4:
        return

    load_factor = excess ** LOAD_DRAG_EXP
    f_mag = load_factor * (LOAD_ROLL_COEF * math.tanh(3.8 * spd) + LOAD_VISC_COEF * spd)
    data.xfrc_applied[root_body_id, 0] = -f_mag * (vx / spd)
    data.xfrc_applied[root_body_id, 1] = -f_mag * (vy / spd)


def classify_weight_class(mass_kg: float) -> str:
    if mass_kg < 28.0:
        return "lightweight"
    if mass_kg < 44.0:
        return "middleweight"
    if mass_kg < 66.0:
        return "heavyweight"
    return "super_heavy"


def _euler_deg_to_mat(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, sx = math.cos(math.radians(rx)), math.sin(math.radians(rx))
    cy, sy = math.cos(math.radians(ry)), math.sin(math.radians(ry))
    cz, sz = math.cos(math.radians(rz)), math.sin(math.radians(rz))
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx


def _mat_to_quat_str(R: np.ndarray) -> str:
    q = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(q, R.reshape(-1).astype(np.float64))
    return f"{q[0]:.5f} {q[1]:.5f} {q[2]:.5f} {q[3]:.5f}"


def _vec_str(v: Tuple[float, float, float] | List[float] | np.ndarray) -> str:
    return f"{float(v[0]):.5f} {float(v[1]):.5f} {float(v[2]):.5f}"


def _rgba_str(rgb: Tuple[float, float, float], alpha: float = 1.0) -> str:
    return f"{rgb[0]:.4f} {rgb[1]:.4f} {rgb[2]:.4f} {alpha:.4f}"


def _resolve_mount_pose(
    part: Dict[str, Any],
    parent_half: np.ndarray,
    ptype: str,
    params: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute (pos_in_parent [3], R_in_parent [3,3]) from either:
      - 'mount' (+ optional 'offset' and 'euler'), OR
      - explicit 'pos' (+ optional 'euler')
    """
    mount_name = part.get("mount", None)
    offset = np.array(part.get("offset", [0.0, 0.0, 0.0]), dtype=np.float64)
    user_euler = part.get("euler", [0.0, 0.0, 0.0])
    R_user = _euler_deg_to_mat(float(user_euler[0]), float(user_euler[1]), float(user_euler[2]))

    if "pos" in part and part["pos"] is not None and mount_name is None:
        pos = np.array(part["pos"], dtype=np.float64) + offset
        return pos, R_user

    mkey = str(mount_name or "front").strip()
    norm_mul, auto_yaw = MOUNT_ANCHORS.get(mkey, ((1.0, 0.0, 0.0), 0.0))
    anchor = np.array(norm_mul, dtype=np.float64) * parent_half

    if ptype in ("wheel", "monster_wheel"):
        w = max(0.03, float(params.get("width", 0.06 if ptype == "wheel" else 0.105)))
        if abs(norm_mul[1]) > 0.5:
            anchor[1] += math.copysign(w * 0.62 + 0.012, norm_mul[1])
        auto_yaw = 0.0
    elif ptype == "frame":
        fsize = params.get("size", [0.36, 0.10, 0.08])
        hl = max(0.06, float(fsize[0])) * 0.5
        rad_yaw = math.radians(auto_yaw)
        if abs(norm_mul[0]) > 0.5 or abs(norm_mul[1]) > 0.5:
            anchor[0] += math.cos(rad_yaw) * hl * 0.85
            anchor[1] += math.sin(rad_yaw) * hl * 0.85
        elif norm_mul[2] > 0.5:
            anchor[2] += max(0.04, float(fsize[2])) * 0.5
    elif ptype == "wedge":
        wlen = max(0.10, float(params.get("length", 0.28)))
        rad_yaw = math.radians(auto_yaw)
        anchor[0] += math.cos(rad_yaw) * (wlen * 0.48)
        anchor[1] += math.sin(rad_yaw) * (wlen * 0.48)
        anchor[2] = -parent_half[2] * 0.45
    elif ptype == "armor_shield":
        thick = clamp(float(params.get("thickness", 0.045)), 0.02, 0.14)
        rad_yaw = math.radians(auto_yaw)
        anchor[0] += math.cos(rad_yaw) * (thick * 0.6 + 0.01)
        anchor[1] += math.sin(rad_yaw) * (thick * 0.6 + 0.01)
    elif ptype == "gyro_stabilizer":
        if mkey == "front" and mount_name is None:
            anchor = np.array([0.0, 0.0, -parent_half[2] - 0.022], dtype=np.float64)
        elif norm_mul[2] < -0.5:
            anchor[2] -= 0.022
    elif ptype == "spinner_horiz":
        if norm_mul[2] > 0.5:
            anchor[2] += 0.045
        else:
            rad_yaw = math.radians(auto_yaw)
            anchor[0] += math.cos(rad_yaw) * 0.12
            anchor[1] += math.sin(rad_yaw) * 0.12
            anchor[2] += 0.03
    elif ptype == "spinner_vert":
        rad = clamp(float(params.get("radius", 0.24)), 0.14, 0.42)
        rad_yaw = math.radians(auto_yaw)
        anchor[0] += math.cos(rad_yaw) * (rad * 0.68)
        anchor[1] += math.sin(rad_yaw) * (rad * 0.68)
    elif ptype == "flipper":
        rad_yaw = math.radians(auto_yaw)
        anchor[0] += math.cos(rad_yaw) * 0.04
        anchor[1] += math.sin(rad_yaw) * 0.04
        anchor[2] = -parent_half[2] * 0.55
    elif ptype == "hammer":
        if norm_mul[2] <= 0.2:
            anchor[2] = parent_half[2] + 0.03
        auto_yaw = 0.0
    elif ptype == "piston_punch":
        rad_yaw = math.radians(auto_yaw)
        anchor[0] += math.cos(rad_yaw) * 0.05
        anchor[1] += math.sin(rad_yaw) * 0.05
    elif ptype == "energy_cannon":
        if norm_mul[2] > 0.5:
            anchor[2] += 0.055
        else:
            rad_yaw = math.radians(auto_yaw)
            anchor[0] += math.cos(rad_yaw) * 0.06
            anchor[1] += math.sin(rad_yaw) * 0.06
            anchor[2] += 0.035
    elif ptype == "thruster":
        if mount_name is None:
            anchor = np.array([-parent_half[0] - 0.05, 0.0, 0.02], dtype=np.float64)
        auto_yaw = 0.0

    R_auto = _euler_deg_to_mat(0.0, 0.0, auto_yaw)
    return anchor + offset, (R_auto @ R_user)


def build_vehicle_xml(
    spec: Dict[str, Any],
    prefix: str,
    spawn_xy: Tuple[float, float] = (0.0, 0.0),
    spawn_yaw_deg: float = 0.0,
    contype: int = 2,
    conaffinity: int = 5,
    spawn_z_override: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Compile one vehicle into MJCF body + actuators + weapon metadata.
    v3.0: Unrestricted Length x Width x Height + realistic electric drivetrain power sharing!
    """
    meta = spec.get("meta", {})
    default_primary = (0.22, 0.52, 0.88) if prefix == "A" else (0.88, 0.26, 0.20)
    primary_rgb = parse_hex_color(meta.get("primary_color"), default_primary)
    secondary_rgb = parse_hex_color(meta.get("secondary_color"), (0.82, 0.62, 0.20))
    accent_rgb = parse_hex_color(meta.get("accent_color"), (0.14, 0.84, 0.78))
    chassis_mat_style = str(meta.get("chassis_finish", "glossy")).strip()
    if chassis_mat_style not in ("glossy", "matte", "metallic", "neon", "wood", "glass"):
        chassis_mat_style = "glossy"

    chassis = spec.get("chassis", {})
    raw_csize = chassis.get("size", [0.52, 0.38, 0.18])
    ch_hx = max(CHASSIS_MIN_DIMS[0], float(raw_csize[0])) * 0.5
    ch_hy = max(CHASSIS_MIN_DIMS[1], float(raw_csize[1])) * 0.5
    ch_hz = max(CHASSIS_MIN_DIMS[2], float(raw_csize[2])) * 0.5
    chassis_half = np.array([ch_hx, ch_hy, ch_hz], dtype=np.float64)
    chassis_density = clamp(float(chassis.get("density", 260.0)), 130.0, 520.0)

    parts = spec.get("parts", [])
    decos = spec.get("decorations", [])

    # Count motorized wheels so onboard battery power is shared fairly across 2/4/6/8+ wheel layouts
    n_motor_wheels = 0
    for p in parts:
        if isinstance(p, dict) and str(p.get("type", "")).strip() in ("wheel", "monster_wheel"):
            prm = p.get("params") or {}
            ch_val = str(prm.get("channel", "auto")).strip().lower() if isinstance(prm, dict) else "auto"
            if ch_val != "free":
                n_motor_wheels += 1
    wheel_power_scale = min(1.08, (4.0 / max(1, n_motor_wheels)) ** 0.78)

    decos_by_parent: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {"chassis": []}
    for p in parts:
        decos_by_parent[str(p.get("id", ""))] = []
    for di, d in enumerate(decos):
        dp = str(d.get("parent", "chassis")).strip()
        decos_by_parent.setdefault(dp, []).append((di, d))

    children_by_parent: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {"chassis": []}
    part_half_extents: Dict[str, np.ndarray] = {"chassis": chassis_half}
    for i, p in enumerate(parts):
        pid = str(p["id"]).strip()
        ptype = str(p["type"]).strip()
        if ptype == "frame":
            params = p.get("params", {}) or {}
            fsz = params.get("size", PARTS_CATALOG["frame"]["defaults"]["size"])
            f_half = np.array([
                max(0.06, float(fsz[0])) * 0.5,
                max(0.04, float(fsz[1])) * 0.5,
                max(0.04, float(fsz[2])) * 0.5,
            ], dtype=np.float64)
            part_half_extents[pid] = f_half
            children_by_parent[pid] = []
        parent_id = str(p.get("parent", "chassis")).strip()
        if parent_id not in children_by_parent:
            parent_id = "chassis"
        children_by_parent[parent_id].append((i, p))

    ct_attr = f'contype="{contype}" conaffinity="{conaffinity}"'
    vis_attr = 'contype="0" conaffinity="0"'

    geom_styles: Dict[str, Dict[str, Any]] = {}
    actuators_xml: List[str] = []
    actuators_meta: List[Dict[str, Any]] = []
    thrusters_meta: List[Dict[str, Any]] = []
    cannons_meta: List[Dict[str, Any]] = []
    mocap_bolts_xml: List[str] = []

    lowest_z = -ch_hz

    def register_style(gname: str, role: str, rgb: Tuple[float, float, float], material: str = "glossy", extra: Optional[Dict[str, Any]] = None) -> None:
        info = {
            "owner": prefix,
            "role": role,
            "rgb": [round(rgb[0], 4), round(rgb[1], 4), round(rgb[2], 4)],
            "material": material,
        }
        if extra:
            info.update(extra)
        geom_styles[gname] = info

    def emit_decorations(parent_key: str, indent: str) -> str:
        lines: List[str] = []
        p_half = part_half_extents.get(parent_key, np.array([0.12, 0.12, 0.12]))
        for di, d in decos_by_parent.get(parent_key, []):
            shape = str(d.get("shape", "box")).strip()
            mat = str(d.get("material", "glossy")).strip()
            rgb = parse_hex_color(d.get("color"), accent_rgb)
            if "mount" in d and d.get("pos") is None:
                mkey = str(d["mount"]).strip()
                nm, ay = MOUNT_ANCHORS.get(mkey, ((0.0, 0.0, 1.0), 0.0))
                dpos = np.array(nm, dtype=np.float64) * p_half + np.array(d.get("offset", [0.0, 0.0, 0.0]), dtype=np.float64)
                deul = d.get("euler", [0.0, 0.0, 0.0])
                dR = _euler_deg_to_mat(0.0, 0.0, ay) @ _euler_deg_to_mat(float(deul[0]), float(deul[1]), float(deul[2]))
            else:
                dpos = np.array(d.get("pos", [0.0, 0.0, p_half[2] + 0.05]), dtype=np.float64)
                deul = d.get("euler", [0.0, 0.0, 0.0])
                dR = _euler_deg_to_mat(float(deul[0]), float(deul[1]), float(deul[2]))

            raw_sz = d.get("size", [0.14, 0.08, 0.08])
            if isinstance(raw_sz, (int, float)):
                raw_sz = [float(raw_sz)] * 3
            sx = max(0.015, float(raw_sz[0]))
            sy = max(0.015, float(raw_sz[1]) if len(raw_sz) > 1 else sx)
            sz = max(0.015, float(raw_sz[2]) if len(raw_sz) > 2 else sx)
            gname = f"{prefix}_deco_{di}_{shape}"
            qstr = _mat_to_quat_str(dR)

            if shape == "sphere":
                lines.append(
                    f'{indent}<geom name="{gname}" type="ellipsoid" pos="{_vec_str(dpos)}" quat="{qstr}" '
                    f'size="{sx * 0.5:.4f} {sy * 0.5:.4f} {sz * 0.5:.4f}" mass="0.0005" rgba="{_rgba_str(rgb)}" {vis_attr}/>'
                )
                register_style(gname, "deco", rgb, mat, {"deco_shape": "sphere"})
            elif shape == "eye":
                rad = max(0.018, sx * 0.5)
                lines.append(
                    f'{indent}<geom name="{gname}" type="sphere" pos="{_vec_str(dpos)}" quat="{qstr}" '
                    f'size="{rad:.4f}" mass="0.0005" rgba="0.98 0.98 0.99 1" {vis_attr}/>'
                )
                register_style(gname, "eye", rgb, "glossy", {"deco_shape": "eye", "iris_rgb": list(rgb)})
            elif shape in ("cylinder", "cone", "ring"):
                rad = sx * 0.5
                hh = max(0.008, sz * 0.5)
                lines.append(
                    f'{indent}<geom name="{gname}" type="cylinder" pos="{_vec_str(dpos)}" quat="{qstr}" '
                    f'size="{rad:.4f} {hh:.4f}" mass="0.0005" rgba="{_rgba_str(rgb)}" {vis_attr}/>'
                )
                register_style(gname, "deco", rgb, mat, {"deco_shape": shape})
            elif shape == "fin":
                lines.append(
                    f'{indent}<geom name="{gname}" type="box" pos="{_vec_str(dpos)}" quat="{qstr}" '
                    f'size="{sx * 0.5:.4f} {min(sy * 0.5, 0.030):.4f} {sz * 0.5:.4f}" mass="0.0005" rgba="{_rgba_str(rgb)}" {vis_attr}/>'
                )
                register_style(gname, "deco", rgb, mat, {"deco_shape": "fin"})
            else:
                lines.append(
                    f'{indent}<geom name="{gname}" type="box" pos="{_vec_str(dpos)}" quat="{qstr}" '
                    f'size="{sx * 0.5:.4f} {sy * 0.5:.4f} {sz * 0.5:.4f}" mass="0.0005" rgba="{_rgba_str(rgb)}" {vis_attr}/>'
                )
                register_style(gname, "deco", rgb, mat, {"deco_shape": "box"})
        return "\n".join(lines)

    def emit_part_recursive(pidx: int, part: Dict[str, Any], parent_half: np.ndarray, parent_world_z: float, indent: str) -> str:
        nonlocal lowest_z
        pid = str(part["id"]).strip()
        ptype = str(part["type"]).strip()
        params = dict(PARTS_CATALOG[ptype]["defaults"])
        if isinstance(part.get("params"), dict):
            params.update(part["params"])

        p_mat = str(part.get("material", params.get("material", "metallic"))).strip()
        if p_mat not in ("glossy", "matte", "metallic", "neon", "wood", "glass"):
            p_mat = "metallic"

        pos, R = _resolve_mount_pose(part, parent_half, ptype, params)
        qstr = _mat_to_quat_str(R)
        bname = f"{prefix}_p{pidx}_{pid}"
        sub_decos = emit_decorations(pid, indent + "  ")
        deco_block = f"\n{sub_decos}" if sub_decos else ""

        if ptype == "frame":
            f_half = part_half_extents[pid]
            gname = f"{bname}_beam"
            lowest_z = min(lowest_z, parent_world_z + float(pos[2]) - float(f_half[2]))
            register_style(gname, "frame", parse_hex_color(part.get("color"), (0.52, 0.56, 0.62)), p_mat)
            child_blocks = [
                emit_part_recursive(ci, cp, f_half, parent_world_z + float(pos[2]), indent + "  ")
                for ci, cp in children_by_parent.get(pid, [])
            ]
            children_str = ("\n" + "\n".join(child_blocks)) if child_blocks else ""
            return (
                f'{indent}<body name="{bname}" pos="{_vec_str(pos)}" quat="{qstr}">\n'
                f'{indent}  <geom name="{gname}" type="box" size="{_vec_str(f_half)}" '
                f'density="240" friction="0.65 0.01 0.003" rgba="{_rgba_str(geom_styles[gname]["rgb"])}" {ct_attr}/>'
                f'{deco_block}{children_str}\n'
                f'{indent}</body>'
            )

        elif ptype in ("wheel", "monster_wheel"):
            is_monster = (ptype == "monster_wheel")
            rad = clamp(
                float(params.get("radius", 0.31 if is_monster else 0.18)),
                0.22 if is_monster else 0.09,
                0.52 if is_monster else 0.34,
            )
            wid = clamp(
                float(params.get("width", 0.105 if is_monster else 0.055)),
                0.06 if is_monster else 0.03,
                0.22 if is_monster else 0.14,
            )
            lowest_z = min(lowest_z, parent_world_z + float(pos[2]) - rad)

            jname = f"{bname}_axle"
            g_tire = f"{bname}_tire"
            g_hub = f"{bname}_hub"
            g_spoke = f"{bname}_spoke"
            tire_fric = "1.18 0.014 0.004" if is_monster else "1.06 0.010 0.003"
            tire_dens = "360" if is_monster else "410"
            hub_rgb = parse_hex_color(part.get("color"), primary_rgb)
            register_style(g_tire, "tire", (0.08, 0.085, 0.095), "matte", {"is_monster": is_monster})
            register_style(g_hub, "wheel_hub", hub_rgb, "glossy")
            register_style(g_spoke, "wheel_spoke", secondary_rgb, "metallic")

            chan = str(params.get("channel", "auto")).strip().lower()
            if chan == "auto":
                chan = "left" if pos[1] > 0.025 else ("right" if pos[1] < -0.025 else "center")

            if chan != "free":
                gear_mode = str(params.get("gear", "torque" if is_monster else "balanced")).strip().lower()
                # True DC gearmotor curve:
                #   - 'speed': lower reduction -> higher free wheel RPM (fast on light cars!), bogs down if overloaded
                #   - 'balanced': balanced top speed & shoving force
                #   - 'torque': higher reduction -> strong shoving & pivoting force on heavy cars, lower free RPM
                if is_monster:
                    ref_r = 0.31
                    if gear_mode == "speed":
                        b_gear, b_damp = 30.0, 1.90
                    elif gear_mode == "balanced":
                        b_gear, b_damp = 41.0, 3.15
                    else:
                        b_gear, b_damp = 54.0, 4.95
                else:
                    ref_r = 0.18
                    if gear_mode == "speed":
                        b_gear, b_damp = 17.5, 0.65
                    elif gear_mode == "torque":
                        b_gear, b_damp = 32.5, 1.82
                    else:
                        b_gear, b_damp = 24.5, 1.12

                r_ratio = rad / ref_r
                gear_val = b_gear * wheel_power_scale * (r_ratio ** 0.75)
                damp_val = b_damp * wheel_power_scale * (r_ratio ** 1.75)
                aname = f"act_{bname}"
                actuators_xml.append(
                    f'    <motor name="{aname}" joint="{jname}" gear="{gear_val:.2f}" ctrllimited="true" ctrlrange="-1 1"/>'
                )
                actuators_meta.append({"name": aname, "kind": "wheel", "channel": chan, "part_id": pid, "gear_mode": gear_mode})
            else:
                damp_val = 0.08

            return (
                f'{indent}<body name="{bname}" pos="{_vec_str(pos)}" quat="{qstr}">\n'
                f'{indent}  <joint name="{jname}" type="hinge" axis="0 1 0" damping="{damp_val:.3f}" armature="0.028"/>\n'
                f'{indent}  <geom name="{g_tire}" type="cylinder" euler="90 0 0" size="{rad:.4f} {wid * 0.5:.4f}" '
                f'density="{tire_dens}" friction="{tire_fric}" rgba="0.09 0.095 0.11 1" {ct_attr}/>\n'
                f'{indent}  <geom name="{g_hub}" type="cylinder" euler="90 0 0" size="{rad * 0.52:.4f} {wid * 0.58:.4f}" '
                f'density="120" rgba="{_rgba_str(hub_rgb)}" {vis_attr}/>\n'
                f'{indent}  <geom name="{g_spoke}" type="box" size="{rad * 0.72:.4f} {wid * 0.61:.4f} {rad * 0.13:.4f}" '
                f'mass="0.01" rgba="{_rgba_str(secondary_rgb)}" {vis_attr}/>'
                f'{deco_block}\n'
                f'{indent}</body>'
            )

        elif ptype == "wedge":
            wid = max(0.12, float(params.get("width", 0.44)))
            wlen = max(0.12, float(params.get("length", 0.28)))
            slope = clamp(float(params.get("slope_deg", 26.0)), 14.0, 48.0)
            lowest_z = min(lowest_z, parent_world_z + float(pos[2]) - wlen * 0.42 * math.sin(math.radians(slope)) - 0.018)

            g_ramp = f"{bname}_wedge_ramp"
            g_sole = f"{bname}_wedge_sole"
            w_rgb = parse_hex_color(part.get("color"), (0.86, 0.62, 0.14))
            register_style(g_ramp, "wedge", w_rgb, "metallic")
            register_style(g_sole, "wedge_sole", (0.36, 0.40, 0.46), "metallic")
            return (
                f'{indent}<body name="{bname}" pos="{_vec_str(pos)}" quat="{qstr}">\n'
                f'{indent}  <geom name="{g_ramp}" type="box" pos="0 0 0.015" euler="0 {slope:.2f} 0" '
                f'size="{wlen * 0.52:.4f} {wid * 0.5:.4f} 0.018" density="380" friction="0.14 0.004 0.002" '
                f'rgba="{_rgba_str(w_rgb)}" {ct_attr}/>\n'
                f'{indent}  <geom name="{g_sole}" type="box" pos="{wlen * 0.18:.4f} 0 -0.025" '
                f'size="{wlen * 0.34:.4f} {wid * 0.48:.4f} 0.014" density="320" friction="0.14 0.004 0.002" '
                f'rgba="0.38 0.42 0.48 1" {ct_attr}/>'
                f'{deco_block}\n'
                f'{indent}</body>'
            )

        elif ptype == "armor_shield":
            wid = max(0.10, float(params.get("width", 0.42)))
            hgt = max(0.08, float(params.get("height", 0.22)))
            thk = clamp(float(params.get("thickness", 0.045)), 0.02, 0.14)
            tilt = clamp(float(params.get("tilt_deg", 10.0)), -40.0, 40.0)
            g_sh = f"{bname}_shield"
            s_rgb = parse_hex_color(part.get("color"), primary_rgb)
            register_style(g_sh, "armor", s_rgb, p_mat)
            return (
                f'{indent}<body name="{bname}" pos="{_vec_str(pos)}" quat="{qstr}">\n'
                f'{indent}  <geom name="{g_sh}" type="box" euler="0 {tilt:.2f} 0" '
                f'size="{thk * 0.5:.4f} {wid * 0.5:.4f} {hgt * 0.5:.4f}" density="340" '
                f'friction="0.30 0.006 0.002" rgba="{_rgba_str(s_rgb)}" {ct_attr}/>'
                f'{deco_block}\n'
                f'{indent}</body>'
            )

        elif ptype == "gyro_stabilizer":
            rad = clamp(float(params.get("radius", 0.16)), 0.08, 0.36)
            thk = clamp(float(params.get("thickness", 0.05)), 0.025, 0.12)
            g_gy = f"{bname}_gyro"
            register_style(g_gy, "brass", (0.86, 0.60, 0.14), "metallic")
            return (
                f'{indent}<body name="{bname}" pos="{_vec_str(pos)}" quat="{qstr}">\n'
                f'{indent}  <geom name="{g_gy}" type="cylinder" size="{rad:.4f} {thk * 0.5:.4f}" '
                f'density="1350" friction="0.22 0.005 0.002" rgba="0.86 0.60 0.14 1" {ct_attr}/>'
                f'{deco_block}\n'
                f'{indent}</body>'
            )

        elif ptype == "spinner_horiz":
            rad = clamp(float(params.get("radius", 0.38)), 0.20, 0.64)
            hoff = clamp(float(params.get("height_offset", 0.04)), -0.10, 0.24)
            sdir = 1.0 if float(params.get("spin_dir", 1)) >= 0 else -1.0
            jname = f"{bname}_spin_j"
            aname = f"act_{bname}"
            g_mot = f"{bname}_motor"
            g_bar = f"{bname}_blade"
            g_t1 = f"{bname}_tooth1"
            g_t2 = f"{bname}_tooth2"
            b_rgb = parse_hex_color(part.get("color"), accent_rgb)
            register_style(g_mot, "brass", (0.86, 0.58, 0.14), "metallic")
            register_style(g_bar, "steel", (0.44, 0.48, 0.54), "metallic")
            register_style(g_t1, "weapon_tip", b_rgb, "metallic")
            register_style(g_t2, "weapon_tip", b_rgb, "metallic")

            actuators_xml.append(
                f'    <motor name="{aname}" joint="{jname}" gear="44" ctrllimited="true" ctrlrange="-1 1"/>'
            )
            actuators_meta.append({"name": aname, "kind": "spinner", "spin_dir": sdir, "part_id": pid})
            return (
                f'{indent}<body name="{bname}" pos="{_vec_str(pos)}" quat="{qstr}">\n'
                f'{indent}  <geom name="{g_mot}" type="cylinder" pos="0 0 {hoff * 0.5:.4f}" size="0.075 0.045" density="340" rgba="0.86 0.58 0.14 1" {ct_attr}/>\n'
                f'{indent}  <body name="{bname}_rotor" pos="0 0 {hoff + 0.055:.4f}">\n'
                f'{indent}    <joint name="{jname}" type="hinge" axis="0 0 1" damping="0.16" armature="0.035"/>\n'
                f'{indent}    <geom name="{g_bar}" type="box" size="{rad:.4f} 0.042 0.020" density="560" friction="0.35 0.005 0.002" rgba="0.44 0.48 0.54 1" {ct_attr}/>\n'
                f'{indent}    <geom name="{g_t1}" type="box" pos="{rad * 0.92:.4f} 0.032 0" size="0.055 0.055 0.028" density="880" rgba="{_rgba_str(b_rgb)}" {ct_attr}/>\n'
                f'{indent}    <geom name="{g_t2}" type="box" pos="{-rad * 0.92:.4f} -0.032 0" size="0.055 0.055 0.028" density="880" rgba="{_rgba_str(b_rgb)}" {ct_attr}/>'
                f'{deco_block}\n'
                f'{indent}  </body>\n'
                f'{indent}</body>'
            )

        elif ptype == "spinner_vert":
            rad = clamp(float(params.get("radius", 0.24)), 0.14, 0.42)
            wid = clamp(float(params.get("width", 0.09)), 0.05, 0.20)
            sdir = 1.0 if float(params.get("spin_dir", 1)) >= 0 else -1.0
            lowest_z = min(lowest_z, parent_world_z + float(pos[2]) - rad + 0.02)
            jname = f"{bname}_vspin_j"
            aname = f"act_{bname}"
            g_drum = f"{bname}_vdrum"
            g_vt1 = f"{bname}_vtooth1"
            g_vt2 = f"{bname}_vtooth2"
            v_rgb = parse_hex_color(part.get("color"), secondary_rgb)
            register_style(g_drum, "steel", (0.38, 0.42, 0.48), "metallic")
            register_style(g_vt1, "weapon_tip", v_rgb, "metallic")
            register_style(g_vt2, "weapon_tip", v_rgb, "metallic")

            actuators_xml.append(
                f'    <motor name="{aname}" joint="{jname}" gear="48" ctrllimited="true" ctrlrange="-1 1"/>'
            )
            actuators_meta.append({"name": aname, "kind": "spinner", "spin_dir": sdir, "part_id": pid})
            return (
                f'{indent}<body name="{bname}" pos="{_vec_str(pos)}" quat="{qstr}">\n'
                f'{indent}  <joint name="{jname}" type="hinge" axis="0 -1 0" damping="0.18" armature="0.040"/>\n'
                f'{indent}  <geom name="{g_drum}" type="cylinder" euler="90 0 0" size="{rad * 0.80:.4f} {wid * 0.5:.4f}" density="520" rgba="0.38 0.42 0.48 1" {ct_attr}/>\n'
                f'{indent}  <geom name="{g_vt1}" type="box" pos="{rad * 0.78:.4f} 0 0" size="{rad * 0.28:.4f} {wid * 0.56:.4f} 0.042" density="920" friction="0.85 0.01 0.004" rgba="{_rgba_str(v_rgb)}" {ct_attr}/>\n'
                f'{indent}  <geom name="{g_vt2}" type="box" pos="{-rad * 0.78:.4f} 0 0" size="{rad * 0.28:.4f} {wid * 0.56:.4f} 0.042" density="920" friction="0.85 0.01 0.004" rgba="{_rgba_str(v_rgb)}" {ct_attr}/>'
                f'{deco_block}\n'
                f'{indent}</body>'
            )

        elif ptype == "flipper":
            flen = clamp(float(params.get("length", 0.40)), 0.22, 0.68)
            fwid = clamp(float(params.get("width", 0.28)), 0.14, 0.56)
            lowest_z = min(lowest_z, parent_world_z + float(pos[2]) - 0.035)
            jname = f"{bname}_flip_j"
            aname = f"act_{bname}"
            g_base = f"{bname}_flip_base"
            g_arm = f"{bname}_flip_plate"
            f_rgb = parse_hex_color(part.get("color"), accent_rgb)
            register_style(g_base, "brass", (0.85, 0.58, 0.14), "metallic")
            register_style(g_arm, "flipper", f_rgb, "metallic")

            actuators_xml.append(
                f'    <position name="{aname}" joint="{jname}" kp="1150" kv="42" ctrllimited="true" ctrlrange="0 1.28" forcelimited="true" forcerange="-215 215"/>'
            )
            actuators_meta.append({"name": aname, "kind": "flipper", "part_id": pid})
            return (
                f'{indent}<body name="{bname}" pos="{_vec_str(pos)}" quat="{qstr}">\n'
                f'{indent}  <geom name="{g_base}" type="cylinder" euler="90 0 0" size="0.052 {fwid * 0.42:.4f}" density="320" rgba="0.85 0.58 0.14 1" {ct_attr}/>\n'
                f'{indent}  <body name="{bname}_arm" pos="0.02 0 -0.01">\n'
                f'{indent}    <joint name="{jname}" type="hinge" axis="0 -1 0" limited="true" range="-3 76" damping="2.2" armature="0.04"/>\n'
                f'{indent}    <geom name="{g_arm}" type="box" pos="{flen * 0.50:.4f} 0 0.01" euler="0 10 0" '
                f'size="{flen * 0.50:.4f} {fwid * 0.50:.4f} 0.018" density="420" friction="0.22 0.005 0.002" rgba="{_rgba_str(f_rgb)}" {ct_attr}/>'
                f'{deco_block}\n'
                f'{indent}  </body>\n'
                f'{indent}</body>'
            )

        elif ptype == "hammer":
            alen = clamp(float(params.get("arm_length", 0.46)), 0.24, 0.76)
            hsz = clamp(float(params.get("head_size", 0.12)), 0.07, 0.22)
            jname = f"{bname}_ham_j"
            aname = f"act_{bname}"
            g_tur = f"{bname}_ham_mount"
            g_shaft = f"{bname}_ham_shaft"
            g_head = f"{bname}_ham_head"
            h_rgb = parse_hex_color(part.get("color"), secondary_rgb)
            register_style(g_tur, "brass", (0.85, 0.58, 0.14), "metallic")
            register_style(g_shaft, "steel", (0.45, 0.49, 0.55), "metallic")
            register_style(g_head, "weapon_tip", h_rgb, "metallic")

            actuators_xml.append(
                f'    <position name="{aname}" joint="{jname}" kp="880" kv="34" ctrllimited="true" ctrlrange="-0.30 2.05" forcelimited="true" forcerange="-170 170"/>'
            )
            actuators_meta.append({"name": aname, "kind": "hammer", "part_id": pid})
            return (
                f'{indent}<body name="{bname}" pos="{_vec_str(pos)}" quat="{qstr}">\n'
                f'{indent}  <geom name="{g_tur}" type="cylinder" euler="90 0 0" size="0.060 0.065" density="320" rgba="0.85 0.58 0.14 1" {ct_attr}/>\n'
                f'{indent}  <body name="{bname}_swing" pos="0 0 0.04">\n'
                f'{indent}    <joint name="{jname}" type="hinge" axis="0 1 0" limited="true" range="-20 122" damping="1.8" armature="0.045"/>\n'
                f'{indent}    <geom name="{g_shaft}" type="cylinder" pos="0 0 {alen * 0.5:.4f}" size="0.024 {alen * 0.5:.4f}" density="340" rgba="0.45 0.49 0.55 1" {ct_attr}/>\n'
                f'{indent}    <geom name="{g_head}" type="box" pos="0.05 0 {alen:.4f}" size="{hsz:.4f} {hsz * 0.65:.4f} {hsz * 0.55:.4f}" density="920" rgba="{_rgba_str(h_rgb)}" {ct_attr}/>'
                f'{deco_block}\n'
                f'{indent}  </body>\n'
                f'{indent}</body>'
            )

        elif ptype == "piston_punch":
            stroke = clamp(float(params.get("stroke", 0.42)), 0.20, 0.68)
            hwid = clamp(float(params.get("head_width", 0.18)), 0.09, 0.42)
            jname = f"{bname}_pist_j"
            aname = f"act_{bname}"
            g_barrel = f"{bname}_pist_barrel"
            g_rod = f"{bname}_pist_rod"
            g_fist = f"{bname}_pist_fist"
            p_rgb_c = parse_hex_color(part.get("color"), accent_rgb)
            register_style(g_barrel, "brass", (0.84, 0.56, 0.14), "metallic")
            register_style(g_rod, "steel", (0.55, 0.60, 0.68), "metallic")
            register_style(g_fist, "weapon_tip", p_rgb_c, "metallic")

            actuators_xml.append(
                f'    <position name="{aname}" joint="{jname}" kp="1750" kv="58" ctrllimited="true" ctrlrange="0 {stroke:.4f}" forcelimited="true" forcerange="-340 340"/>'
            )
            actuators_meta.append({"name": aname, "kind": "piston", "stroke": stroke, "part_id": pid})
            return (
                f'{indent}<body name="{bname}" pos="{_vec_str(pos)}" quat="{qstr}">\n'
                f'{indent}  <geom name="{g_barrel}" type="cylinder" euler="0 90 0" size="0.052 0.11" density="320" rgba="0.84 0.56 0.14 1" {ct_attr}/>\n'
                f'{indent}  <body name="{bname}_ram" pos="0.08 0 0">\n'
                f'{indent}    <joint name="{jname}" type="slide" axis="1 0 0" limited="true" range="0 {stroke + 0.02:.4f}" damping="6.5" armature="0.06"/>\n'
                f'{indent}    <geom name="{g_rod}" type="cylinder" pos="{-stroke * 0.32:.4f} 0 0" euler="0 90 0" size="0.028 {stroke * 0.45:.4f}" density="360" rgba="0.55 0.60 0.68 1" {vis_attr}/>\n'
                f'{indent}    <geom name="{g_fist}" type="box" pos="0.05 0 0" size="0.055 {hwid * 0.5:.4f} 0.065" density="780" friction="0.55 0.01 0.003" rgba="{_rgba_str(p_rgb_c)}" {ct_attr}/>'
                f'{deco_block}\n'
                f'{indent}  </body>\n'
                f'{indent}</body>'
            )

        elif ptype == "energy_cannon":
            cmode = str(params.get("mode", "standard")).strip().lower()
            if cmode not in CANNON_MODES:
                cmode = "standard"
            mcfg = CANNON_MODES[cmode]
            pitch_deg = clamp(float(params.get("pitch_deg", 0.0)), -15.0, 24.0)
            R_pitch = _euler_deg_to_mat(0.0, -pitch_deg, 0.0)
            R_tot = R @ R_pitch
            q_can = _mat_to_quat_str(R_tot)

            c_rgb = parse_hex_color(part.get("color"), (0.08, 0.92, 1.0) if prefix == "A" else (1.0, 0.36, 0.14))
            g_breech = f"{bname}_ecannon_breech"
            g_rail_l = f"{bname}_ecannon_rail_l"
            g_rail_r = f"{bname}_ecannon_rail_r"
            g_core = f"{bname}_ecannon_core"
            g_muz = f"{bname}_ecannon_muzzle"
            register_style(g_breech, "steel", (0.22, 0.26, 0.34), "metallic")
            register_style(g_rail_l, "steel", (0.46, 0.52, 0.60), "metallic")
            register_style(g_rail_r, "steel", (0.46, 0.52, 0.60), "metallic")
            register_style(g_core, "energy_core", c_rgb, "neon")
            register_style(g_muz, "energy_core", c_rgb, "neon")

            cidx = len(cannons_meta)
            bolt_body_names: List[str] = []
            for b_i in range(BOLTS_PER_CANNON):
                bb_name = f"{prefix}_bolt_{cidx}_{b_i}"
                g_orb = f"{bb_name}_orb"
                g_tail = f"{bb_name}_tail"
                bolt_body_names.append(bb_name)
                register_style(g_orb, "energy_bolt", c_rgb, "neon", {"is_projectile": True})
                register_style(g_tail, "energy_bolt_tail", c_rgb, "neon", {"is_projectile": True})
                mocap_bolts_xml.append(
                    f'    <body name="{bb_name}" mocap="true" pos="0 0 -9.5">\n'
                    f'      <geom name="{g_orb}" type="sphere" size="0.068" rgba="{_rgba_str(c_rgb, 0.98)}" {vis_attr}/>\n'
                    f'      <geom name="{g_tail}" type="cylinder" pos="-0.12 0 0" euler="0 90 0" size="0.036 0.12" rgba="{_rgba_str(c_rgb, 0.85)}" {vis_attr}/>\n'
                    f'    </body>'
                )

            cannons_meta.append({
                "body_name": bname,
                "mode": cmode,
                "cooldown": mcfg["cooldown"],
                "bolt_speed": mcfg["speed"],
                "bolt_impulse": mcfg["impulse"],
                "recoil_impulse": mcfg["recoil"],
                "bolt_damage": mcfg["damage"],
                "muzzle_local": [0.34, 0.0, 0.0],
                "bolt_body_names": bolt_body_names,
                "part_id": pid,
            })
            return (
                f'{indent}<body name="{bname}" pos="{_vec_str(pos)}" quat="{q_can}">\n'
                f'{indent}  <geom name="{g_breech}" type="box" pos="0.02 0 0" size="0.10 0.075 0.052" density="340" rgba="0.22 0.26 0.34 1" {ct_attr}/>\n'
                f'{indent}  <geom name="{g_rail_l}" type="box" pos="0.17 0.052 0" size="0.16 0.016 0.024" density="300" rgba="0.46 0.52 0.60 1" {ct_attr}/>\n'
                f'{indent}  <geom name="{g_rail_r}" type="box" pos="0.17 -0.052 0" size="0.16 0.016 0.024" density="300" rgba="0.46 0.52 0.60 1" {ct_attr}/>\n'
                f'{indent}  <geom name="{g_core}" type="cylinder" pos="0.15 0 0" euler="0 90 0" size="0.026 0.13" mass="0.2" rgba="{_rgba_str(c_rgb)}" {vis_attr}/>\n'
                f'{indent}  <geom name="{g_muz}" type="cylinder" pos="0.31 0 0" euler="0 90 0" size="0.048 0.012" mass="0.1" rgba="{_rgba_str(c_rgb)}" {vis_attr}/>'
                f'{deco_block}\n'
                f'{indent}</body>'
            )

        else:  # thruster
            aim = str(params.get("aim", "push_forward")).strip().lower()
            pwr = clamp(float(params.get("power", 1.0)), 0.3, 1.0)
            aim_map = {
                "push_forward":  ([1.0, 0.0, 0.0],  [-1.0, 0.0, 0.0], "0 -90 0"),
                "push_backward": ([-1.0, 0.0, 0.0], [1.0, 0.0, 0.0],  "0 90 0"),
                "push_left":     ([0.0, 1.0, 0.0],  [0.0, -1.0, 0.0], "90 0 0"),
                "push_right":    ([0.0, -1.0, 0.0], [0.0, 1.0, 0.0],  "-90 0 0"),
                "push_down":     ([0.0, 0.0, -1.0], [0.0, 0.0, 1.0],  "0 0 0"),
            }
            force_local, exhaust_local, bell_euler = aim_map.get(aim, aim_map["push_forward"])
            j_flame = f"{bname}_flame_j"
            a_flame = f"act_{bname}_flame"
            g_bell = f"{bname}_nozzle"
            g_flame = f"{bname}_flame"
            fl_rgb = parse_hex_color(part.get("color"), (1.0, 0.45, 0.08))
            register_style(g_bell, "steel", (0.28, 0.31, 0.36), "metallic")
            register_style(g_flame, "flame", fl_rgb, "neon", {"deco_shape": "cone"})

            actuators_xml.append(
                f'    <position name="{a_flame}" joint="{j_flame}" kp="120" kv="12" ctrllimited="true" ctrlrange="0 0.24"/>'
            )
            thrusters_meta.append({
                "body_name": bname,
                "flame_act_name": a_flame,
                "force_local": force_local,
                "max_force_n": 95.0 * pwr,
                "part_id": pid,
            })
            return (
                f'{indent}<body name="{bname}" pos="{_vec_str(pos)}" quat="{qstr}">\n'
                f'{indent}  <geom name="{g_bell}" type="cylinder" euler="{bell_euler}" size="0.058 0.068" density="300" rgba="0.28 0.31 0.36 1" {ct_attr}/>\n'
                f'{indent}  <body name="{bname}_plume" pos="0 0 0">\n'
                f'{indent}    <joint name="{j_flame}" type="slide" axis="{_vec_str(exhaust_local)}" limited="true" range="0 0.26" damping="1.2" armature="0.002"/>\n'
                f'{indent}    <geom name="{g_flame}" type="cylinder" euler="{bell_euler}" size="0.044 0.060" mass="0.0005" rgba="{_rgba_str(fl_rgb, 0.92)}" {vis_attr}/>\n'
                f'{indent}  </body>'
                f'{deco_block}\n'
                f'{indent}</body>'
            )

    child_xml_list = [
        emit_part_recursive(ci, cp, chassis_half, 0.0, "      ")
        for ci, cp in children_by_parent.get("chassis", [])
    ]
    chassis_decos_xml = emit_decorations("chassis", "      ")

    spawn_z = spawn_z_override if spawn_z_override is not None else (max(ch_hz, -lowest_z) + 0.012)
    R_spawn = _euler_deg_to_mat(0.0, 0.0, spawn_yaw_deg)

    g_core = f"{prefix}_core_box"
    register_style(g_core, "chassis", primary_rgb, chassis_mat_style)

    body_xml = (
        f'    <body name="{prefix}_root" pos="{spawn_xy[0]:.4f} {spawn_xy[1]:.4f} {spawn_z:.4f}" quat="{_mat_to_quat_str(R_spawn)}">\n'
        f'      <freejoint name="{prefix}_free"/>\n'
        f'      <geom name="{g_core}" type="box" size="{_vec_str(chassis_half)}" density="{chassis_density:.1f}" '
        f'friction="0.58 0.01 0.003" rgba="{_rgba_str(primary_rgb)}" {ct_attr}/>\n'
        + (chassis_decos_xml + "\n" if chassis_decos_xml else "")
        + ("\n".join(child_xml_list) + "\n" if child_xml_list else "")
        + '    </body>'
    )

    return {
        "body_xml": body_xml,
        "mocap_bolts_xml": "\n".join(mocap_bolts_xml),
        "actuator_xml": "\n".join(actuators_xml),
        "actuators_meta": actuators_meta,
        "thrusters_meta": thrusters_meta,
        "cannons_meta": cannons_meta,
        "geom_styles": geom_styles,
        "spawn_z": spawn_z,
    }


def _compute_subtree_mass_kg(model: mujoco.MjModel, prefix: str) -> float:
    tot = 0.0
    for b in range(1, model.nbody):
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if bname.startswith(f"{prefix}_") and "_bolt_" not in bname:
            tot += float(model.body_mass[b])
    return tot


def compile_single_car_model(spec: Dict[str, Any]) -> Tuple[mujoco.MjModel, mujoco.MjData, Dict[str, Any]]:
    """Compile a single car on a flat inspection floor to check mass, CoM, unrestricted bbox & real flat-sprint top speed."""
    car_bundle = build_vehicle_xml(spec, prefix="A", spawn_xy=(0.0, 0.0), spawn_yaw_deg=0.0, contype=2, conaffinity=5)
    xml = f"""<mujoco model="inspect_car">
  <compiler angle="degree" autolimits="true"/>
  <option timestep="{PHYSICS_DT:.8f}" gravity="0 0 -9.81" integrator="implicitfast"/>
  <default>
    <geom condim="3" solref="0.018 1.05" solimp="0.86 0.93 0.002"/>
    <joint solreflimit="0.026 1.05" solimplimit="0.84 0.92 0.004"/>
  </default>
  <worldbody>
    <geom name="env_floor" type="plane" pos="0 0 0" size="40 40 0.1" friction="1.18 0.012 0.004" contype="1" conaffinity="6"/>
{car_bundle["body_xml"]}
{car_bundle["mocap_bolts_xml"]}
  </worldbody>
  <actuator>
{car_bundle["actuator_xml"]}
  </actuator>
</mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    min_pt = np.array([999.0, 999.0, 999.0], dtype=np.float64)
    max_pt = np.array([-999.0, -999.0, -999.0], dtype=np.float64)
    total_mass = 0.0
    com = np.zeros(3, dtype=np.float64)
    for b in range(1, model.nbody):
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if "_bolt_" in bname:
            continue
        bm = float(model.body_mass[b])
        total_mass += bm
        com += bm * data.xipos[b]
    if total_mass > 1e-6:
        com /= total_mass

    for gi in range(model.ngeom):
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gi) or ""
        if not gname.startswith("A_") or gname.endswith("_flame") or "_bolt_" in gname:
            continue
        gpos = data.geom_xpos[gi]
        rbound = float(model.geom_rbound[gi])
        gtype = int(model.geom_type[gi])
        R = data.geom_xmat[gi].reshape(3, 3)
        sz = model.geom_size[gi]
        if gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
            ext = np.abs(R) @ sz[:3]
        elif gtype == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
            ext = np.abs(R) @ np.array([sz[0], sz[0], sz[1]])
        else:
            ext = np.array([rbound, rbound, rbound])
        min_pt = np.minimum(min_pt, gpos - ext)
        max_pt = np.maximum(max_pt, gpos + ext)

    bbox = (
        round(float(max_pt[0] - min_pt[0]), 3),
        round(float(max_pt[1] - min_pt[1]), 3),
        round(float(max_pt[2] - min_pt[2]), 3),
    )

    # Measure true flat-ground sprint top speed over 2.0s in a disposable MjData (~2.2ms CPU!)
    sprint_data = mujoco.MjData(model)
    root_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "A_root")
    wheel_aids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, am["name"])
        for am in car_bundle["actuators_meta"]
        if am["kind"] == "wheel"
    ]
    wheel_aids = [aid for aid in wheel_aids if aid >= 0]
    floor_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "env_floor")
    est_top_spd = 0.0
    if wheel_aids and root_bid >= 0:
        for _ in range(720):
            for aid in wheel_aids:
                sprint_data.ctrl[aid] = 1.0
            grounded = any(
                int(sprint_data.contact[ci].geom1) == floor_gid or int(sprint_data.contact[ci].geom2) == floor_gid
                for ci in range(sprint_data.ncon)
            )
            apply_grounded_mass_drag(sprint_data, root_bid, total_mass, grounded)
            mujoco.mj_step(model, sprint_data)
            vxy = math.hypot(float(sprint_data.cvel[root_bid, 3]), float(sprint_data.cvel[root_bid, 4]))
            if vxy > est_top_spd:
                est_top_spd = vxy

    diagnostics = {
        "total_mass_kg": round(total_mass, 2),
        "weight_class": classify_weight_class(total_mass),
        "est_top_speed_mps": round(est_top_spd, 2),
        "center_of_mass_m": [round(float(com[0]), 3), round(float(com[1]), 3), round(float(com[2]), 3)],
        "bbox_m": list(bbox),
        "bbox_limit": "unrestricted (governed naturally by volume x density mass & inertia)",
        "spawn_z_m": round(float(car_bundle["spawn_z"]), 3),
        "geom_styles": car_bundle["geom_styles"],
    }
    return model, data, diagnostics


def full_validate_car(spec: Dict[str, Any]) -> Tuple[bool, List[str], List[str], Dict[str, Any]]:
    """Run schema/budget validation AND 3D MuJoCo physics & mobility diagnostics (NO upper bbox restriction!)."""
    ok, errors, warnings, summary = validate_car_dict(spec)
    if not ok:
        return False, errors, warnings, summary

    try:
        _, _, diag = compile_single_car_model(spec)
    except Exception as exc:
        errors.append(f"MuJoCo 3D physics compilation failed: {exc}")
        return False, errors, warnings, summary

    summary.update({k: v for k, v in diag.items() if k != "geom_styles"})

    com_z = diag["center_of_mass_m"][2]
    bx, by, _ = diag["bbox_m"]
    min_footprint = max(0.10, min(bx, by))
    if com_z > 0.65 * min_footprint and com_z > 0.34:
        warnings.append(
            f"High Center of Mass (z={com_z:.2f}m vs footprint {bx:.2f}m x {by:.2f}m) — car may tip during sharp turns or impacts. "
            f"Consider widening stance or adding a 'gyro_stabilizer' on 'bottom'."
        )
    if diag["total_mass_kg"] > 78.0:
        warnings.append(
            f"Super-heavy build ({diag['total_mass_kg']:.1f} kg -> est. top speed {diag['est_top_speed_mps']:.2f} m/s): "
            f"high flip/bolt resistance, but noticeably slower speed & turn rate."
        )

    return (len(errors) == 0), errors, warnings, summary


def _terrain_surface_z_at(stage_key: str, x: float, y: float) -> float:
    """Exact analytical terrain top-surface height z(x, y) so cars spawn cleanly on flat or sloped decks."""
    if stage_key == "spring_arena":
        r = math.hypot(x, y)
        return -(r - 2.45) * math.tan(math.radians(7.5)) if r > 2.45 else 0.0
    if stage_key == "camelback_ridge":
        ax = abs(x)
        if ax <= 0.52:
            return 0.24
        if ax < 1.75:
            return 0.24 * (1.75 - ax) / (1.75 - 0.52)
        return 0.0
    # portal_arena (flat core |x|<=2.60m, smooth -8.0 deg downward danger aprons |x|>2.60m)
    ax = abs(x)
    return -(ax - 2.60) * math.tan(math.radians(8.0)) if ax > 2.60 else 0.0


def compile_match_scene(
    spec_a: Dict[str, Any],
    spec_b: Dict[str, Any],
    stage: str,
    swap_sides: bool = False,
) -> Dict[str, Any]:
    """Compile two cars + one of the 3 slope & spring arenas into a deterministic 360Hz MuJoCo match scene."""
    stage_key = resolve_stage_key(stage)
    ameta = ARENA_META[stage_key]
    spawn_a = ameta["spawn_b"] if swap_sides else ameta["spawn_a"]
    spawn_b = ameta["spawn_a"] if swap_sides else ameta["spawn_b"]
    yaw_a = ameta["yaw_b"] if swap_sides else ameta["yaw_a"]
    yaw_b = ameta["yaw_a"] if swap_sides else ameta["yaw_b"]

    arena_world_xml, arena_act_xml, trap_act_names = build_arena_mjcf(stage_key)
    car_a = build_vehicle_xml(spec_a, prefix="A", spawn_xy=spawn_a, spawn_yaw_deg=yaw_a, contype=2, conaffinity=5)
    car_b = build_vehicle_xml(spec_b, prefix="B", spawn_xy=spawn_b, spawn_yaw_deg=yaw_b, contype=4, conaffinity=3)

    # Elevate spawn_z if spawn_xy sits on an inclined bowl/slope (e.g., Stage 3 turbo_bowl!)
    surf_za = _terrain_surface_z_at(stage_key, float(spawn_a[0]), float(spawn_a[1]))
    surf_zb = _terrain_surface_z_at(stage_key, float(spawn_b[0]), float(spawn_b[1]))
    if surf_za > 0.001:
        car_a = build_vehicle_xml(
            spec_a, prefix="A", spawn_xy=spawn_a, spawn_yaw_deg=yaw_a, contype=2, conaffinity=5,
            spawn_z_override=float(car_a["spawn_z"]) + surf_za + 0.015,
        )
    if surf_zb > 0.001:
        car_b = build_vehicle_xml(
            spec_b, prefix="B", spawn_xy=spawn_b, spawn_yaw_deg=yaw_b, contype=4, conaffinity=3,
            spawn_z_override=float(car_b["spawn_z"]) + surf_zb + 0.015,
        )

    model_name_xml = saxutils.quoteattr(f"match_{stage_key}")
    xml = f"""<mujoco model={model_name_xml}>
  <compiler angle="degree" autolimits="true"/>
  <option timestep="{PHYSICS_DT:.8f}" gravity="0 0 -9.81" integrator="implicitfast" iterations="35" ls_iterations="12"/>
  <size nconmax="640" njmax="1920"/>
  <default>
    <geom condim="3" solref="0.018 1.05" solimp="0.86 0.93 0.002"/>
    <joint solreflimit="0.026 1.05" solimplimit="0.84 0.92 0.004"/>
  </default>
  <worldbody>
{arena_world_xml}
{car_a["body_xml"]}
{car_b["body_xml"]}
{car_a["mocap_bolts_xml"]}
{car_b["mocap_bolts_xml"]}
  </worldbody>
  <actuator>
{arena_act_xml}
{car_a["actuator_xml"]}
{car_b["actuator_xml"]}
  </actuator>
</mujoco>"""

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    mass_a = _compute_subtree_mass_kg(model, "A")
    mass_b = _compute_subtree_mass_kg(model, "B")

    geom_styles = {}
    geom_styles.update(car_a["geom_styles"])
    geom_styles.update(car_b["geom_styles"])

    return {
        "stage_key": stage_key,
        "arena_meta": ameta,
        "model": model,
        "data": data,
        "car_a": car_a,
        "car_b": car_b,
        "mass_a": mass_a,
        "mass_b": mass_b,
        "trap_act_names": trap_act_names,
        "geom_styles": geom_styles,
    }
