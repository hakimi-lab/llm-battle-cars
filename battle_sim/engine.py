"""
Deterministic 360Hz MuJoCo Match Simulation Engine, 3D Ranged Energy Bolt Physics,
Mass-vs-Speed Grounded Load Dynamics, Tactical Floor Springs & 30fps Replay Exporter (v3.0).
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mujoco
import numpy as np

from .arenas import (
    ARENA_META,
    ARENA_ORDER,
    compute_car_mud_factor,
    create_hazard_runtime,
    point_on_ice,
    resolve_stage_key,
    step_arena_hazards,
    step_arena_portals,
    try_warp_bolt,
)
from .catalog import FPS, MATCH_FRAMES, PHYSICS_DT, POST_KO_FRAMES, SUBSTEPS
from .compiler import apply_grounded_mass_drag, compile_match_scene
from .strategy import VehicleController

BOLT_RADIUS = 0.068

WEAPON_GEOM_TOKENS = (
    "_blade",
    "_tooth",
    "_vdrum",
    "_vtooth",
    "_flip_plate",
    "_ham_head",
    "_pist_fist",
    "_wedge_ramp",
)


def _is_weapon_geom(gname: str) -> bool:
    return any(tok in gname for tok in WEAPON_GEOM_TOKENS)


def _segment_vs_aabb(p0_local: np.ndarray, p1_local: np.ndarray, half_ext: np.ndarray) -> Optional[float]:
    """Exact Slab intersection test of segment p0->p1 against [-half_ext, +half_ext]. Returns t in [0, 1] or None."""
    d = p1_local - p0_local
    t_min = 0.0
    t_max = 1.0
    for i in range(3):
        h = float(half_ext[i])
        if abs(float(d[i])) < 1e-9:
            if abs(float(p0_local[i])) > h:
                return None
        else:
            inv_d = 1.0 / float(d[i])
            t1 = (-h - float(p0_local[i])) * inv_d
            t2 = (h - float(p0_local[i])) * inv_d
            if t1 > t2:
                t1, t2 = t2, t1
            t_min = max(t_min, t1)
            t_max = min(t_max, t2)
            if t_min > t_max:
                return None
    return float(t_min)


def _segment_vs_geom(
    p0_w: np.ndarray,
    p1_w: np.ndarray,
    gpos: np.ndarray,
    gR: np.ndarray,
    gtype: int,
    gsize: np.ndarray,
    inflate_r: float = BOLT_RADIUS,
) -> Optional[float]:
    """Fast exact/tight segment intersection against a sphere, cylinder, or box geom inflated by `inflate_r`."""
    p0 = gR.T @ (p0_w - gpos)
    p1 = gR.T @ (p1_w - gpos)

    if gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
        he = np.array([gsize[0] + inflate_r, gsize[1] + inflate_r, gsize[2] + inflate_r], dtype=np.float64)
        return _segment_vs_aabb(p0, p1, he)

    if gtype == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        rad = float(gsize[0]) + inflate_r
        hh = float(gsize[1]) + inflate_r
        t_box = _segment_vs_aabb(p0, p1, np.array([rad, rad, hh], dtype=np.float64))
        if t_box is None:
            return None
        pt = p0 + t_box * (p1 - p0)
        if pt[0] * pt[0] + pt[1] * pt[1] <= rad * rad * 1.08:
            return t_box
        pm = 0.5 * (p0 + p1)
        if abs(pm[2]) <= hh and (pm[0] * pm[0] + pm[1] * pm[1] <= rad * rad):
            return 0.5
        return None

    rad_s = float(gsize[0]) + inflate_r
    d = p1 - p0
    a = float(np.dot(d, d))
    b = 2.0 * float(np.dot(p0, d))
    c = float(np.dot(p0, p0)) - rad_s * rad_s
    if c <= 0.0:
        return 0.0
    if a < 1e-12:
        return None
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return None
    t = (-b - math.sqrt(disc)) / (2.0 * a)
    return float(t) if 0.0 <= t <= 1.0 else None


def _dir_to_quat(dir_unit: np.ndarray) -> np.ndarray:
    """Build [w, x, y, z] quaternion aligning local +X axis along `dir_unit`."""
    x_ax = dir_unit / max(1e-8, float(np.linalg.norm(dir_unit)))
    up_ref = np.array([0.0, 0.0, 1.0]) if abs(float(x_ax[2])) < 0.92 else np.array([0.0, 1.0, 0.0])
    y_ax = np.cross(up_ref, x_ax)
    y_ax /= max(1e-8, float(np.linalg.norm(y_ax)))
    z_ax = np.cross(x_ax, y_ax)
    R = np.column_stack([x_ax, y_ax, z_ax])
    q = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(q, R.reshape(-1))
    return q


def run_single_match(
    spec_a: Dict[str, Any],
    spec_b: Dict[str, Any],
    stage: str,
    swap_sides: bool = False,
    record_replay: bool = False,
    return_trace_data: bool = False,
) -> Dict[str, Any]:
    """
    Simulate one deterministic 15.0s match at 360Hz (exact 30.000 FPS frame cadence).
    v3.0 features:
      - Open slope & reactive floor-spring physics (`trap_spring_*`)
      - Mass-dependent grounded rolling & yaw scrub dynamics (`apply_grounded_mass_drag`)
      - 3D ranged plasma bolts with physical armor/wedge deflection (30% reduction) & recoil
      - White-box per-rule strategy attribution & frame-accurate foley events
    """
    t0_wall = time.perf_counter()
    scene = compile_match_scene(spec_a, spec_b, stage=stage, swap_sides=swap_sides)
    stage_key = scene["stage_key"]
    model: mujoco.MjModel = scene["model"]
    data: mujoco.MjData = scene["data"]
    mass_a = float(scene["mass_a"])
    mass_b = float(scene["mass_b"])

    trap_act_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, nm)
        for nm in scene["trap_act_names"]
    ]
    hazard_state = create_hazard_runtime(stage_key)

    geom_names: List[str] = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gi) or f"geom_{gi}"
        for gi in range(model.ngeom)
    ]
    hazard_geom_ids = [
        gi for gi, nm in enumerate(geom_names)
        if nm.startswith("trap_") and int(model.geom_contype[gi]) != 0
    ]

    solid_env_gids: List[int] = []
    solid_a_gids: List[int] = []
    solid_b_gids: List[int] = []
    for gi, nm in enumerate(geom_names):
        if int(model.geom_contype[gi]) == 0 or nm == "env_catch_floor":
            continue
        if nm.startswith("A_"):
            solid_a_gids.append(gi)
        elif nm.startswith("B_"):
            solid_b_gids.append(gi)
        else:
            solid_env_gids.append(gi)

    ctrl_a = VehicleController("A", spec_a, stage_key, model, scene["car_a"], "B", hazard_geom_ids)
    ctrl_b = VehicleController("B", spec_b, stage_key, model, scene["car_b"], "A", hazard_geom_ids)

    body_a = ctrl_a.root_body_id
    body_b = ctrl_b.root_body_id

    # Scoring & telemetry accumulators
    imp_a = 0.0
    imp_b = 0.0
    peak_hit_a = 0.0
    peak_hit_b = 0.0
    big_hits_a = 0
    big_hits_b = 0
    spring_launches_a = 0
    spring_launches_b = 0
    spring_herds_by_a = 0
    spring_herds_by_b = 0
    last_spring_t_a = -9.0
    last_spring_t_b = -9.0
    last_spring_name_a = ""
    last_spring_name_b = ""

    flips_inflicted_a = 0
    flips_inflicted_b = 0
    prev_tilted_a = False
    prev_tilted_b = False

    shots_fired_a = 0
    shots_fired_b = 0
    shots_hit_a = 0
    shots_hit_b = 0

    aggression_a = 0.0
    aggression_b = 0.0
    upright_frames_a = 0
    upright_frames_b = 0
    immob_inverted_sec_a = 0.0
    immob_inverted_sec_b = 0.0
    min_edge_a = 99.0
    min_edge_b = 99.0
    max_speed_a = 0.0
    max_speed_b = 0.0

    grounded_a = True
    grounded_b = True

    highlights: List[Dict[str, Any]] = []
    foley_events: List[Dict[str, Any]] = []
    timeline_1hz_a: List[Dict[str, Any]] = []
    recent_states_a: List[Dict[str, Any]] = []
    trace_frames_2d: List[Dict[str, Any]] = []
    shot_traces_2d: List[Dict[str, Any]] = []

    last_clang_f = -4
    crashed_floor_a = False
    crashed_floor_b = False

    ko_winner: Optional[str] = None
    ko_reason: str = "judges_decision"
    ko_time_sec: Optional[float] = None
    ko_frame: Optional[int] = None
    ko_diagnosis: Optional[Dict[str, Any]] = None
    ko_note_a: Optional[str] = None

    xpos_frames: List[np.ndarray] = []
    xmat_frames: List[np.ndarray] = []
    live_scores_a_frames: List[List[float]] = []
    live_scores_b_frames: List[List[float]] = []
    last_live_a: List[float] = [60.0, 25.0, 15.0, 20.0]
    last_live_b: List[float] = [60.0, 25.0, 15.0, 20.0]

    active_bolts: List[Dict[str, Any]] = []
    zero_torque = np.zeros(3, dtype=np.float64)
    force6 = np.zeros(6, dtype=np.float64)
    max_total_frames = MATCH_FRAMES + (POST_KO_FRAMES if record_replay else 0)

    sens_a: Dict[str, Any] = {}
    sens_b: Dict[str, Any] = {}

    for frame_idx in range(max_total_frames):
        if ko_frame is not None:
            if not record_replay or (frame_idx - ko_frame) >= POST_KO_FRAMES:
                break

        frame_best_ab_imp = 0.0
        frame_best_ab_weapon = False
        frame_spring_fired = False
        frame_floor_crash = 0.0

        for sub in range(SUBSTEPS):
            t_sec = (frame_idx * SUBSTEPS + sub) * PHYSICS_DT
            data.qfrc_applied[:] = 0.0

            pa_xyz = (float(data.xpos[body_a, 0]), float(data.xpos[body_a, 1]), float(data.xpos[body_a, 2]))
            pb_xyz = (float(data.xpos[body_b, 0]), float(data.xpos[body_b, 1]), float(data.xpos[body_b, 2]))
            trig_events = step_arena_hazards(
                stage_key,
                model,
                data,
                t_sec,
                trap_act_ids,
                car_positions=(pa_xyz, pb_xyz),
                hazard_state=hazard_state,
                car_geom_groups=(solid_a_gids, solid_b_gids),
            )
            if trig_events:
                frame_spring_fired = True
                ab_dist_now = math.hypot(pa_xyz[0] - pb_xyz[0], pa_xyz[1] - pb_xyz[1])
                for who, sp_nm in trig_events:
                    if ko_frame is None:
                        if who == "A":
                            spring_launches_a += 1
                            last_spring_t_a = t_sec
                            last_spring_name_a = sp_nm
                            ctrl_a.record_spring_launch(is_self=True)
                            ctrl_b.record_spring_launch(is_self=False)
                            if ab_dist_now < 1.85:
                                spring_herds_by_b += 1
                            if len(highlights) < 18:
                                highlights.append({
                                    "t": round(t_sec, 2),
                                    "event": f"A_launched_by_{sp_nm}",
                                    "rule": ctrl_a.active_rule_name,
                                })
                        else:
                            spring_launches_b += 1
                            last_spring_t_b = t_sec
                            last_spring_name_b = sp_nm
                            ctrl_b.record_spring_launch(is_self=True)
                            ctrl_a.record_spring_launch(is_self=False)
                            if ab_dist_now < 1.85:
                                spring_herds_by_a += 1
                            if len(highlights) < 18:
                                highlights.append({
                                    "t": round(t_sec, 2),
                                    "event": f"B_launched_by_{sp_nm}",
                                    "rule": ctrl_a.active_rule_name,
                                })

            warp_events = step_arena_portals(
                stage_key, model, data, t_sec, body_a, body_b, hazard_state
            )
            if warp_events:
                for who_w, in_tag, out_tag in warp_events:
                    foley_events.append({"frame": frame_idx, "kind": "portal_warp", "strength": 1.1})
                    if ko_frame is None:
                        if who_w == "A":
                            ctrl_a.record_portal_warp(is_self=True)
                            ctrl_b.record_portal_warp(is_self=False)
                            if len(highlights) < 18:
                                highlights.append({
                                    "t": round(t_sec, 2),
                                    "event": f"A_warped_{in_tag}_to_{out_tag}",
                                    "rule": ctrl_a.active_rule_name,
                                })
                        else:
                            ctrl_b.record_portal_warp(is_self=True)
                            ctrl_a.record_portal_warp(is_self=False)
                            if len(highlights) < 18:
                                highlights.append({
                                    "t": round(t_sec, 2),
                                    "event": f"B_warped_{in_tag}_to_{out_tag}",
                                    "rule": ctrl_a.active_rule_name,
                                })

            mud_a = compute_car_mud_factor(stage_key, model, data, body_a, solid_a_gids) if grounded_a else 0.0
            mud_b = compute_car_mud_factor(stage_key, model, data, body_b, solid_b_gids) if grounded_b else 0.0
            ice_a = point_on_ice(stage_key, pa_xyz[0], pa_xyz[1])
            ice_b = point_on_ice(stage_key, pb_xyz[0], pb_xyz[1])

            if sub % 3 == 0:
                ctrl_dt = 3.0 * PHYSICS_DT
                counting = (ko_frame is None)
                bolts_from_b = [b for b in active_bolts if b["owner"] == "B"]
                bolts_from_a = [b for b in active_bolts if b["owner"] == "A"]
                sens_a = ctrl_a.step(
                    data, t_sec, ctrl_dt, enemy_bolts=bolts_from_b, hazard_state=hazard_state,
                    self_mud_factor=mud_a, enemy_mud_factor=mud_b, record_stats=counting,
                )
                sens_b = ctrl_b.step(
                    data, t_sec, ctrl_dt, enemy_bolts=bolts_from_a, hazard_state=hazard_state,
                    self_mud_factor=mud_b, enemy_mud_factor=mud_a, record_stats=counting,
                )

                # Spawn newly fired energy bolts & apply real Newton's-3rd-law recoil!
                for new_b in ctrl_a.pop_fired_bolts() + ctrl_b.pop_fired_bolts():
                    for old_ab in active_bolts:
                        if old_ab["mocap_id"] == new_b["mocap_id"] and ko_frame is None:
                            src_ctrl = ctrl_a if old_ab["owner"] == "A" else ctrl_b
                            src_ctrl.record_shot_outcome(old_ab["rule_slot"], "missed")
                    active_bolts = [ab for ab in active_bolts if ab["mocap_id"] != new_b["mocap_id"]]
                    active_bolts.append(new_b)
                    rec_f = (-new_b["recoil"] / PHYSICS_DT) * new_b["dir"]
                    mujoco.mj_applyFT(
                        model, data, rec_f, zero_torque, new_b["pos"], new_b["cannon_body_id"], data.qfrc_applied
                    )
                    if ko_frame is None:
                        if new_b["owner"] == "A":
                            shots_fired_a += 1
                        else:
                            shots_fired_b += 1
                    foley_events.append({"frame": frame_idx, "kind": "plasma_fire", "strength": 0.9})

            # Apply grounded mass-vs-speed tire rolling, viscous mud drag & low-drag mirror ice dynamics
            apply_grounded_mass_drag(data, body_a, mass_a, grounded_a, mud_factor=mud_a, on_ice=ice_a)
            apply_grounded_mass_drag(data, body_b, mass_b, grounded_b, mud_factor=mud_b, on_ice=ice_b)

            # Step all active 3D energy bolts & test slope/terrain / hull impact / portal warp!
            surviving_bolts: List[Dict[str, Any]] = []
            for bolt in active_bolts:
                is_a_shot = (bolt["owner"] == "A")
                shooter_ctrl = ctrl_a if is_a_shot else ctrl_b

                bolt["ttl"] -= PHYSICS_DT
                if bolt["ttl"] <= 0.0:
                    data.mocap_pos[bolt["mocap_id"]] = (0.0, 0.0, -9.5)
                    if ko_frame is None:
                        shooter_ctrl.record_shot_outcome(bolt["rule_slot"], "missed")
                        if return_trace_data:
                            shot_traces_2d.append({
                                "owner": bolt["owner"],
                                "p0": [float(bolt["spawn_pos"][0]), float(bolt["spawn_pos"][1])],
                                "p1": [float(bolt["pos"][0]), float(bolt["pos"][1])],
                                "outcome": "missed",
                                "rule": bolt["rule_name"],
                                "t": round(t_sec, 2),
                            })
                    continue

                p0 = bolt["pos"].copy()
                bolt["vel"][2] -= 1.10 * PHYSICS_DT
                p1 = p0 + bolt["vel"] * PHYSICS_DT

                enemy_gids = solid_b_gids if is_a_shot else solid_a_gids

                # 1. Check intersection against solid sloped terrain / extended spring pads
                t_env: Optional[float] = None
                hit_env_gid: int = -1
                for egid in solid_env_gids:
                    th = _segment_vs_geom(
                        p0, p1, data.geom_xpos[egid], data.geom_xmat[egid].reshape(3, 3),
                        int(model.geom_type[egid]), model.geom_size[egid], inflate_r=0.045
                    )
                    if th is not None and (t_env is None or th < t_env):
                        t_env = th
                        hit_env_gid = egid

                # 2. Check intersection against enemy vehicle geoms
                t_enemy: Optional[float] = None
                hit_enemy_gid: int = -1
                for tgid in enemy_gids:
                    th = _segment_vs_geom(
                        p0, p1, data.geom_xpos[tgid], data.geom_xmat[tgid].reshape(3, 3),
                        int(model.geom_type[tgid]), model.geom_size[tgid]
                    )
                    if th is not None and (t_enemy is None or th < t_enemy):
                        t_enemy = th
                        hit_enemy_gid = tgid

                best_env = t_env if t_env is not None else 99.0
                best_en = t_enemy if t_enemy is not None else 99.0

                if best_env < best_en and best_env <= 1.0:
                    hit_pt = p0 + best_env * (p1 - p0)
                    data.mocap_pos[bolt["mocap_id"]] = (0.0, 0.0, -9.5)
                    if ko_frame is None:
                        shooter_ctrl.record_shot_outcome(bolt["rule_slot"], "blocked_by_terrain")
                        env_nm = geom_names[hit_env_gid] if hit_env_gid >= 0 else "terrain"
                        if is_a_shot and len(highlights) < 18:
                            highlights.append({
                                "t": round(t_sec, 2),
                                "event": f"A_shot_hit_terrain_{env_nm}",
                                "rule": bolt["rule_name"],
                            })
                        if return_trace_data:
                            shot_traces_2d.append({
                                "owner": bolt["owner"],
                                "p0": [float(bolt["spawn_pos"][0]), float(bolt["spawn_pos"][1])],
                                "p1": [float(hit_pt[0]), float(hit_pt[1])],
                                "outcome": "blocked_by_terrain",
                                "rule": bolt["rule_name"],
                                "t": round(t_sec, 2),
                            })
                    foley_events.append({"frame": frame_idx, "kind": "hull_thud", "strength": 0.55})
                    continue

                if best_en <= 1.0 and hit_enemy_gid >= 0:
                    hit_pt = p0 + best_en * (p1 - p0)
                    hit_body = int(model.geom_bodyid[hit_enemy_gid])
                    hit_gname = geom_names[hit_enemy_gid]
                    armor_mult = 0.70 if ("_shield" in hit_gname or "_wedge_" in hit_gname) else 1.0

                    v_norm = bolt["vel"] / max(1e-6, float(np.linalg.norm(bolt["vel"])))
                    knock_dir = v_norm + np.array([0.0, 0.0, 0.36], dtype=np.float64)
                    knock_dir /= float(np.linalg.norm(knock_dir))
                    imp_mag = bolt["impulse"] * armor_mult
                    f_vec = (imp_mag / PHYSICS_DT) * knock_dir
                    mujoco.mj_applyFT(model, data, f_vec, zero_torque, hit_pt, hit_body, data.qfrc_applied)

                    if ko_frame is None:
                        shooter_ctrl.record_shot_outcome(bolt["rule_slot"], "hit_hull")
                        dmg_pts = bolt["damage"] * armor_mult
                        if is_a_shot:
                            imp_a += dmg_pts
                            peak_hit_a = max(peak_hit_a, dmg_pts)
                            big_hits_a += 1
                            shots_hit_a += 1
                            if len(highlights) < 18:
                                highlights.append({
                                    "t": round(t_sec, 2),
                                    "event": "A_energy_bolt_direct_hit",
                                    "target": hit_gname,
                                    "impulse": round(imp_mag, 1),
                                    "rule": bolt["rule_name"],
                                })
                        else:
                            imp_b += dmg_pts
                            peak_hit_b = max(peak_hit_b, dmg_pts)
                            big_hits_b += 1
                            shots_hit_b += 1
                            if len(highlights) < 18:
                                highlights.append({
                                    "t": round(t_sec, 2),
                                    "event": "B_energy_bolt_direct_hit",
                                    "target": hit_gname,
                                    "impulse": round(imp_mag, 1),
                                    "rule": bolt["rule_name"],
                                })
                        if return_trace_data:
                            shot_traces_2d.append({
                                "owner": bolt["owner"],
                                "p0": [float(bolt["spawn_pos"][0]), float(bolt["spawn_pos"][1])],
                                "p1": [float(hit_pt[0]), float(hit_pt[1])],
                                "outcome": "hit_hull",
                                "rule": bolt["rule_name"],
                                "t": round(t_sec, 2),
                            })

                    foley_events.append({"frame": frame_idx, "kind": "plasma_hit", "strength": 1.35})
                    data.mocap_pos[bolt["mocap_id"]] = (0.0, 0.0, -9.5)
                    continue

                bolt["pos"] = p1
                warp_seg = try_warp_bolt(stage_key, bolt)
                if warp_seg is not None:
                    if return_trace_data and ko_frame is None:
                        shot_traces_2d.append({
                            "owner": bolt["owner"],
                            "p0": [float(bolt["spawn_pos"][0]), float(bolt["spawn_pos"][1])],
                            "p1": warp_seg[0],
                            "outcome": "warped",
                            "rule": bolt["rule_name"],
                            "t": round(t_sec, 2),
                        })
                    bolt["spawn_pos"] = bolt["pos"].copy()
                    foley_events.append({"frame": frame_idx, "kind": "portal_warp", "strength": 0.75})
                data.mocap_pos[bolt["mocap_id"]] = bolt["pos"]
                data.mocap_quat[bolt["mocap_id"]] = _dir_to_quat(bolt["vel"])
                surviving_bolts.append(bolt)

            active_bolts = surviving_bolts

            mujoco.mj_step(model, data)
            np.clip(data.qvel, -42.0, 42.0, out=data.qvel)

            # Inspect physical melee contacts + update grounded state for next substep!
            step_hit_a = 0.0
            step_hit_b = 0.0
            next_grounded_a = False
            next_grounded_b = False

            for ci in range(data.ncon):
                con = data.contact[ci]
                gi1, gi2 = int(con.geom1), int(con.geom2)
                g1 = geom_names[gi1]
                g2 = geom_names[gi2]

                a_involved = g1.startswith("A_") or g2.startswith("A_")
                b_involved = g1.startswith("B_") or g2.startswith("B_")
                env_involved = (g1.startswith("env_") or g1.startswith("trap_") or g2.startswith("env_") or g2.startswith("trap_"))
                is_catch = ("env_catch_floor" in (g1, g2))

                if env_involved and not is_catch:
                    if a_involved:
                        next_grounded_a = True
                    if b_involved:
                        next_grounded_b = True

                is_ab = a_involved and b_involved
                if not (is_ab or is_catch):
                    continue

                mujoco.mj_contactForce(model, data, ci, force6)
                imp = abs(float(force6[0])) * PHYSICS_DT
                if imp < 0.08:
                    continue

                if is_ab:
                    ga = g1 if g1.startswith("A_") else g2
                    gb = g2 if g1.startswith("A_") else g1
                    w_a = _is_weapon_geom(ga)
                    w_b = _is_weapon_geom(gb)
                    if ko_frame is None:
                        step_hit_a += imp * (1.85 if w_a else 0.75)
                        step_hit_b += imp * (1.85 if w_b else 0.75)
                    if imp > frame_best_ab_imp:
                        frame_best_ab_imp = imp
                        frame_best_ab_weapon = (w_a or w_b)
                elif is_catch and imp > 0.6:
                    if a_involved and not crashed_floor_a:
                        crashed_floor_a = True
                        frame_floor_crash = max(frame_floor_crash, imp)
                    if b_involved and not crashed_floor_b:
                        crashed_floor_b = True
                        frame_floor_crash = max(frame_floor_crash, imp)

            grounded_a = next_grounded_a
            grounded_b = next_grounded_b

            if ko_frame is None:
                if step_hit_a > 0.0:
                    imp_a += step_hit_a
                    ctrl_a.record_melee_damage(step_hit_a)
                if step_hit_b > 0.0:
                    imp_b += step_hit_b
                    ctrl_b.record_melee_damage(step_hit_b)
                if step_hit_a > peak_hit_a:
                    peak_hit_a = step_hit_a
                if step_hit_b > peak_hit_b:
                    peak_hit_b = step_hit_b
                if step_hit_a > 4.2:
                    big_hits_a += 1
                    if len(highlights) < 18:
                        highlights.append({
                            "t": round(t_sec, 2),
                            "event": "heavy_melee_hit_by_A",
                            "impulse": round(step_hit_a, 2),
                            "rule": ctrl_a.active_rule_name,
                        })
                if step_hit_b > 4.2:
                    big_hits_b += 1
                    if len(highlights) < 18:
                        highlights.append({
                            "t": round(t_sec, 2),
                            "event": "heavy_melee_hit_by_B",
                            "impulse": round(step_hit_b, 2),
                            "rule": ctrl_b.active_rule_name,
                        })

        if record_replay:
            xpos_frames.append(data.geom_xpos.astype(np.float32).copy())
            xmat_frames.append(data.geom_xmat.astype(np.float32).copy())

        if frame_best_ab_imp > 0.85 and (frame_idx - last_clang_f) >= 3:
            last_clang_f = frame_idx
            foley_events.append({
                "frame": frame_idx,
                "kind": "weapon_clang" if frame_best_ab_weapon else "hull_thud",
                "strength": round(min(2.5, frame_best_ab_imp / 5.0), 3),
            })
        if frame_spring_fired:
            foley_events.append({
                "frame": frame_idx,
                "kind": "trap_clang",
                "strength": 1.45,
            })
        if frame_floor_crash > 0.5:
            foley_events.append({"frame": frame_idx, "kind": "floor_crash", "strength": 1.6})

        if ko_frame is not None:
            if record_replay:
                live_scores_a_frames.append(last_live_a)
                live_scores_b_frames.append(last_live_b)
            continue

        frame_t = (frame_idx + 1) / float(FPS)
        xa, ya, za = float(data.xpos[body_a, 0]), float(data.xpos[body_a, 1]), float(data.xpos[body_a, 2])
        xb, yb, zb = float(data.xpos[body_b, 0]), float(data.xpos[body_b, 1]), float(data.xpos[body_b, 2])

        if sens_a and sens_b:
            min_edge_a = min(min_edge_a, float(sens_a["edge_dist"]))
            min_edge_b = min(min_edge_b, float(sens_b["edge_dist"]))
            max_speed_a = max(max_speed_a, float(sens_a["self_speed"]))
            max_speed_b = max(max_speed_b, float(sens_b["self_speed"]))

            snap_a = {
                "t": round(frame_t, 2),
                "self_xy": [round(xa, 2), round(ya, 2)],
                "self_z": round(za, 2),
                "enemy_xy": [round(xb, 2), round(yb, 2)],
                "dist": round(float(sens_a["enemy_dist"]), 2),
                "active_rule": ctrl_a.active_rule_name,
                "wheel_cmd": [round(ctrl_a.last_wheel_cmd[0], 2), round(ctrl_a.last_wheel_cmd[1], 2)],
                "turret_yaw_deg": round(float(sens_a.get("cannon_yaw_deg", 0.0)), 1),
                "speed_mps": round(float(sens_a["self_speed"]), 2),
                "edge_dist_m": round(float(sens_a["edge_dist"]), 2),
                "spring_dist_m": round(float(sens_a["spring_dist"]), 2),
                "enemy_spring_dist_m": round(float(sens_a["enemy_spring_dist"]), 2),
                "los": bool(sens_a["has_line_of_sight"]),
            }
            recent_states_a.append(snap_a)
            if len(recent_states_a) > 16:
                recent_states_a.pop(0)

            if frame_idx == 0 or (frame_idx + 1) % FPS == 0:
                timeline_1hz_a.append(snap_a)

            if return_trace_data:
                Ra = data.xmat[body_a].reshape(3, 3)
                yawa = math.atan2(float(Ra[1, 0]), float(Ra[0, 0]))
                trace_frames_2d.append({
                    "t": round(frame_t, 3),
                    "xa": xa,
                    "ya": ya,
                    "yawa": yawa,
                    "rule_a": ctrl_a.active_rule_name,
                    "xb": xb,
                    "yb": yb,
                })

            if float(sens_a["tilt_deg"]) < 45.0:
                upright_frames_a += 1
            if float(sens_b["tilt_deg"]) < 45.0:
                upright_frames_b += 1

            cur_tilt_b = float(sens_b["tilt_deg"]) > 70.0
            if cur_tilt_b and not prev_tilted_b and float(sens_a["enemy_dist"]) < 1.85:
                flips_inflicted_a += 1
                highlights.append({"t": round(frame_t, 2), "event": "A_flipped_B", "rule": ctrl_a.active_rule_name})
            prev_tilted_b = cur_tilt_b

            cur_tilt_a = float(sens_a["tilt_deg"]) > 70.0
            if cur_tilt_a and not prev_tilted_a and float(sens_b["enemy_dist"]) < 1.85:
                flips_inflicted_b += 1
                highlights.append({"t": round(frame_t, 2), "event": "B_flipped_A", "rule": ctrl_a.active_rule_name})
            prev_tilted_a = cur_tilt_a

            if float(sens_a["abs_enemy_angle_deg"]) < 45.0 and float(sens_a["self_speed"]) > 0.25:
                aggression_a += 0.6
            if float(sens_a["edge_dist"]) > float(sens_b["edge_dist"]) + 0.20:
                aggression_a += 0.4

            if float(sens_b["abs_enemy_angle_deg"]) < 45.0 and float(sens_b["self_speed"]) > 0.25:
                aggression_b += 0.6
            if float(sens_b["edge_dist"]) > float(sens_a["edge_dist"]) + 0.20:
                aggression_b += 0.4

            if float(sens_a["tilt_deg"]) > 92.0 and float(sens_a["self_speed"]) < 0.12:
                immob_inverted_sec_a += 1.0 / FPS
            else:
                immob_inverted_sec_a = 0.0

            if float(sens_b["tilt_deg"]) > 92.0 and float(sens_b["self_speed"]) < 0.12:
                immob_inverted_sec_b += 1.0 / FPS
            else:
                immob_inverted_sec_b = 0.0

        cur_f = float(frame_idx + 1)
        raw_da = imp_a + flips_inflicted_a * 12.0 + spring_herds_by_a * 8.0
        raw_db = imp_b + flips_inflicted_b * 12.0 + spring_herds_by_b * 8.0
        d_sum = raw_da + raw_db
        s_dmg_a = (50.0 * raw_da / d_sum) if d_sum > 1e-4 else 25.0
        s_dmg_b = (50.0 * raw_db / d_sum) if d_sum > 1e-4 else 25.0
        a_sum = aggression_a + aggression_b
        s_agg_a = (30.0 * aggression_a / a_sum) if a_sum > 1e-4 else 15.0
        s_agg_b = (30.0 * aggression_b / a_sum) if a_sum > 1e-4 else 15.0
        s_stb_a = 20.0 * (upright_frames_a / cur_f)
        s_stb_b = 20.0 * (upright_frames_b / cur_f)
        last_live_a = [round(s_dmg_a + s_agg_a + s_stb_a, 2), round(s_dmg_a, 2), round(s_agg_a, 2), round(s_stb_a, 2)]
        last_live_b = [round(s_dmg_b + s_agg_b + s_stb_b, 2), round(s_dmg_b, 2), round(s_agg_b, 2), round(s_stb_b, 2)]
        if record_replay:
            live_scores_a_frames.append(last_live_a)
            live_scores_b_frames.append(last_live_b)

        # Check Ring-Out KO!
        fell_a = za < -0.45
        fell_b = zb < -0.45
        if fell_a or fell_b:
            if fell_a and fell_b:
                ko_winner = "A" if za > zb else "B"
            elif fell_b:
                ko_winner = "A"
            else:
                ko_winner = "B"
            ko_reason = "ring_out_ko"
            ko_time_sec = round(frame_t, 2)
            ko_frame = frame_idx
            pre_snap = recent_states_a[0] if recent_states_a else None
            end_snap = recent_states_a[-1] if recent_states_a else None
            if end_snap and (not timeline_1hz_a or timeline_1hz_a[-1]["t"] != end_snap["t"]):
                timeline_1hz_a.append(end_snap)

            if ko_winner == "B":
                sp_hint = f" (launched by {last_spring_name_a} shortly before falling)" if (frame_t - last_spring_t_a) < 2.2 and last_spring_name_a else ""
                ko_note_a = (
                    f"RING-OUT LOSS AT t={ko_time_sec}s: Your car went off the cliff at ({xa:.2f}, {ya:.2f}){sp_hint} "
                    f"while running rule {ctrl_a.active_rule_name!r} "
                    f"(0.5s prior -> rule {(pre_snap['active_rule'] if pre_snap else 'unknown')!r}, wheel_cmd={(pre_snap['wheel_cmd'] if pre_snap else None)})."
                )
            else:
                sp_hint = f" (launched by {last_spring_name_b})" if (frame_t - last_spring_t_b) < 2.2 and last_spring_name_b else ""
                ko_note_a = (
                    f"RING-OUT WIN AT t={ko_time_sec}s: Opponent went off the cliff at ({xb:.2f}, {yb:.2f}){sp_hint} "
                    f"while your car ran rule {ctrl_a.active_rule_name!r} at ({xa:.2f}, {ya:.2f})."
                )
            ko_diagnosis = {
                "winner": ko_winner,
                "method": ko_reason,
                "time_sec": ko_time_sec,
                "your_state_0p5s_before_ko": pre_snap,
                "your_state_at_ko": end_snap,
                "summary": ko_note_a,
            }
            highlights.append({"t": ko_time_sec, "event": f"{ko_winner}_wins_by_ring_out_KO", "rule_A": ctrl_a.active_rule_name})
            foley_events.append({"frame": frame_idx, "kind": "ring_out_whoosh", "strength": 1.2})
            if not record_replay:
                break
            continue

        # Check Immobilized-Inverted TKO
        if immob_inverted_sec_b >= 4.0 and immob_inverted_sec_a < 2.0:
            ko_winner = "A"
            ko_reason = "immobilized_tko"
            ko_time_sec = round(frame_t, 2)
            ko_frame = frame_idx
            ko_note_a = f"IMMOBILIZE TKO WIN AT t={ko_time_sec}s: Opponent stayed inverted & immobile >= 4.0s."
            ko_diagnosis = {"winner": "A", "method": ko_reason, "time_sec": ko_time_sec, "summary": ko_note_a}
            highlights.append({"t": ko_time_sec, "event": "A_wins_by_immobilize_TKO"})
            if not record_replay:
                break
            continue
        elif immob_inverted_sec_a >= 4.0 and immob_inverted_sec_b < 2.0:
            ko_winner = "B"
            ko_reason = "immobilized_tko"
            ko_time_sec = round(frame_t, 2)
            ko_frame = frame_idx
            ko_note_a = (
                f"IMMOBILIZE TKO LOSS AT t={ko_time_sec}s: Your car stayed inverted & unable to move >= 4.0s "
                f"(active rule: {ctrl_a.active_rule_name!r}) — consider larger monster_wheel clearance, lower CoM, or flipper/hammer self-righting."
            )
            ko_diagnosis = {"winner": "B", "method": ko_reason, "time_sec": ko_time_sec, "summary": ko_note_a}
            highlights.append({"t": ko_time_sec, "event": "B_wins_by_immobilize_TKO"})
            if not record_replay:
                break
            continue

        if frame_idx + 1 >= MATCH_FRAMES and ko_frame is None:
            break

    active_frames = max(1, (ko_frame + 1) if ko_frame is not None else MATCH_FRAMES)
    match_dur_sec = round(active_frames / float(FPS), 2)

    # BattleBots 100-Point Judges' Scorecard:
    #   - 50 pts: Damage, Flips & Tactical Spring Launches Inflicted
    #   - 30 pts: Aggression & Stage Control
    #   - 20 pts: Control & Upright Stability
    raw_dmg_a = imp_a + flips_inflicted_a * 12.0 + spring_herds_by_a * 8.0
    raw_dmg_b = imp_b + flips_inflicted_b * 12.0 + spring_herds_by_b * 8.0
    dmg_sum = raw_dmg_a + raw_dmg_b
    if dmg_sum > 1e-4:
        score_dmg_a = 50.0 * (raw_dmg_a / dmg_sum)
        score_dmg_b = 50.0 * (raw_dmg_b / dmg_sum)
    else:
        score_dmg_a = 25.0
        score_dmg_b = 25.0

    agg_sum = aggression_a + aggression_b
    if agg_sum > 1e-4:
        score_agg_a = 30.0 * (aggression_a / agg_sum)
        score_agg_b = 30.0 * (aggression_b / agg_sum)
    else:
        score_agg_a = 15.0
        score_agg_b = 15.0

    score_stab_a = 20.0 * (upright_frames_a / float(active_frames))
    score_stab_b = 20.0 * (upright_frames_b / float(active_frames))

    total_score_a = round(score_dmg_a + score_agg_a + score_stab_a, 2)
    total_score_b = round(score_dmg_b + score_agg_b + score_stab_b, 2)

    if ko_winner is not None:
        final_winner = ko_winner
    else:
        if abs(total_score_a - total_score_b) > 0.05:
            final_winner = "A" if total_score_a > total_score_b else "B"
        elif abs(peak_hit_a - peak_hit_b) > 0.01:
            final_winner = "A" if peak_hit_a >= peak_hit_b else "B"
        else:
            dist_c_a = math.hypot(float(data.xpos[body_a, 0]), float(data.xpos[body_a, 1]))
            dist_c_b = math.hypot(float(data.xpos[body_b, 0]), float(data.xpos[body_b, 1]))
            final_winner = "A" if dist_c_a <= dist_c_b else "B"

    wall_ms = round((time.perf_counter() - t0_wall) * 1000.0, 1)
    strat_diag_a = ctrl_a.build_strategy_diagnostics(match_dur_sec, ko_note=ko_note_a)

    result: Dict[str, Any] = {
        "stage_key": stage_key,
        "stage_name_zh": ARENA_META[stage_key]["name_zh"],
        "swap_sides": swap_sides,
        "winner": final_winner,
        "win_method": ko_reason,
        "ko_time_sec": ko_time_sec,
        "duration_sec": match_dur_sec,
        "sim_wall_ms": wall_ms,
        "scores": {
            "A": {
                "total": total_score_a,
                "damage": round(score_dmg_a, 2),
                "aggression": round(score_agg_a, 2),
                "stability": round(score_stab_a, 2),
            },
            "B": {
                "total": total_score_b,
                "damage": round(score_dmg_b, 2),
                "aggression": round(score_agg_b, 2),
                "stability": round(score_stab_b, 2),
            },
        },
        "strategy_diagnostics_A": strat_diag_a,
        "ko_diagnosis": ko_diagnosis,
        "tactical_timeline_A_1hz": timeline_1hz_a,
        "telemetry": {
            "A": {
                "mass_kg": round(mass_a, 2),
                "weapon_damage_impulse": round(imp_a, 2),
                "peak_single_hit": round(peak_hit_a, 2),
                "heavy_hits_landed": big_hits_a,
                "flips_inflicted": flips_inflicted_a,
                "energy_shots_fired": shots_fired_a,
                "energy_shots_hit": shots_hit_a,
                "spring_launches_taken": spring_launches_a,
                "enemy_herded_onto_spring": spring_herds_by_a,
                "min_edge_clearance_m": round(min_edge_a, 3),
                "max_speed_mps": round(max_speed_a, 2),
                "final_pos_xyz": [round(float(x), 3) for x in data.xpos[body_a]],
                "top_rules_triggered": dict(sorted(ctrl_a.rule_trigger_counts.items(), key=lambda kv: -kv[1])[:5]),
            },
            "B": {
                "mass_kg": round(mass_b, 2),
                "weapon_damage_impulse": round(imp_b, 2),
                "peak_single_hit": round(peak_hit_b, 2),
                "heavy_hits_landed": big_hits_b,
                "flips_inflicted": flips_inflicted_b,
                "energy_shots_fired": shots_fired_b,
                "energy_shots_hit": shots_hit_b,
                "spring_launches_taken": spring_launches_b,
                "enemy_herded_onto_spring": spring_herds_by_b,
                "min_edge_clearance_m": round(min_edge_b, 3),
                "max_speed_mps": round(max_speed_b, 2),
                "final_pos_xyz": [round(float(x), 3) for x in data.xpos[body_b]],
                "top_rules_triggered": dict(sorted(ctrl_b.rule_trigger_counts.items(), key=lambda kv: -kv[1])[:5]),
            },
        },
        "highlights": highlights[:18],
        "foley_events": foley_events,
    }

    if return_trace_data:
        result["_trace"] = {
            "stage_key": stage_key,
            "stage_name_zh": ARENA_META[stage_key]["name_zh"],
            "car_a_name": spec_a.get("meta", {}).get("car_name", "Car A"),
            "car_b_name": spec_b.get("meta", {}).get("car_name", "Car B"),
            "winner": final_winner,
            "win_method": ko_reason,
            "duration_sec": match_dur_sec,
            "frames": trace_frames_2d,
            "shots": shot_traces_2d,
            "rule_coverage": strat_diag_a["rule_coverage"],
        }

    if record_replay:
        geom_info: List[Dict[str, Any]] = []
        styles = scene["geom_styles"]
        for gi, gname in enumerate(geom_names):
            owner = "A" if gname.startswith("A_") else ("B" if gname.startswith("B_") else "env")
            geom_info.append({
                "id": gi,
                "name": gname,
                "owner": owner,
                "type": int(model.geom_type[gi]),
                "size": [round(float(x), 5) for x in model.geom_size[gi]],
                "rgba": [round(float(x), 4) for x in model.geom_rgba[gi]],
                "style": styles.get(gname, {}),
            })
        result["_replay"] = {
            "fps": FPS,
            "ko_frame": ko_frame,
            "num_frames": len(xpos_frames),
            "stage_key": stage_key,
            "car_a_meta": spec_a.get("meta", {}),
            "car_b_meta": spec_b.get("meta", {}),
            "geom_info": geom_info,
            "xpos": np.stack(xpos_frames, axis=0),
            "xmat": np.stack(xmat_frames, axis=0),
            "live_scores_a": np.array(live_scores_a_frames, dtype=np.float32),
            "live_scores_b": np.array(live_scores_b_frames, dtype=np.float32),
        }

    return result


def save_replay_bundle(match_result: Dict[str, Any], out_prefix: Path) -> Path:
    """Save 30fps replay geometry poses (.npz) + scene metadata (.json) for Blender 4.2 OPTIX rendering."""
    rep = match_result.pop("_replay")
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    npz_path = out_prefix.with_suffix(".npz")
    json_path = out_prefix.with_suffix(".json")

    np.savez_compressed(
        npz_path,
        xpos=rep["xpos"],
        xmat=rep["xmat"],
        live_scores_a=rep["live_scores_a"],
        live_scores_b=rep["live_scores_b"],
    )
    meta_doc = {
        "fps": rep["fps"],
        "ko_frame": rep["ko_frame"],
        "num_frames": rep["num_frames"],
        "stage_key": rep["stage_key"],
        "car_a_meta": rep["car_a_meta"],
        "car_b_meta": rep["car_b_meta"],
        "match": match_result,
        "geom_info": rep["geom_info"],
    }
    json_path.write_text(json.dumps(meta_doc, ensure_ascii=False, indent=2), encoding="utf-8")
    return npz_path


def run_bo3_series(
    spec_a: Dict[str, Any],
    spec_b: Dict[str, Any],
    play_all_three: bool = False,
    record_replays_dir: Optional[Path] = None,
    series_tag: str = "bo3",
    return_trace_data: bool = False,
) -> Dict[str, Any]:
    """
    Run an official Best-of-3 across the 3 distinct slope & tactical-spring arenas:
      Game 1: spring_arena    (normal spawn)
      Game 2: camelback_ridge (swapped spawn sides for symmetry fairness!)
      Game 3: turbo_bowl      (decider)
    """
    wins_a = 0
    wins_b = 0
    games: List[Dict[str, Any]] = []

    for idx, stage_key in enumerate(ARENA_ORDER):
        swap = (idx == 1)
        need_rep = record_replays_dir is not None
        mres = run_single_match(
            spec_a,
            spec_b,
            stage=stage_key,
            swap_sides=swap,
            record_replay=need_rep,
            return_trace_data=return_trace_data,
        )
        if need_rep and record_replays_dir is not None:
            save_replay_bundle(mres, record_replays_dir / f"{series_tag}_G{idx + 1}_{stage_key}")

        if mres["winner"] == "A":
            wins_a += 1
        else:
            wins_b += 1
        games.append(mres)

        if not play_all_three and (wins_a == 2 or wins_b == 2):
            break

    series_winner = "A" if wins_a > wins_b else "B"
    return {
        "series_winner": series_winner,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "series_score": f"{wins_a}-{wins_b}",
        "games": games,
    }
