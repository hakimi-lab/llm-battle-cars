"""
100% Model-Authored DIY Strategy & Actuator Controller for LLM Battle-Cars (v3.1).

Zero pre-baked driving autopilots ('pursue', 'dodge_bolt', 'herd_to_spring', 'retreat_to_safe' do NOT exist).
Contestant JSONs compute exact differential wheel commands ('left', 'right') and weapon triggers at 120Hz
using safe pure-math/boolean AST formulas, 'let' bindings, stateful 'vars', and prioritized 'rules' over raw physical sensors!
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import mujoco
import numpy as np

from .arenas import ARENA_META, query_arena_sensors, resolve_stage_key
from .catalog import VALID_ALL_SENSORS, clamp
from .expr import BASE_EVAL_ENV, validate_and_compile_expr


def _local_angle_deg(fwd_xy: Tuple[float, float], left_xy: Tuple[float, float], dir_xy: Tuple[float, float]) -> float:
    """Signed angle in degrees from +X nose (viewed from +Z top): 0 = ahead, + = CCW (left when upright), - = CW."""
    dot_f = fwd_xy[0] * dir_xy[0] + fwd_xy[1] * dir_xy[1]
    dot_l = left_xy[0] * dir_xy[0] + left_xy[1] * dir_xy[1]
    return math.degrees(math.atan2(dot_l, dot_f))


def _compile_when_predicate(when: Any, allowed_vars: Set[str], ctx: str) -> Callable[[Dict[str, Any]], bool]:
    if when is None or (isinstance(when, dict) and len(when) == 0):
        return lambda _env: True
    if isinstance(when, (str, bool)):
        fn, _ = validate_and_compile_expr(when, allowed_vars, f"{ctx}.when")
        return lambda env, f=fn: bool(f(env))

    if not isinstance(when, dict):
        return lambda _env: False

    checks: List[Callable[[Dict[str, Any]], bool]] = []
    for key, val in when.items():
        if key == "stage_in" and isinstance(val, (list, tuple)):
            allowed_stages = tuple(int(x) for x in val)
            checks.append(lambda env, st=allowed_stages: int(env["stage_id"]) in st)
            continue
        if key == "expr":
            fn_e, _ = validate_and_compile_expr(val, allowed_vars, f"{ctx}.when['expr']")
            checks.append(lambda env, f=fn_e: bool(f(env)))
            continue
        if key.endswith("_between") and isinstance(val, (list, tuple)) and len(val) == 2:
            sname = key[: -len("_between")]
            lo, hi = float(val[0]), float(val[1])
            checks.append(lambda env, k=sname, a=lo, b=hi: a <= float(env.get(k, 0.0)) <= b)
            continue
        matched_suf = False
        for suf, op in (
            ("_lte", lambda a, b: a <= b),
            ("_lt",  lambda a, b: a < b),
            ("_gte", lambda a, b: a >= b),
            ("_gt",  lambda a, b: a > b),
        ):
            if key.endswith(suf):
                sname = key[: -len(suf)]
                thr = float(val)
                checks.append(lambda env, k=sname, f_op=op, t=thr: f_op(float(env.get(k, 0.0)), t))
                matched_suf = True
                break
        if matched_suf:
            continue
        if isinstance(val, bool):
            checks.append(lambda env, k=key, bv=val: bool(env.get(k, False)) == bv)
        else:
            checks.append(lambda env, k=key, vv=val: env.get(k) == vv)

    return lambda env, cs=checks: all(c(env) for c in cs)


def _compile_action_evaluators(
    act_dict: Dict[str, Any],
    allowed_vars: Set[str],
    ctx: str,
) -> Dict[str, Callable[[Dict[str, Any]], Any]]:
    compiled: Dict[str, Callable[[Dict[str, Any]], Any]] = {}
    for k, v in act_dict.items():
        if k == "drive":
            continue
        fn, _ = validate_and_compile_expr(v, allowed_vars, f"{ctx}[{k!r}]")
        compiled[k] = fn
    return compiled


class VehicleController:
    """Stateful 120Hz controller executing contestant-authored AST wheel & weapon formulas."""

    def __init__(
        self,
        prefix: str,
        spec: Dict[str, Any],
        stage_key: str,
        model: mujoco.MjModel,
        car_bundle: Dict[str, Any],
        enemy_prefix: str,
        hazard_geom_ids: List[int],
    ) -> None:
        self.prefix = prefix
        self.enemy_prefix = enemy_prefix
        self.stage_key = stage_key
        self.stage_id = int(ARENA_META[stage_key]["stage_id"])
        self.model = model
        self.hazard_geom_ids = hazard_geom_ids

        raw_strat = spec.get("strategy", {}) or {}
        raw_ov_map = raw_strat.get("stage_overrides", {}) or {}
        overrides: Dict[str, Any] = {}
        for k_ov, v_ov in raw_ov_map.items():
            try:
                if resolve_stage_key(k_ov) == stage_key and isinstance(v_ov, dict):
                    overrides = v_ov
                    break
            except ValueError:
                continue

        # 1. Persistent state variables ('vars')
        self.state_vars: Dict[str, Any] = {}
        if isinstance(raw_strat.get("vars"), dict):
            self.state_vars.update(raw_strat["vars"])
        if isinstance(overrides.get("vars"), dict):
            self.state_vars.update(overrides["vars"])

        allowed_names: Set[str] = set(VALID_ALL_SENSORS) | set(self.state_vars.keys())

        # 2. Ordered per-tick helper formulas ('let')
        merged_let: Dict[str, Any] = {}
        if isinstance(raw_strat.get("let"), dict):
            merged_let.update(raw_strat["let"])
        if isinstance(overrides.get("let"), dict):
            merged_let.update(overrides["let"])

        self.compiled_lets: List[Tuple[str, Callable[[Dict[str, Any]], Any]]] = []
        for lk, lexpr in merged_let.items():
            fn_l, _ = validate_and_compile_expr(lexpr, allowed_names, f"let[{lk!r}]")
            self.compiled_lets.append((lk, fn_l))
            allowed_names.add(lk)

        # 3. Prioritized rules ('rules') + 'default_action'
        raw_rules: List[Dict[str, Any]] = list(overrides.get("rules", raw_strat.get("rules", [])) or [])
        raw_default: Dict[str, Any] = dict(
            overrides.get("default_action", raw_strat.get("default_action", {"left": 0.0, "right": 0.0})) or {}
        )

        self.compiled_rules: List[Dict[str, Any]] = []
        self.rule_records: Dict[str, Dict[str, Any]] = {}

        for idx, r in enumerate(raw_rules):
            if not isinstance(r, dict):
                continue
            rname = str(r.get("name", f"rule_{idx}")).strip()
            slot_key = f"#{idx}:{rname}"
            when_raw = r.get("when", {})
            pred_fn = _compile_when_predicate(when_raw, allowed_names, f"rules[{idx}]")

            local_allowed = set(allowed_names)
            set_vars_compiled: List[Tuple[str, Callable[[Dict[str, Any]], Any]]] = []
            if isinstance(r.get("set_vars"), dict):
                for svk, sv_expr in r["set_vars"].items():
                    fn_sv, _ = validate_and_compile_expr(sv_expr, local_allowed, f"rules[{idx}].set_vars[{svk!r}]")
                    set_vars_compiled.append((svk, fn_sv))
                    local_allowed.add(svk)

            act_raw = r.get("action") if isinstance(r.get("action"), dict) else {}
            act_compiled = _compile_action_evaluators(act_raw, local_allowed, f"rules[{idx}].action")

            self.compiled_rules.append({
                "index": idx,
                "name": rname,
                "slot": slot_key,
                "pred": pred_fn,
                "set_vars": set_vars_compiled,
                "act": act_compiled,
            })
            self.rule_records[slot_key] = {
                "order": idx,
                "name": rname,
                "when": when_raw,
                "wheel_expr": {"left": act_raw.get("left", 0.0), "right": act_raw.get("right", 0.0)},
                "ticks": 0,
                "active_sec": 0.0,
                "entries": 0,
                "first_t": None,
                "last_t": None,
                "distance_driven_m": 0.0,
                "melee_damage_dealt": 0.0,
                "shots_fired": 0,
                "shots_hit_hull": 0,
                "shots_blocked_by_terrain": 0,
                "shots_missed": 0,
                "spring_launches_taken": 0,
            }

        self.compiled_default = _compile_action_evaluators(raw_default, allowed_names, "default_action")
        self.rule_records["default"] = {
            "order": len(self.compiled_rules),
            "name": "default_action",
            "when": "fallback_when_no_rule_matches",
            "wheel_expr": {"left": raw_default.get("left", 0.0), "right": raw_default.get("right", 0.0)},
            "ticks": 0,
            "active_sec": 0.0,
            "entries": 0,
            "first_t": None,
            "last_t": None,
            "distance_driven_m": 0.0,
            "melee_damage_dealt": 0.0,
            "shots_fired": 0,
            "shots_hit_hull": 0,
            "shots_blocked_by_terrain": 0,
            "shots_missed": 0,
            "spring_launches_taken": 0,
        }

        self.root_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}_root")
        self.enemy_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{enemy_prefix}_root")

        self.wheel_acts: List[Tuple[int, str]] = []
        self.spinner_acts: List[Tuple[int, float]] = []
        self.flipper_acts: List[int] = []
        self.hammer_acts: List[int] = []
        self.piston_acts: List[Tuple[int, float]] = []

        for am in car_bundle["actuators_meta"]:
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, am["name"])
            if aid < 0:
                continue
            kind = am["kind"]
            if kind == "wheel":
                self.wheel_acts.append((aid, am["channel"]))
            elif kind == "spinner":
                self.spinner_acts.append((aid, float(am["spin_dir"])))
            elif kind == "flipper":
                self.flipper_acts.append(aid)
            elif kind == "hammer":
                self.hammer_acts.append(aid)
            elif kind == "piston":
                self.piston_acts.append((aid, float(am["stroke"])))

        self.thrusters: List[Dict[str, Any]] = []
        for tm in car_bundle["thrusters_meta"]:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, tm["body_name"])
            faid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, tm["flame_act_name"])
            if bid >= 0:
                self.thrusters.append({
                    "body_id": bid,
                    "flame_act_id": faid,
                    "force_local": np.array(tm["force_local"], dtype=np.float64),
                    "max_force_n": float(tm["max_force_n"]),
                })

        self.cannons: List[Dict[str, Any]] = []
        for cm in car_bundle.get("cannons_meta", []):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cm["body_name"])
            tur_aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, cm.get("turret_act_name", ""))
            mocap_ids = []
            for bb_name in cm["bolt_body_names"]:
                bb_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bb_name)
                if bb_id >= 0 and int(model.body_mocapid[bb_id]) >= 0:
                    mocap_ids.append(int(model.body_mocapid[bb_id]))
            if bid >= 0 and mocap_ids:
                self.cannons.append({
                    "body_id": bid,
                    "turret_act_id": tur_aid,
                    "mode": cm["mode"],
                    "cooldown": float(cm["cooldown"]),
                    "bolt_speed": float(cm["bolt_speed"]),
                    "bolt_impulse": float(cm["bolt_impulse"]),
                    "recoil_impulse": float(cm["recoil_impulse"]),
                    "bolt_damage": float(cm["bolt_damage"]),
                    "muzzle_local": np.array(cm["muzzle_local"], dtype=np.float64),
                    "dir_local": np.array(cm.get("dir_local", [1.0, 0.0, 0.0]), dtype=np.float64),
                    "mocap_ids": mocap_ids,
                    "ready_t": 0.15,
                    "slot_idx": 0,
                })

        self._pending_shots: List[Dict[str, Any]] = []

        self.flipper_ready_t = 0.0
        self.flipper_active_until = -1.0
        self.hammer_ready_t = 0.0
        self.hammer_active_until = -1.0
        self.piston_ready_t = 0.0
        self.piston_active_until = -1.0
        self.thruster_fuel = 1.80
        self.stalled_sec = 0.0

        self.active_rule_name = "default_action"
        self.active_rule_slot = "default"
        self.last_wheel_cmd: Tuple[float, float] = (0.0, 0.0)
        self._prev_rule_slot: Optional[str] = None
        self.rule_trigger_counts: Dict[str, int] = {}

        self.actuator_exec: Dict[str, Any] = {
            "flipper_fires": 0,
            "hammer_slams": 0,
            "piston_punches": 0,
            "thruster_burn_sec": 0.0,
            "spinner_active_sec": 0.0,
            "turret_slewed_deg": 0.0,
            "total_stalled_sec": 0.0,
            "total_inverted_sec": 0.0,
            "total_distance_m": 0.0,
            "mud_time_sec": 0.0,
            "ice_time_sec": 0.0,
            "spring_launches_self": 0,
            "spring_launches_enemy": 0,
            "portal_warps_self": 0,
            "portal_warps_enemy": 0,
        }

        self._ray_geomid = np.zeros(1, dtype=np.int32)

    def pop_fired_bolts(self) -> List[Dict[str, Any]]:
        out = self._pending_shots
        self._pending_shots = []
        return out

    def compute_sensors(
        self,
        data: mujoco.MjData,
        t_sec: float,
        dt: float,
        enemy_bolts: Optional[List[Dict[str, Any]]] = None,
        hazard_state: Optional[Dict[str, Dict[str, float]]] = None,
        self_mud_factor: Optional[float] = None,
        enemy_mud_factor: Optional[float] = None,
    ) -> Dict[str, Any]:
        pos = data.xpos[self.root_body_id]
        R = data.xmat[self.root_body_id].reshape(3, 3)
        vel6 = data.cvel[self.root_body_id]
        ang_v = vel6[0:3]
        lin_v = vel6[3:6]

        epos = data.xpos[self.enemy_body_id]
        eR = data.xmat[self.enemy_body_id].reshape(3, 3)
        e_lin_v = data.cvel[self.enemy_body_id][3:6]

        up_vec = R[:, 2]
        tilt_deg = math.degrees(math.acos(clamp(float(up_vec[2]), -1.0, 1.0)))
        is_inverted = tilt_deg > 108.0

        # Consistent horizontal projection of +X nose; left_xy is +90 deg CCW viewed from top (+Z)
        fx, fy = float(R[0, 0]), float(R[1, 0])
        fn = math.hypot(fx, fy)
        fwd_xy = (fx / fn, fy / fn) if fn > 1e-5 else (1.0, 0.0)
        left_xy = (-fwd_xy[1], fwd_xy[0])
        self_yaw_deg = math.degrees(math.atan2(fwd_xy[1], fwd_xy[0]))

        efx, efy = float(eR[0, 0]), float(eR[1, 0])
        efn = math.hypot(efx, efy)
        enemy_yaw_deg = math.degrees(math.atan2(efy, efx)) if efn > 1e-5 else 0.0

        sx_w, sy_w, sz_w = float(pos[0]), float(pos[1]), float(pos[2])
        ex_w, ey_w, ez_w = float(epos[0]), float(epos[1]), float(epos[2])

        dx, dy = ex_w - sx_w, ey_w - sy_w
        enemy_dist = math.hypot(dx, dy)
        edir_xy = (dx / enemy_dist, dy / enemy_dist) if enemy_dist > 1e-5 else fwd_xy
        enemy_angle_deg = _local_angle_deg(fwd_xy, left_xy, edir_xy)

        vx_s, vy_s = float(lin_v[0]), float(lin_v[1])
        evx, evy = float(e_lin_v[0]), float(e_lin_v[1])
        rel_vx, rel_vy = vx_s - evx, vy_s - evy
        closing_speed = rel_vx * edir_xy[0] + rel_vy * edir_xy[1]
        self_speed = math.hypot(vx_s, vy_s)
        self_forward_speed = vx_s * fwd_xy[0] + vy_s * fwd_xy[1]
        self_yaw_rate_deg = math.degrees(float(ang_v[2]))
        enemy_speed = math.hypot(evx, evy)

        e_tilt_deg = math.degrees(math.acos(clamp(float(eR[2, 2]), -1.0, 1.0)))
        enemy_is_tilted = (e_tilt_deg > 38.0) or (ez_w > 0.45)

        arena_q = query_arena_sensors(
            self.stage_key, sx_w, sy_w, t_sec, data, self.hazard_geom_ids,
            enemy_xy=(ex_w, ey_w), hazard_state=hazard_state,
            self_mud_factor=self_mud_factor, enemy_mud_factor=enemy_mud_factor,
        )
        enemy_arena_q = query_arena_sensors(
            self.stage_key, ex_w, ey_w, t_sec, data, self.hazard_geom_ids,
            hazard_state=hazard_state,
        )

        edge_angle_deg = _local_angle_deg(fwd_xy, left_xy, arena_q["edge_dir"])
        center_dist = math.hypot(sx_w, sy_w)
        cdir_xy = (-sx_w / center_dist, -sy_w / center_dist) if center_dist > 1e-4 else (-arena_q["edge_dir"][0], -arena_q["edge_dir"][1])
        center_angle_deg = _local_angle_deg(fwd_xy, left_xy, cdir_xy)
        spring_angle_deg = _local_angle_deg(fwd_xy, left_xy, arena_q["spring_dir"])
        mud_angle_deg = _local_angle_deg(fwd_xy, left_xy, arena_q["mud_dir"])
        ice_angle_deg = _local_angle_deg(fwd_xy, left_xy, arena_q["ice_dir"])
        portal_angle_deg = _local_angle_deg(fwd_xy, left_xy, arena_q["portal_dir"])

        cannon_ready = any(t_sec >= c["ready_t"] for c in self.cannons)
        if self.cannons:
            c0 = self.cannons[0]
            c0_R = data.xmat[c0["body_id"]].reshape(3, 3)
            cd0 = c0_R @ c0["dir_local"]
            cfn = math.hypot(float(cd0[0]), float(cd0[1]))
            cfwd_xy = (float(cd0[0]) / cfn, float(cd0[1]) / cfn) if cfn > 1e-5 else fwd_xy
            cleft_xy = (-cfwd_xy[1], cfwd_xy[0])
            cannon_world_yaw_deg = math.degrees(math.atan2(cfwd_xy[1], cfwd_xy[0]))
            cannon_yaw_deg = _local_angle_deg(fwd_xy, left_xy, cfwd_xy)
            cpx, cpy = float(data.xpos[c0["body_id"]][0]), float(data.xpos[c0["body_id"]][1])
            cdx, cdy = ex_w - cpx, ey_w - cpy
            cd_en = math.hypot(cdx, cdy)
            cedir_xy = (cdx / cd_en, cdy / cd_en) if cd_en > 1e-5 else edir_xy
            cannon_enemy_angle_deg = _local_angle_deg(cfwd_xy, cleft_xy, cedir_xy)
        else:
            cannon_world_yaw_deg = self_yaw_deg
            cannon_yaw_deg = 0.0
            cannon_enemy_angle_deg = enemy_angle_deg

        p_eye = np.array([sx_w, sy_w, sz_w + 0.14], dtype=np.float64)
        v_to_e = np.array([ex_w, ey_w, ez_w + 0.12], dtype=np.float64) - p_eye
        dist_3d = float(np.linalg.norm(v_to_e))
        has_los = True
        if dist_3d > 0.15:
            ray_d = mujoco.mj_ray(
                self.model, data, p_eye, v_to_e / dist_3d, None, 1, self.root_body_id, self._ray_geomid
            )
            if 0.0 < ray_d < (dist_3d - 0.35):
                hit_gid = int(self._ray_geomid[0])
                if hit_gid >= 0:
                    gname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, hit_gid) or ""
                    if gname.startswith("env_") or gname.startswith("trap_"):
                        has_los = False

        inc_dist = 9.9
        inc_ang = 0.0
        inc_vx = 0.0
        inc_vy = 0.0
        if enemy_bolts:
            for b in enemy_bolts:
                bx, by = float(b["pos"][0]) - sx_w, float(b["pos"][1]) - sy_w
                bd = math.hypot(bx, by)
                if bd < inc_dist and bd > 1e-5:
                    inc_dist = bd
                    inc_ang = _local_angle_deg(fwd_xy, left_xy, (bx / bd, by / bd))
                    inc_vx = float(b["vel"][0])
                    inc_vy = float(b["vel"][1])

        if self_speed < 0.14:
            self.stalled_sec += dt
        else:
            self.stalled_sec = max(0.0, self.stalled_sec - dt * 2.5)

        return {
            "self_x": round(sx_w, 4),
            "self_y": round(sy_w, 4),
            "self_z": round(sz_w, 4),
            "self_yaw_deg": round(self_yaw_deg, 2),
            "self_speed": round(self_speed, 3),
            "self_forward_speed": round(self_forward_speed, 3),
            "self_yaw_rate_deg": round(self_yaw_rate_deg, 2),
            "tilt_deg": round(tilt_deg, 2),
            "is_inverted": is_inverted,
            "stalled_sec": round(self.stalled_sec, 3),

            "enemy_x": round(ex_w, 4),
            "enemy_y": round(ey_w, 4),
            "enemy_z": round(ez_w, 4),
            "enemy_yaw_deg": round(enemy_yaw_deg, 2),
            "enemy_vx": round(evx, 3),
            "enemy_vy": round(evy, 3),
            "enemy_speed": round(enemy_speed, 3),
            "enemy_closing_speed": round(closing_speed, 3),
            "enemy_dist": round(enemy_dist, 4),
            "enemy_angle_deg": round(enemy_angle_deg, 2),
            "abs_enemy_angle_deg": round(abs(enemy_angle_deg), 2),
            "enemy_is_tilted": enemy_is_tilted,
            "enemy_edge_dist": round(float(enemy_arena_q["edge_dist"]), 4),

            "edge_dist": round(float(arena_q["edge_dist"]), 4),
            "edge_angle_deg": round(edge_angle_deg, 2),
            "abs_edge_angle_deg": round(abs(edge_angle_deg), 2),
            "center_dist": round(center_dist, 4),
            "center_angle_deg": round(center_angle_deg, 2),
            "abs_center_angle_deg": round(abs(center_angle_deg), 2),

            "spring_x": round(float(arena_q["spring_x"]), 4),
            "spring_y": round(float(arena_q["spring_y"]), 4),
            "spring_dist": round(float(arena_q["spring_dist"]), 4),
            "spring_angle_deg": round(spring_angle_deg, 2),
            "abs_spring_angle_deg": round(abs(spring_angle_deg), 2),
            "spring_fling_yaw_deg": round(float(arena_q["spring_fling_yaw_deg"]), 2),
            "spring_ready": bool(arena_q["spring_ready"]),
            "enemy_spring_x": round(float(arena_q["enemy_spring_x"]), 4),
            "enemy_spring_y": round(float(arena_q["enemy_spring_y"]), 4),
            "enemy_spring_dist": round(float(arena_q["enemy_spring_dist"]), 4),
            "enemy_spring_fling_yaw_deg": round(float(arena_q["enemy_spring_fling_yaw_deg"]), 2),
            "enemy_spring_ready": bool(arena_q["enemy_spring_ready"]),

            "in_mud": bool(arena_q["in_mud"]),
            "enemy_in_mud": bool(arena_q["enemy_in_mud"]),
            "mud_x": round(float(arena_q["mud_x"]), 4),
            "mud_y": round(float(arena_q["mud_y"]), 4),
            "mud_dist": round(float(arena_q["mud_dist"]), 4),
            "mud_angle_deg": round(mud_angle_deg, 2),
            "abs_mud_angle_deg": round(abs(mud_angle_deg), 2),

            "on_ice": bool(arena_q["on_ice"]),
            "enemy_on_ice": bool(arena_q["enemy_on_ice"]),
            "ice_dist": round(float(arena_q["ice_dist"]), 4),
            "ice_angle_deg": round(ice_angle_deg, 2),
            "abs_ice_angle_deg": round(abs(ice_angle_deg), 2),

            "portal_x": round(float(arena_q["portal_x"]), 4),
            "portal_y": round(float(arena_q["portal_y"]), 4),
            "portal_dist": round(float(arena_q["portal_dist"]), 4),
            "portal_angle_deg": round(portal_angle_deg, 2),
            "abs_portal_angle_deg": round(abs(portal_angle_deg), 2),
            "portal_out_x": round(float(arena_q["portal_out_x"]), 4),
            "portal_out_y": round(float(arena_q["portal_out_y"]), 4),
            "portal_out_yaw_deg": round(float(arena_q["portal_out_yaw_deg"]), 2),
            "portal_ready": bool(arena_q["portal_ready"]),
            "enemy_portal_x": round(float(arena_q["enemy_portal_x"]), 4),
            "enemy_portal_y": round(float(arena_q["enemy_portal_y"]), 4),
            "enemy_portal_dist": round(float(arena_q["enemy_portal_dist"]), 4),
            "enemy_portal_out_yaw_deg": round(float(arena_q["enemy_portal_out_yaw_deg"]), 2),
            "enemy_portal_ready": bool(arena_q["enemy_portal_ready"]),

            "cannon_yaw_deg": round(cannon_yaw_deg, 2),
            "cannon_world_yaw_deg": round(cannon_world_yaw_deg, 2),
            "cannon_enemy_angle_deg": round(cannon_enemy_angle_deg, 2),
            "abs_cannon_enemy_angle_deg": round(abs(cannon_enemy_angle_deg), 2),

            "has_line_of_sight": has_los,
            "cannon_ready": cannon_ready,
            "incoming_bolt_dist": round(inc_dist, 3),
            "incoming_bolt_angle_deg": round(inc_ang, 2),
            "abs_incoming_bolt_angle_deg": round(abs(inc_ang), 2),
            "incoming_bolt_vx": round(inc_vx, 3),
            "incoming_bolt_vy": round(inc_vy, 3),

            "thruster_fuel": round(self.thruster_fuel, 3),
            "time_sec": round(t_sec, 3),
            "dt": round(dt, 5),
            "stage_id": self.stage_id,
        }

    def step(
        self,
        data: mujoco.MjData,
        t_sec: float,
        dt: float,
        enemy_bolts: Optional[List[Dict[str, Any]]] = None,
        hazard_state: Optional[Dict[str, Dict[str, float]]] = None,
        self_mud_factor: float = 0.0,
        enemy_mud_factor: float = 0.0,
        record_stats: bool = True,
    ) -> Dict[str, Any]:
        sensors = self.compute_sensors(
            data, t_sec, dt,
            enemy_bolts=enemy_bolts, hazard_state=hazard_state,
            self_mud_factor=self_mud_factor, enemy_mud_factor=enemy_mud_factor,
        )

        # Build evaluation environment: base math funcs + raw sensors + persistent state_vars + ordered 'let' bindings
        env: Dict[str, Any] = dict(BASE_EVAL_ENV)
        env.update(sensors)
        env.update(self.state_vars)
        for lk, lfn in self.compiled_lets:
            env[lk] = lfn(env)

        chosen_act = self.compiled_default
        chosen_name = "default_action"
        chosen_slot = "default"

        for r in self.compiled_rules:
            if r["pred"](env):
                for svk, svfn in r["set_vars"]:
                    val_sv = svfn(env)
                    self.state_vars[svk] = val_sv
                    env[svk] = val_sv
                chosen_act = r["act"]
                chosen_name = r["name"]
                chosen_slot = r["slot"]
                break

        self.active_rule_name = chosen_name
        self.active_rule_slot = chosen_slot

        # 1. Direct model-authored differential wheel motor commands!
        left_cmd = clamp(float(chosen_act["left"](env)) if "left" in chosen_act else 0.0, -1.0, 1.0)
        right_cmd = clamp(float(chosen_act["right"](env)) if "right" in chosen_act else 0.0, -1.0, 1.0)
        center_cmd = (
            clamp(float(chosen_act["center"](env)), -1.0, 1.0)
            if "center" in chosen_act
            else 0.5 * (left_cmd + right_cmd)
        )
        self.last_wheel_cmd = (left_cmd, right_cmd)

        # Uniform 50% mud speed factor (keeps 100% breakaway torque at v≈0 so cars never dead-stall!)
        mud_scale = 1.0
        if self_mud_factor > 1e-4:
            mud_scale = 1.0 - 0.39 * self_mud_factor * clamp(sensors["self_speed"] / 0.55, 0.0, 1.0)

        for aid, chan in self.wheel_acts:
            if chan == "left":
                data.ctrl[aid] = left_cmd * mud_scale
            elif chan == "right":
                data.ctrl[aid] = right_cmd * mud_scale
            else:
                data.ctrl[aid] = center_cmd * mud_scale

        if record_stats:
            self.rule_trigger_counts[chosen_name] = self.rule_trigger_counts.get(chosen_name, 0) + 1
            rec = self.rule_records.get(chosen_slot)
            if rec is not None:
                rec["ticks"] += 1
                rec["active_sec"] += dt
                if self._prev_rule_slot != chosen_slot:
                    rec["entries"] += 1
                if rec["first_t"] is None:
                    rec["first_t"] = round(t_sec, 2)
                rec["last_t"] = round(t_sec, 2)
                rec["distance_driven_m"] += sensors["self_speed"] * dt
            self._prev_rule_slot = chosen_slot

            self.actuator_exec["total_distance_m"] += sensors["self_speed"] * dt
            if sensors["self_speed"] < 0.14:
                self.actuator_exec["total_stalled_sec"] += dt
            if sensors["is_inverted"]:
                self.actuator_exec["total_inverted_sec"] += dt
            if sensors["in_mud"]:
                self.actuator_exec["mud_time_sec"] += dt
            if sensors["on_ice"]:
                self.actuator_exec["ice_time_sec"] += dt

        # 2. Spinners (defaults to 1.0 if omitted when spinner installed)
        spin_pwr = clamp(float(chosen_act["spinner"](env)) if "spinner" in chosen_act else 1.0, -1.0, 1.0)
        for aid, sdir in self.spinner_acts:
            data.ctrl[aid] = spin_pwr * sdir
        if record_stats and self.spinner_acts and abs(spin_pwr) > 0.05:
            self.actuator_exec["spinner_active_sec"] += dt

        # 3. Pneumatic Flipper burst + auto-retract
        want_flip = bool(chosen_act["flipper"](env)) if "flipper" in chosen_act else False
        if want_flip and t_sec >= self.flipper_ready_t:
            self.flipper_active_until = t_sec + 0.28
            self.flipper_ready_t = t_sec + 1.30
            if record_stats and self.flipper_acts:
                self.actuator_exec["flipper_fires"] += 1
        flip_pos = 1.26 if t_sec < self.flipper_active_until else 0.0
        for aid in self.flipper_acts:
            data.ctrl[aid] = flip_pos

        # 4. Overhead Hammer slam + auto-cock
        want_ham = bool(chosen_act["hammer"](env)) if "hammer" in chosen_act else False
        if want_ham and t_sec >= self.hammer_ready_t:
            self.hammer_active_until = t_sec + 0.30
            self.hammer_ready_t = t_sec + 1.15
            if record_stats and self.hammer_acts:
                self.actuator_exec["hammer_slams"] += 1
        ham_pos = 1.98 if t_sec < self.hammer_active_until else -0.24
        for aid in self.hammer_acts:
            data.ctrl[aid] = ham_pos

        # 5. Pneumatic Piston Punch burst + auto-retract
        want_pist = bool(chosen_act["piston"](env)) if "piston" in chosen_act else False
        if want_pist and t_sec >= self.piston_ready_t:
            self.piston_active_until = t_sec + 0.24
            self.piston_ready_t = t_sec + 1.10
            if record_stats and self.piston_acts:
                self.actuator_exec["piston_punches"] += 1
        pist_firing = t_sec < self.piston_active_until
        for aid, stroke in self.piston_acts:
            data.ctrl[aid] = stroke if pist_firing else 0.0

        # 6. 360-deg Motorized Turret Slew + Ranged Energy Cannons
        if self.cannons:
            R_root = data.xmat[self.root_body_id].reshape(3, 3)
            rfx, rfy = float(R_root[0, 0]), float(R_root[1, 0])
            rfn = math.hypot(rfx, rfy)
            rfwd_xy = (rfx / rfn, rfy / rfn) if rfn > 1e-5 else (1.0, 0.0)
            rleft_xy = (-rfwd_xy[1], rfwd_xy[0])

            yaw_fn = chosen_act.get("cannon_yaw_deg", chosen_act.get("turret_yaw_deg"))
            turn_fn = chosen_act.get("cannon_turn", chosen_act.get("turret_turn"))
            target_yaw_deg = float(yaw_fn(env)) if yaw_fn is not None else None
            direct_turn = clamp(float(turn_fn(env)), -1.0, 1.0) if turn_fn is not None else None

            for c in self.cannons:
                taid = c["turret_act_id"]
                if taid < 0:
                    continue
                cR = data.xmat[c["body_id"]].reshape(3, 3)
                z_sign = 1.0 if float(cR[2, 2]) >= 0.0 else -1.0
                if direct_turn is not None and target_yaw_deg is None:
                    w_cmd = direct_turn * 6.0 * z_sign
                else:
                    tgt_deg = target_yaw_deg if target_yaw_deg is not None else 0.0
                    cd = cR @ c["dir_local"]
                    cfn = math.hypot(float(cd[0]), float(cd[1]))
                    cfwd_xy = (float(cd[0]) / cfn, float(cd[1]) / cfn) if cfn > 1e-5 else rfwd_xy
                    cur_rel_deg = _local_angle_deg(rfwd_xy, rleft_xy, cfwd_xy)
                    err_deg = ((tgt_deg - cur_rel_deg + 180.0) % 360.0) - 180.0
                    w_cmd = clamp(math.radians(err_deg) * 14.0, -6.0, 6.0) * z_sign
                data.ctrl[taid] = w_cmd
                if record_stats:
                    self.actuator_exec["turret_slewed_deg"] += math.degrees(abs(w_cmd)) * dt

        c_fn = chosen_act.get("cannon", chosen_act.get("fire_cannon"))
        want_cannon = bool(c_fn(env)) if c_fn is not None else False
        if want_cannon and self.cannons:
            root_vel = data.cvel[self.root_body_id][3:6].copy()
            for c in self.cannons:
                if t_sec >= c["ready_t"]:
                    c["ready_t"] = t_sec + c["cooldown"]
                    cb_id = c["body_id"]
                    cR = data.xmat[cb_id].reshape(3, 3)
                    cpos = data.xpos[cb_id]
                    muz_pos = cpos + cR @ c["muzzle_local"]
                    muz_dir = (cR @ c["dir_local"]).copy()
                    dn = float(np.linalg.norm(muz_dir))
                    if dn > 1e-6:
                        muz_dir /= dn
                    else:
                        muz_dir = np.array([1.0, 0.0, 0.0], dtype=np.float64)

                    mid = c["mocap_ids"][c["slot_idx"] % len(c["mocap_ids"])]
                    c["slot_idx"] += 1
                    if record_stats and chosen_slot in self.rule_records:
                        self.rule_records[chosen_slot]["shots_fired"] += 1
                    self._pending_shots.append({
                        "owner": self.prefix,
                        "cannon_body_id": cb_id,
                        "mocap_id": mid,
                        "spawn_pos": muz_pos.copy(),
                        "pos": muz_pos.copy(),
                        "vel": muz_dir * c["bolt_speed"] + root_vel * 0.25,
                        "dir": muz_dir.copy(),
                        "impulse": c["bolt_impulse"],
                        "recoil": c["recoil_impulse"],
                        "damage": c["bolt_damage"],
                        "ttl": 0.95,
                        "rule_slot": chosen_slot,
                        "rule_name": chosen_name,
                        "t_fired": round(t_sec, 2),
                    })

        # 7. Booster Thrusters
        want_thrust = bool(chosen_act["thruster"](env)) if "thruster" in chosen_act else False
        thrust_active = False
        if want_thrust and self.thruster_fuel > 0.04 and len(self.thrusters) > 0:
            thrust_active = True
            self.thruster_fuel = max(0.0, self.thruster_fuel - dt)
            if record_stats:
                self.actuator_exec["thruster_burn_sec"] += dt
        else:
            self.thruster_fuel = min(1.80, self.thruster_fuel + dt * 0.36)

        for th in self.thrusters:
            bid = th["body_id"]
            faid = th["flame_act_id"]
            if thrust_active:
                bR = data.xmat[bid].reshape(3, 3)
                f_world = bR @ (th["force_local"] * th["max_force_n"])
                data.xfrc_applied[bid, :3] = f_world
                data.xfrc_applied[bid, 3:] = 0.0
                if faid >= 0:
                    data.ctrl[faid] = 0.19 + 0.04 * math.sin(t_sec * 95.0)
            else:
                data.xfrc_applied[bid, :] = 0.0
                if faid >= 0:
                    data.ctrl[faid] = 0.0

        return sensors

    def record_melee_damage(self, dmg_impulse: float) -> None:
        rec = self.rule_records.get(self.active_rule_slot)
        if rec is not None:
            rec["melee_damage_dealt"] += float(dmg_impulse)

    def record_spring_launch(self, is_self: bool) -> None:
        if is_self:
            self.actuator_exec["spring_launches_self"] += 1
            rec = self.rule_records.get(self.active_rule_slot)
            if rec is not None:
                rec["spring_launches_taken"] += 1
        else:
            self.actuator_exec["spring_launches_enemy"] += 1

    def record_portal_warp(self, is_self: bool) -> None:
        if is_self:
            self.actuator_exec["portal_warps_self"] += 1
        else:
            self.actuator_exec["portal_warps_enemy"] += 1

    def record_shot_outcome(self, rule_slot: str, outcome: str) -> None:
        rec = self.rule_records.get(rule_slot)
        if rec is None:
            return
        if outcome == "hit_hull":
            rec["shots_hit_hull"] += 1
        elif outcome == "blocked_by_terrain":
            rec["shots_blocked_by_terrain"] += 1
        elif outcome == "missed":
            rec["shots_missed"] += 1

    def build_strategy_diagnostics(
        self,
        match_duration_sec: float,
        ko_note: Optional[str] = None,
    ) -> Dict[str, Any]:
        dur = max(0.05, float(match_duration_sec))
        ordered_rules: List[Dict[str, Any]] = []
        insights: List[str] = []

        tot_fired = 0
        tot_hit = 0
        tot_ter_blk = 0
        tot_miss = 0

        for slot_key, r in self.rule_records.items():
            is_default = (slot_key == "default")
            act_sec = round(float(r["active_sec"]), 2)
            act_pct = round(100.0 * float(r["active_sec"]) / dur, 1)
            never = (r["ticks"] == 0)

            tot_fired += int(r["shots_fired"])
            tot_hit += int(r["shots_hit_hull"])
            tot_ter_blk += int(r["shots_blocked_by_terrain"])
            tot_miss += int(r["shots_missed"])

            row: Dict[str, Any] = {
                "priority": "fallback" if is_default else r["order"],
                "rule": r["name"],
                "status": "NEVER_TRIGGERED" if never else "ACTIVE",
                "wheel_expr": r["wheel_expr"],
                "active_sec": act_sec,
                "active_pct": act_pct,
                "times_entered": int(r["entries"]),
                "active_window_sec": [r["first_t"], r["last_t"]] if not never else None,
                "distance_driven_m": round(float(r["distance_driven_m"]), 2),
                "melee_damage_dealt": round(float(r["melee_damage_dealt"]), 2),
                "spring_launches_taken": int(r["spring_launches_taken"]),
            }
            if self.cannons or int(r["shots_fired"]) > 0:
                row["cannon_shots"] = {
                    "fired": int(r["shots_fired"]),
                    "hit_hull": int(r["shots_hit_hull"]),
                    "blocked_by_terrain": int(r["shots_blocked_by_terrain"]),
                    "missed": int(r["shots_missed"]),
                }
            ordered_rules.append(row)

            if never and not is_default:
                w_raw = r["when"]
                st_in = w_raw.get("stage_in") if isinstance(w_raw, dict) else None
                if isinstance(st_in, (list, tuple)) and self.stage_id not in st_in:
                    row["status"] = f"SKIPPED_ON_STAGE_{self.stage_id}"
                else:
                    insights.append(
                        f"NEVER TRIGGERED: Rule #{r['order']} ({r['name']!r}) matched 0.0s (0%) this match — "
                        f"check if its 'when' condition is too strict or shadowed by an earlier rule."
                    )
            elif not never and float(r["active_sec"]) >= 2.2 and float(r["distance_driven_m"]) < 0.25:
                insights.append(
                    f"LOW MOBILITY UNDER RULE {r['name']!r}: active for {act_sec}s but moved only "
                    f"{round(float(r['distance_driven_m']), 2)}m."
                )

        act_report: Dict[str, Any] = {
            "distance_driven_m": round(float(self.actuator_exec["total_distance_m"]), 2),
            "stalled_time_sec": round(float(self.actuator_exec["total_stalled_sec"]), 2),
            "inverted_time_sec": round(float(self.actuator_exec["total_inverted_sec"]), 2),
            "mud_time_sec": round(float(self.actuator_exec["mud_time_sec"]), 2),
            "ice_time_sec": round(float(self.actuator_exec["ice_time_sec"]), 2),
            "spring_launches_self": int(self.actuator_exec["spring_launches_self"]),
            "spring_launches_enemy": int(self.actuator_exec["spring_launches_enemy"]),
        }
        if self.spinner_acts:
            act_report["spinner_active_sec"] = round(float(self.actuator_exec["spinner_active_sec"]), 2)
        if self.flipper_acts:
            act_report["flipper_fires"] = int(self.actuator_exec["flipper_fires"])
            if act_report["flipper_fires"] == 0:
                insights.append("UNUSED WEAPON: Installed 'flipper' fired 0 times this match.")
        if self.hammer_acts:
            act_report["hammer_slams"] = int(self.actuator_exec["hammer_slams"])
            if act_report["hammer_slams"] == 0:
                insights.append("UNUSED WEAPON: Installed 'hammer' slammed 0 times this match.")
        if self.piston_acts:
            act_report["piston_punches"] = int(self.actuator_exec["piston_punches"])
            if act_report["piston_punches"] == 0:
                insights.append("UNUSED WEAPON: Installed 'piston_punch' extended 0 times this match.")
        if self.thrusters:
            act_report["thruster_burn_sec"] = round(float(self.actuator_exec["thruster_burn_sec"]), 2)
            if act_report["thruster_burn_sec"] < 0.02:
                insights.append("UNUSED BOOSTER: Installed 'thruster' burned 0.0s of fuel this match.")
        if self.cannons:
            in_flight = max(0, tot_fired - tot_hit - tot_ter_blk - tot_miss)
            act_report["energy_cannon"] = {
                "installed_count": len(self.cannons),
                "turret_slewed_deg": round(float(self.actuator_exec["turret_slewed_deg"]), 1),
                "shots_fired": tot_fired,
                "hit_enemy_hull": tot_hit,
                "blocked_by_slope_crest": tot_ter_blk,
                "missed_or_dodged": tot_miss + in_flight,
                "hull_hit_rate_pct": round(100.0 * tot_hit / max(1, tot_fired), 1) if tot_fired > 0 else 0.0,
            }
            if tot_fired == 0:
                insights.append("ZERO SHOTS FIRED: Installed 'energy_cannon' fired 0 shots this match.")
            elif tot_ter_blk >= 2 and tot_ter_blk >= 0.4 * tot_fired:
                insights.append(
                    f"SHOTS BLOCKED BY SLOPE CREST: {tot_ter_blk}/{tot_fired} cannon shots hit sloped terrain — "
                    f"consider gating fire with 'has_line_of_sight' or mounting cannon higher."
                )

        if float(self.actuator_exec["total_stalled_sec"]) >= 3.5:
            insights.append(
                f"HIGH STALL DURATION: Speed stayed < 0.14 m/s for {round(float(self.actuator_exec['total_stalled_sec']), 2)}s — "
                f"consider adding 'stalled_sec' reverse/oscillation wheel logic."
            )
        if ko_note:
            insights.append(ko_note)
        if not insights:
            insights.append("All stage-applicable rules and installed actuators executed cleanly without dead-rule or stall anomalies.")

        return {
            "insights": insights,
            "actuator_execution": act_report,
            "rule_coverage": ordered_rules,
        }
