"""
BO3 Knockout Promotion Tournament & 360° Beauty Turntable Exporter (`arena.py tournament`).

Supports any number of submitted cars (N >= 2):
  1. Validates every car JSON in `submissions_dir`
  2. Exports 360° Beauty Turntable pose bundles for the Vehicle Aesthetics Evaluation segment
  3. Runs deterministic seeding against the 4 sparring bots if N > 2
  4. Runs a clean single-elimination BO3 promotion bracket (Play-Ins -> QF -> SF -> 3rd Place -> Grand Final)
     with every game played across the 3 distinct hazard stages and exported as native 30fps replays!
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mujoco
import numpy as np

from .arenas import ARENA_ORDER
from .catalog import FPS
from .compiler import compile_single_car_model, full_validate_car
from .engine import run_bo3_series


def export_beauty_turntable_bundle(spec: Dict[str, Any], out_prefix: Path, seconds: float = 5.0) -> Path:
    """
    Export a native 30fps 360-degree studio turntable animation bundle (.npz + .json)
    for the Vehicle Aesthetics Evaluation (`战车外形评价环节`) in Blender 4.2 OPTIX!
    """
    model, data, diag = compile_single_car_model(spec)
    num_frames = int(seconds * FPS)
    styles = diag["geom_styles"]

    geom_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gi) or f"geom_{gi}"
        for gi in range(model.ngeom)
    ]
    geom_info = []
    for gi, gname in enumerate(geom_names):
        owner = "A" if gname.startswith("A_") else "env"
        geom_info.append({
            "id": gi,
            "name": gname,
            "owner": owner,
            "type": int(model.geom_type[gi]),
            "size": [round(float(x), 5) for x in model.geom_size[gi]],
            "rgba": [round(float(x), 4) for x in model.geom_rgba[gi]],
            "style": styles.get(gname, {}),
        })

    # Spin any rotary blades slowly during showcase for extra visual flair while rotating car 360 deg!
    xpos_frames = []
    xmat_frames = []
    base_z = float(data.qpos[2])

    for f in range(num_frames):
        yaw = 2.0 * math.pi * (f / float(num_frames))
        data.qpos[0] = 0.0
        data.qpos[1] = 0.0
        data.qpos[2] = base_z
        data.qpos[3] = math.cos(0.5 * yaw)
        data.qpos[4] = 0.0
        data.qpos[5] = 0.0
        data.qpos[6] = math.sin(0.5 * yaw)
        # Gently animate weapon joints during showcase!
        for j in range(1, model.njnt):
            jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
            qadr = int(model.jnt_qposadr[j])
            if "spin_j" in jname or "vspin_j" in jname:
                data.qpos[qadr] = f * 0.28
            elif "flip_j" in jname and 45 <= f <= 72:
                data.qpos[qadr] = 0.85 * math.sin(math.pi * (f - 45) / 27.0)
            elif "ham_j" in jname and 75 <= f <= 105:
                data.qpos[qadr] = 1.45 * math.sin(math.pi * (f - 75) / 30.0)
            elif "pist_j" in jname and 95 <= f <= 122:
                data.qpos[qadr] = 0.30 * math.sin(math.pi * (f - 95) / 27.0)
            elif "turret_j" in jname:
                data.qpos[qadr] = 0.62 * math.sin(4.0 * math.pi * f / float(num_frames))
        mujoco.mj_forward(model, data)
        xpos_frames.append(data.geom_xpos.astype(np.float32).copy())
        xmat_frames.append(data.geom_xmat.astype(np.float32).copy())

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    npz_path = out_prefix.with_suffix(".npz")
    json_path = out_prefix.with_suffix(".json")
    np.savez_compressed(npz_path, xpos=np.stack(xpos_frames, axis=0), xmat=np.stack(xmat_frames, axis=0))
    json_path.write_text(
        json.dumps(
            {
                "mode": "beauty_turntable",
                "fps": FPS,
                "num_frames": num_frames,
                "car_meta": spec.get("meta", {}),
                "summary": {k: v for k, v in diag.items() if k != "geom_styles"},
                "geom_info": geom_info,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return npz_path


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def _standard_bracket_order(size: int) -> List[int]:
    """Return classic tournament seed positions (0-indexed) of length `size` (e.g. [0,7,3,4,1,6,2,5] for 8)."""
    seeds = [0, 1]
    while len(seeds) < size:
        nxt: List[int] = []
        cur_max = len(seeds) * 2 - 1
        for s in seeds:
            nxt.append(s)
            nxt.append(cur_max - s)
        seeds = nxt
    return seeds


def run_knockout_tournament(
    submissions_dir: Path,
    sparring_dir: Path,
    out_dir: Path,
    record_replays: bool = True,
) -> Dict[str, Any]:
    """Run full BO3 seeding + single-elimination championship bracket (+ 3rd place match)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    replays_dir = out_dir / "replays" if record_replays else None
    showcase_dir = out_dir / "beauty_turntables"

    entries: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []

    for jp in sorted(submissions_dir.glob("*.json")):
        try:
            spec = json.loads(jp.read_text(encoding="utf-8"))
        except Exception as exc:
            rejected.append({"file": jp.name, "errors": [f"Invalid JSON syntax: {exc}"]})
            continue
        ok, errs, warns, summ = full_validate_car(spec)
        if not ok:
            rejected.append({"file": jp.name, "errors": errs})
            continue
        slug = jp.stem
        if record_replays:
            export_beauty_turntable_bundle(spec, showcase_dir / f"turntable_{slug}")
        entries.append({
            "slug": slug,
            "file": jp.name,
            "spec": spec,
            "model_name": summ["model_name"],
            "car_name": summ["car_name"],
            "summary": summ,
            "warnings": warns,
        })

    if len(entries) < 2:
        raise ValueError(f"Need at least 2 valid contestant JSON files in {submissions_dir}, found {len(entries)}.")

    # Quick deterministic seeding trial against dummy_rusher & dummy_spinner on Stage 1 + Stage 2
    sparring_specs = []
    for sname in ("dummy_rusher.json", "dummy_spinner.json"):
        sp_path = sparring_dir / sname
        if sp_path.exists():
            sparring_specs.append(json.loads(sp_path.read_text(encoding="utf-8")))

    for ent in entries:
        seed_pts = 0.0
        for sp_idx, sp in enumerate(sparring_specs):
            st_key = ARENA_ORDER[sp_idx % len(ARENA_ORDER)]
            from .engine import run_single_match
            mres = run_single_match(ent["spec"], sp, stage=st_key, record_replay=False)
            seed_pts += (100.0 if mres["winner"] == "A" else 0.0) + float(mres["scores"]["A"]["total"])
            if mres["winner"] == "A" and mres["ko_time_sec"] is not None:
                seed_pts += max(0.0, 15.0 - float(mres["ko_time_sec"]))
        ent["seed_score"] = round(seed_pts, 2)

    entries.sort(key=lambda e: (-e["seed_score"], e["slug"]))
    for rank, ent in enumerate(entries, start=1):
        ent["seed"] = rank

    # Build power-of-2 bracket with byes for top seeds
    n = len(entries)
    bsize = _next_pow2(n)
    order = _standard_bracket_order(bsize)
    current_slots: List[Optional[Dict[str, Any]]] = [
        entries[s] if s < n else None
        for s in order
    ]

    rounds_log: List[Dict[str, Any]] = []
    round_num = 1
    semifinal_losers: List[Dict[str, Any]] = []

    while len(current_slots) > 1:
        num_matches = len(current_slots) // 2
        if num_matches == 1:
            rname = "Grand Final"
        elif num_matches == 2:
            rname = "Semifinals"
        elif num_matches == 4:
            rname = "Quarterfinals"
        else:
            rname = f"Round of {len(current_slots)}"

        round_matches: List[Dict[str, Any]] = []
        next_slots: List[Optional[Dict[str, Any]]] = []

        for m_idx in range(num_matches):
            ea = current_slots[2 * m_idx]
            eb = current_slots[2 * m_idx + 1]
            if ea is not None and eb is None:
                next_slots.append(ea)
                round_matches.append({
                    "match_id": f"R{round_num}_M{m_idx + 1}",
                    "bye": True,
                    "advances": {"seed": ea["seed"], "model": ea["model_name"], "car": ea["car_name"]},
                })
            elif eb is not None and ea is None:
                next_slots.append(eb)
                round_matches.append({
                    "match_id": f"R{round_num}_M{m_idx + 1}",
                    "bye": True,
                    "advances": {"seed": eb["seed"], "model": eb["model_name"], "car": eb["car_name"]},
                })
            elif ea is not None and eb is not None:
                mtag = f"R{round_num:02d}_M{m_idx + 1:02d}_{ea['slug']}_vs_{eb['slug']}"
                bo3 = run_bo3_series(
                    ea["spec"],
                    eb["spec"],
                    play_all_three=False,
                    record_replays_dir=replays_dir,
                    series_tag=mtag,
                )
                win_ent = ea if bo3["series_winner"] == "A" else eb
                lose_ent = eb if bo3["series_winner"] == "A" else ea
                if num_matches == 2:
                    semifinal_losers.append(lose_ent)
                next_slots.append(win_ent)
                round_matches.append({
                    "match_id": mtag,
                    "bye": False,
                    "car_a": {"seed": ea["seed"], "model": ea["model_name"], "car": ea["car_name"], "slug": ea["slug"]},
                    "car_b": {"seed": eb["seed"], "model": eb["model_name"], "car": eb["car_name"], "slug": eb["slug"]},
                    "series_score": bo3["series_score"],
                    "winner": {"seed": win_ent["seed"], "model": win_ent["model_name"], "car": win_ent["car_name"], "slug": win_ent["slug"]},
                    "games": bo3["games"],
                })
            else:
                next_slots.append(None)

        rounds_log.append({"round_index": round_num, "round_name": rname, "matches": round_matches})
        current_slots = next_slots
        round_num += 1

    champion = current_slots[0]
    bronze_match_log: Optional[Dict[str, Any]] = None
    third_place: Optional[Dict[str, Any]] = None
    if len(semifinal_losers) == 2:
        la, lb = semifinal_losers[0], semifinal_losers[1]
        btag = f"Bronze_{la['slug']}_vs_{lb['slug']}"
        b_bo3 = run_bo3_series(
            la["spec"],
            lb["spec"],
            play_all_three=False,
            record_replays_dir=replays_dir,
            series_tag=btag,
        )
        third_place = la if b_bo3["series_winner"] == "A" else lb
        bronze_match_log = {
            "match_id": btag,
            "car_a": {"seed": la["seed"], "model": la["model_name"], "car": la["car_name"]},
            "car_b": {"seed": lb["seed"], "model": lb["model_name"], "car": lb["car_name"]},
            "series_score": b_bo3["series_score"],
            "winner": {"seed": third_place["seed"], "model": third_place["model_name"], "car": third_place["car_name"]},
            "games": b_bo3["games"],
        }

    # Runner-up is loser of Grand Final
    gf_match = rounds_log[-1]["matches"][0]
    runner_up = gf_match["car_b"] if gf_match["winner"]["slug"] == gf_match["car_a"]["slug"] else gf_match["car_a"]

    report = {
        "contestants_count": len(entries),
        "rejected": rejected,
        "seedings": [
            {
                "seed": e["seed"],
                "slug": e["slug"],
                "model_name": e["model_name"],
                "car_name": e["car_name"],
                "seed_score": e["seed_score"],
                "build_points": e["summary"]["build_points"],
                "functional_parts": e["summary"]["functional_parts"],
                "decorations": e["summary"]["decorations"],
                "mass_kg": e["summary"]["total_mass_kg"],
                "slogan": e["spec"].get("meta", {}).get("slogan", ""),
            }
            for e in entries
        ],
        "podium": {
            "champion": {"seed": champion["seed"], "model": champion["model_name"], "car": champion["car_name"], "slug": champion["slug"]},
            "runner_up": runner_up,
            "third_place": (
                {"seed": third_place["seed"], "model": third_place["model_name"], "car": third_place["car_name"], "slug": third_place["slug"]}
                if third_place else None
            ),
        },
        "bronze_match": bronze_match_log,
        "rounds": rounds_log,
    }
    (out_dir / "tournament_results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
