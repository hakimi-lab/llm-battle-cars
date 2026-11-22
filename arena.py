#!/usr/bin/env python3
"""
LLM Battle-Cars — Unified Contestant & Tournament CLI (`arena.py`, v3.2).

Contestant Commands:
  python arena.py parts
      List all 13 functional parts, point costs, mounts, decoration primitives, and 3 BO3 arenas (0 credits).

  python arena.py validate path/to/my_car.json
      Validate JSON schema, part & point caps, mass, CoM, and real flat-sprint top speed (0 credits).

  python arena.py preview path/to/my_car.json [--out previews/my_car.png]
      Render a 4-view 3D shaded blueprint image (Isometric/Side/Front/Top) to inspect vehicle layout (0 credits).

  python arena.py preview-arena [--stage all|1|2|3] [--out previews/]
      Render 3D shaded & top-down tactical blueprints of the 3 BO3 tactical arenas (0 credits).

  python arena.py status
      Show remaining simulation test credits out of 120 (0 credits).

  python arena.py test path/to/my_car.json [--opponent custom_opponents/enemy.json|dummy_rusher|dummy_spinner|dummy_flipper|dummy_fortress|dummy_ranger|mirror] [--stage bo3|1|2|3] [--swap-sides]
      Run deterministic 360Hz MuJoCo combat simulation (30s rounds) against built-in bots OR custom opponents!
      Cost: 1 credit for a single stage (1, 2, or 3), 3 credits for `--stage bo3`.
      Hard limit: 120 simulation credits total per workspace.

Host Commands:
  python arena.py tournament submissions_dir/ [--out tournament_out/] [--no-replays]
      Run full BO3 promotion tournament + export 30fps Blender OPTIX replay & 360° turntable bundles.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from battle_sim.arenas import ARENA_META, ARENA_ORDER, resolve_stage_key
from battle_sim.catalog import (
    DECO_SHAPES,
    MAX_ACTIVE_WEAPONS,
    MAX_BUILD_POINTS,
    MAX_DECO_PARTS,
    MAX_ENERGY_CANNONS,
    MAX_FUNCTIONAL_PARTS,
    MAX_SIM_CREDITS,
    MOUNT_ANCHORS,
    PARTS_CATALOG,
    VALID_MATERIALS,
)
from battle_sim.compiler import full_validate_car
from battle_sim.engine import run_bo3_series, run_single_match
from battle_sim.preview import render_arena_preview, render_car_preview, render_tactical_trace_png
from battle_sim.quota import check_can_spend, get_quota_status, record_sim_spend
from battle_sim.tournament import run_knockout_tournament

REPO_ROOT = Path(__file__).resolve().parent
SPARRING_DIR = REPO_ROOT / "sparring_bots"


def _load_car_json(path_str: str) -> Tuple[Path, Dict[str, Any]]:
    p = Path(path_str)
    if not p.exists():
        print(f"[ERROR] File not found: {p}", file=sys.stderr)
        sys.exit(1)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[ERROR] Failed to parse JSON in {p}: {exc}", file=sys.stderr)
        sys.exit(1)
    return p, data


def cmd_parts(_args: argparse.Namespace) -> int:
    print("=" * 86)
    print(
        f"LLM BATTLE-CARS v3.2 CATALOG  (Max {MAX_FUNCTIONAL_PARTS} parts | Max {MAX_BUILD_POINTS} pts | "
        f"Max {MAX_ACTIVE_WEAPONS} active weapons | Max {MAX_ENERGY_CANNONS} 360°-turret cannons | Unrestricted LxWxH!)"
    )
    print("=" * 86)
    for ptype, info in PARTS_CATALOG.items():
        print(f"  - {ptype:<16} | {info['cost']:>2} pts | [{info['category']:<9}] {info['name_zh']} — {info['summary']}")
        print(f"      defaults: {json.dumps(info['defaults'])}")
    print(f"\nDecoration Primitives (`decorations`, 0 pts, zero collision, max {MAX_DECO_PARTS} items):")
    print(f"  - Shapes   : {', '.join(DECO_SHAPES)}")
    print(f"  - Materials: {', '.join(VALID_MATERIALS)}")
    print(f"  - Mounts   : {', '.join(MOUNT_ANCHORS.keys())}")
    print("\nThree Distinct BO3 Tactical Arenas (30s/round, Catapults, Ridge Vaults & Quantum Portals):")
    for sk in ARENA_ORDER:
        am = ARENA_META[sk]
        print(f"  - Stage {am['stage_id']} ({sk} / {am['name_zh']}): {am['description']}")
    return 0


def cmd_status(_args: argparse.Namespace) -> int:
    st = get_quota_status()
    print(json.dumps(st, ensure_ascii=False, indent=2))
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    p, spec = _load_car_json(args.car)
    ok, errors, warnings, summary = full_validate_car(spec)
    qst = get_quota_status()
    payload = {
        "valid": ok,
        "file": str(p),
        "summary": summary,
        "warnings": warnings,
        "errors": errors,
        "sim_quota": f"{qst['credits_used']}/{qst['max_credits']} used ({qst['credits_remaining']} remaining)",
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if ok else 2


def cmd_preview(args: argparse.Namespace) -> int:
    p, spec = _load_car_json(args.car)
    out_png = Path(args.out) if args.out else (REPO_ROOT / "previews" / f"{p.stem}_preview.png")
    try:
        summary = render_car_preview(spec, out_png)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    qst = get_quota_status()
    print(json.dumps({
        "ok": True,
        "preview_png": str(out_png.resolve()),
        "summary": summary,
        "sim_quota": f"{qst['credits_used']}/{qst['max_credits']} used ({qst['credits_remaining']} remaining — preview costs 0 credits)",
    }, ensure_ascii=False, indent=2))
    return 0


def cmd_preview_arena(args: argparse.Namespace) -> int:
    out_dir = Path(args.out) if args.out else (REPO_ROOT / "previews")
    out_dir.mkdir(parents=True, exist_ok=True)
    stages = list(ARENA_ORDER) if args.stage.strip().lower() == "all" else [resolve_stage_key(args.stage)]
    written: List[str] = []
    for sk in stages:
        dst = out_dir / f"arena_{sk}.png"
        render_arena_preview(sk, dst, t_sec=float(args.time))
        written.append(str(dst.resolve()))
    print(json.dumps({"ok": True, "arena_previews": written}, ensure_ascii=False, indent=2))
    return 0


def _resolve_opponent(opp_arg: str, self_spec: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    key = opp_arg.strip()
    if key.lower() == "mirror":
        import copy
        mirror_spec = copy.deepcopy(self_spec)
        m = mirror_spec.setdefault("meta", {})
        m["car_name"] = str(m.get("car_name", "Car")) + " (Mirror)"
        return "mirror", mirror_spec

    custom_path = Path(key)
    if custom_path.exists() and custom_path.is_file():
        try:
            spec = json.loads(custom_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[ERROR] Failed to parse custom opponent JSON {custom_path}: {exc}", file=sys.stderr)
            sys.exit(1)
        ok, errs, _, _ = full_validate_car(spec)
        if not ok:
            print(f"[ERROR] Custom opponent JSON {custom_path} failed rule validation:\n  - " + "\n  - ".join(errs), file=sys.stderr)
            sys.exit(2)
        return str(custom_path), spec

    custom_dir_path = REPO_ROOT / "custom_opponents" / (key if key.endswith(".json") else f"{key}.json")
    if custom_dir_path.exists():
        try:
            spec = json.loads(custom_dir_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[ERROR] Failed to parse custom opponent JSON {custom_dir_path}: {exc}", file=sys.stderr)
            sys.exit(1)
        ok, errs, _, _ = full_validate_car(spec)
        if not ok:
            print(f"[ERROR] Custom opponent JSON {custom_dir_path} failed rule validation:\n  - " + "\n  - ".join(errs), file=sys.stderr)
            sys.exit(2)
        return f"custom_opponents/{custom_dir_path.name}", spec

    builtin_path = SPARRING_DIR / (key if key.endswith(".json") else f"{key}.json")
    if builtin_path.exists():
        return builtin_path.stem, json.loads(builtin_path.read_text(encoding="utf-8"))

    avail = [f.stem for f in sorted(SPARRING_DIR.glob("*.json"))] + ["mirror", "path/to/your_custom_opponent.json"]
    print(f"[ERROR] Unknown opponent {opp_arg!r}. Options: {avail}.", file=sys.stderr)
    sys.exit(1)


def cmd_test(args: argparse.Namespace) -> int:
    car_path, spec_a = _load_car_json(args.car)
    ok, errors, warnings, _ = full_validate_car(spec_a)
    if not ok:
        print(json.dumps({"valid": False, "errors": errors}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2

    opp_name, spec_b = _resolve_opponent(args.opponent, spec_a)
    stage_raw = str(args.stage).strip().lower()
    is_bo3 = (stage_raw == "bo3")
    cost = 3 if is_bo3 else 1

    allowed, msg, _ = check_can_spend(cost)
    if not allowed:
        print(json.dumps({"ok": False, "error": "quota_exceeded", "message": msg}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 3

    car_a_title = spec_a.get("meta", {}).get("car_name", car_path.stem)
    car_b_title = spec_b.get("meta", {}).get("car_name", opp_name)

    opp_slug = Path(opp_name).stem.replace(" ", "_")

    if is_bo3:
        series = run_bo3_series(spec_a, spec_b, play_all_three=True, record_replays_dir=None, return_trace_data=True)
        outcome_str = f"BO3 {'WIN' if series['series_winner'] == 'A' else 'LOSS'} ({series['series_score']})"
        new_q = record_sim_spend(car_path, opp_name, "bo3", cost, outcome_str)
        trace_pngs: List[str] = []
        for g in series["games"]:
            g.pop("foley_events", None)
            tr = g.pop("_trace", None)
            if tr is not None:
                t_png = REPO_ROOT / "previews" / f"trace_{car_path.stem}_vs_{opp_slug}_{g['stage_key']}.png"
                render_tactical_trace_png(tr, t_png)
                g["tactical_trace_png"] = str(t_png.resolve())
                trace_pngs.append(str(t_png.resolve()))
        output: Dict[str, Any] = {
            "ok": True,
            "mode": "bo3_all_3_stages",
            "car_A_you": f"{car_a_title} ({car_path})",
            "car_B_opponent": f"{car_b_title} ({opp_name})",
            "series_result": "WIN" if series["series_winner"] == "A" else "LOSS",
            "series_score": series["series_score"],
            "warnings": warnings,
            "tactical_trace_pngs": trace_pngs,
            "sim_quota": {
                "cost_this_run": cost,
                "credits_used": new_q["credits_used"],
                "credits_remaining": new_q["credits_remaining"],
                "max_credits": new_q["max_credits"],
            },
            "games": series["games"],
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return 0
    else:
        stage_key = resolve_stage_key(stage_raw)
        mres = run_single_match(
            spec_a,
            spec_b,
            stage=stage_key,
            swap_sides=bool(args.swap_sides),
            record_replay=False,
            return_trace_data=True,
        )
        mres.pop("foley_events", None)
        tr = mres.pop("_trace", None)
        t_png_str: str = ""
        if tr is not None:
            t_png = REPO_ROOT / "previews" / f"trace_{car_path.stem}_vs_{opp_slug}_{stage_key}.png"
            render_tactical_trace_png(tr, t_png)
            t_png_str = str(t_png.resolve())
            mres["tactical_trace_png"] = t_png_str

        outcome_str = f"{stage_key}: {'WIN' if mres['winner'] == 'A' else 'LOSS'} ({mres['win_method']})"
        new_q = record_sim_spend(car_path, opp_name, stage_key, cost, outcome_str)
        output = {
            "ok": True,
            "mode": "single_stage",
            "car_A_you": f"{car_a_title} ({car_path})",
            "car_B_opponent": f"{car_b_title} ({opp_name})",
            "match_result": "WIN" if mres["winner"] == "A" else "LOSS",
            "warnings": warnings,
            "tactical_trace_png": t_png_str,
            "sim_quota": {
                "cost_this_run": cost,
                "credits_used": new_q["credits_used"],
                "credits_remaining": new_q["credits_remaining"],
                "max_credits": new_q["max_credits"],
            },
            "match": mres,
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return 0


def cmd_tournament(args: argparse.Namespace) -> int:
    sub_dir = Path(args.submissions_dir)
    if not sub_dir.is_dir():
        print(f"[ERROR] Submissions directory not found: {sub_dir}", file=sys.stderr)
        return 1
    out_dir = Path(args.out)
    report = run_knockout_tournament(
        submissions_dir=sub_dir,
        sparring_dir=SPARRING_DIR,
        out_dir=out_dir,
        record_replays=(not args.no_replays),
    )
    print(json.dumps({
        "ok": True,
        "contestants_count": report["contestants_count"],
        "rejected": report["rejected"],
        "podium": report["podium"],
        "results_json": str((out_dir / "tournament_results.json").resolve()),
    }, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LLM Battle-Cars Simulation & BO3 Tournament CLI (v3.0)")
    sub = p.add_subparsers(dest="command", required=True)

    sp_parts = sub.add_parser("parts", help="Show parts catalog, point costs, mounts, and arenas (0 credits)")
    sp_parts.set_defaults(func=cmd_parts)

    sp_status = sub.add_parser("status", help="Show remaining simulation test quota (0 credits)")
    sp_status.set_defaults(func=cmd_status)

    sp_val = sub.add_parser("validate", help="Validate any car/opponent JSON schema, budget & mobility diagnostics (0 credits)")
    sp_val.add_argument("car", help="Path to car JSON file")
    sp_val.set_defaults(func=cmd_validate)

    sp_prev = sub.add_parser("preview", help="Render 4-view 3D blueprint PNG of any car (0 credits)")
    sp_prev.add_argument("car", help="Path to car JSON file")
    sp_prev.add_argument("--out", default=None, help="Output PNG path (default: previews/<car>_preview.png)")
    sp_prev.set_defaults(func=cmd_preview)

    sp_aprev = sub.add_parser("preview-arena", help="Render 3D blueprint PNGs of the 3 BO3 arenas (0 credits)")
    sp_aprev.add_argument("--stage", default="all", help="all | 1 / spring_arena | 2 / camelback_ridge | 3 / portal_arena")
    sp_aprev.add_argument("--time", type=float, default=0.0, help="Simulation timestamp (s) to snapshot spring mechanisms (0.0 = symmetrical blueprint)")
    sp_aprev.add_argument("--out", default=None, help="Output directory (default: previews/)")
    sp_aprev.set_defaults(func=cmd_preview_arena)

    sp_test = sub.add_parser(
        "test",
        help=f"Run 360Hz MuJoCo combat test against built-in OR custom opponent JSON (1 credit/stage, 3 credits/bo3, max {MAX_SIM_CREDITS} credits)",
    )
    sp_test.add_argument("car", help="Path to your car JSON file (spawns as Car A)")
    sp_test.add_argument(
        "--opponent",
        default="dummy_rusher",
        help="Opponent (Car B): path to ANY custom opponent .json (e.g. custom_opponents/enemy.json), OR built-in dummy_rusher | dummy_spinner | dummy_flipper | dummy_fortress | dummy_ranger | mirror",
    )
    sp_test.add_argument(
        "--stage",
        default="bo3",
        help="Arena stage: bo3 (3 credits) | 1 / spring_arena | 2 / camelback_ridge | 3 / portal_arena (1 credit)",
    )
    sp_test.add_argument(
        "--swap-sides",
        action="store_true",
        help="Swap spawn positions on single-stage test",
    )
    sp_test.set_defaults(func=cmd_test)

    sp_tour = sub.add_parser("tournament", help="[Host] Run full BO3 knockout bracket & export 30fps Blender replays")
    sp_tour.add_argument("submissions_dir", help="Directory containing contestant *.json files")
    sp_tour.add_argument("--out", default="tournament_out", help="Output directory for bracket results & 30fps replays")
    sp_tour.add_argument("--no-replays", action="store_true", help="Skip saving .npz/.json 3D frame trajectories")
    sp_tour.set_defaults(func=cmd_tournament)

    return p


if __name__ == "__main__":
    parser = build_parser()
    ns = parser.parse_args()
    sys.exit(ns.func(ns))
