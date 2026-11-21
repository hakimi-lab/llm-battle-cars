"""
Simulation Test Quota Manager (.arena_sim_quota.json).

Enforces the strict fair-play rule: each contestant AI may spend at most
MAX_SIM_CREDITS (120) simulation credits during design, custom-opponent red-teaming & A/B testing:
  - 1 single-stage match = 1 credit
  - 1 full BO3 series    = 3 credits (up to 40 full BO3 tests total)
`validate` and `preview` (on both your car and any custom opponent JSONs you design!) cost 0 credits.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Tuple

from .catalog import MAX_SIM_CREDITS

DEFAULT_QUOTA_FILE = Path(__file__).resolve().parent.parent / ".arena_sim_quota.json"


def _load_ledger(quota_path: Path = DEFAULT_QUOTA_FILE) -> Dict[str, Any]:
    if quota_path.exists():
        try:
            data = json.loads(quota_path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("credits_used"), int):
                return data
        except Exception:
            pass
    return {
        "max_credits": MAX_SIM_CREDITS,
        "credits_used": 0,
        "history": [],
    }


def get_quota_status(quota_path: Path = DEFAULT_QUOTA_FILE) -> Dict[str, Any]:
    ledger = _load_ledger(quota_path)
    used = int(ledger.get("credits_used", 0))
    rem = max(0, MAX_SIM_CREDITS - used)
    return {
        "max_credits": MAX_SIM_CREDITS,
        "credits_used": used,
        "credits_remaining": rem,
        "runs_logged": len(ledger.get("history", [])),
        "history": ledger.get("history", []),
    }


def check_can_spend(cost: int, quota_path: Path = DEFAULT_QUOTA_FILE) -> Tuple[bool, str, Dict[str, Any]]:
    status = get_quota_status(quota_path)
    if status["credits_used"] + cost > MAX_SIM_CREDITS:
        msg = (
            f"SIMULATION QUOTA EXCEEDED: Requested {cost} credit(s), "
            f"but only {status['credits_remaining']}/{MAX_SIM_CREDITS} credit(s) remain! "
            f"Per tournament fair-play rules, finalize your car JSON now using `python arena.py validate` "
            f"and `python arena.py preview` (0 credits) without deleting .arena_sim_quota.json."
        )
        return False, msg, status
    return True, "", status


def record_sim_spend(
    car_path: Path,
    opponent: str,
    stage: str,
    cost: int,
    outcome_summary: str,
    quota_path: Path = DEFAULT_QUOTA_FILE,
) -> Dict[str, Any]:
    ledger = _load_ledger(quota_path)
    raw_bytes = car_path.read_bytes() if car_path.exists() else b""
    sha = hashlib.sha256(raw_bytes).hexdigest()[:12]
    ledger["max_credits"] = MAX_SIM_CREDITS
    ledger["credits_used"] = int(ledger.get("credits_used", 0)) + int(cost)
    ledger.setdefault("history", []).append({
        "utc_time": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "car_file": car_path.name,
        "car_sha12": sha,
        "opponent": opponent,
        "stage": stage,
        "cost": cost,
        "result": outcome_summary,
    })
    quota_path.write_text(json.dumps(ledger, ensure_ascii=False, indent=2), encoding="utf-8")
    return get_quota_status(quota_path)
