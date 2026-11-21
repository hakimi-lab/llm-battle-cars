"""LLM Battle-Cars Simulation & BO3 Tournament Package."""
from .catalog import MAX_BUILD_POINTS, MAX_DECO_PARTS, MAX_FUNCTIONAL_PARTS, MAX_SIM_CREDITS, PARTS_CATALOG
from .compiler import full_validate_car
from .engine import run_bo3_series, run_single_match

__all__ = [
    "MAX_BUILD_POINTS",
    "MAX_DECO_PARTS",
    "MAX_FUNCTIONAL_PARTS",
    "MAX_SIM_CREDITS",
    "PARTS_CATALOG",
    "full_validate_car",
    "run_single_match",
    "run_bo3_series",
]
