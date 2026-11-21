"""
Safe Pure-Math & Boolean AST Expression Engine for DIY Vehicle Strategies (`battle_sim/expr.py`).

Allows AI contestants to write their own continuous differential wheel controllers, ballistic lead-aiming,
bolt-dodging geometry, cliff-recovery brakes, and spring-herding vector math directly inside JSON strings
with ZERO arbitrary code execution (no attribute access, no imports, no loops, no allocations).
"""
from __future__ import annotations

import ast
import difflib
import math
from typing import Any, Callable, Dict, FrozenSet, List, Set, Tuple

MAX_EXPR_CHARS = 650
MAX_AST_NODES = 160


def _clamp(val: float, lo: float, hi: float) -> float:
    v = float(val)
    lo_f, hi_f = float(lo), float(hi)
    if lo_f > hi_f:
        lo_f, hi_f = hi_f, lo_f
    return max(lo_f, min(hi_f, v))


def _wrap_deg(deg: float) -> float:
    return ((float(deg) + 180.0) % 360.0) - 180.0


def _sign(x: float) -> float:
    xf = float(x)
    if xf > 0.0:
        return 1.0
    if xf < 0.0:
        return -1.0
    return 0.0


def _safe_div(a: float, b: float) -> float:
    bf = float(b)
    if abs(bf) < 1e-9:
        bf = 1e-9 if bf >= 0.0 else -1e-9
    return float(a) / bf


def _safe_floordiv(a: float, b: float) -> float:
    bf = float(b)
    if abs(bf) < 1e-9:
        bf = 1e-9 if bf >= 0.0 else -1e-9
    return math.floor(float(a) / bf)


def _safe_mod(a: float, b: float) -> float:
    bf = float(b)
    if abs(bf) < 1e-9:
        return 0.0
    return float(a) % bf


def _safe_pow(a: float, b: float) -> float:
    af = float(a)
    bf = _clamp(float(b), -6.0, 6.0)
    if af < 0.0 and abs(bf - round(bf)) > 1e-9:
        return -(abs(af) ** bf)
    if abs(af) < 1e-9 and bf < 0.0:
        return 0.0
    return float(af ** bf)


def _safe_sqrt(x: float) -> float:
    return math.sqrt(max(0.0, float(x)))


def _safe_asin(x: float) -> float:
    return math.asin(_clamp(float(x), -1.0, 1.0))


def _safe_acos(x: float) -> float:
    return math.acos(_clamp(float(x), -1.0, 1.0))


def _sin_deg(d: float) -> float:
    return math.sin(math.radians(float(d)))


def _cos_deg(d: float) -> float:
    return math.cos(math.radians(float(d)))


def _tan_deg(d: float) -> float:
    return _clamp(math.tan(math.radians(float(d))), -1e4, 1e4)


def _atan2_deg(y: float, x: float) -> float:
    return math.degrees(math.atan2(float(y), float(x)))


def _iff(cond: Any, a: Any, b: Any) -> Any:
    return a if bool(cond) else b


SAFE_FUNCTIONS: Dict[str, Callable[..., Any]] = {
    "abs": abs,
    "min": min,
    "max": max,
    "clamp": _clamp,
    "sign": _sign,
    "sin": lambda x: math.sin(float(x)),
    "cos": lambda x: math.cos(float(x)),
    "tan": lambda x: _clamp(math.tan(float(x)), -1e4, 1e4),
    "sin_deg": _sin_deg,
    "cos_deg": _cos_deg,
    "tan_deg": _tan_deg,
    "asin": _safe_asin,
    "acos": _safe_acos,
    "atan2": lambda y, x: math.atan2(float(y), float(x)),
    "atan2_deg": _atan2_deg,
    "deg": lambda r: math.degrees(float(r)),
    "rad": lambda d: math.radians(float(d)),
    "wrap_deg": _wrap_deg,
    "sqrt": _safe_sqrt,
    "hypot": lambda x, y: math.hypot(float(x), float(y)),
    "iff": _iff,
}

SAFE_CONSTANTS: Dict[str, Any] = {
    "pi": math.pi,
    "true": True,
    "false": False,
    "True": True,
    "False": False,
}

_INTERNAL_HELPERS: Dict[str, Callable[..., Any]] = {
    "_sdiv": _safe_div,
    "_sfdiv": _safe_floordiv,
    "_smod": _safe_mod,
    "_spow": _safe_pow,
}

BASE_EVAL_ENV: Dict[str, Any] = {}
BASE_EVAL_ENV.update(SAFE_FUNCTIONS)
BASE_EVAL_ENV.update(SAFE_CONSTANTS)
BASE_EVAL_ENV.update(_INTERNAL_HELPERS)

VALID_FUNC_NAMES: FrozenSet[str] = frozenset(SAFE_FUNCTIONS.keys())
VALID_CONST_NAMES: FrozenSet[str] = frozenset(SAFE_CONSTANTS.keys())


class _SafeMathRewriter(ast.NodeTransformer):
    """Rewrite /, //, %, and ** into safe total math functions so runtime ZeroDivisionError/complex never occur."""

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        self.generic_visit(node)
        helper_name = None
        if isinstance(node.op, ast.Div):
            helper_name = "_sdiv"
        elif isinstance(node.op, ast.FloorDiv):
            helper_name = "_sfdiv"
        elif isinstance(node.op, ast.Mod):
            helper_name = "_smod"
        elif isinstance(node.op, ast.Pow):
            helper_name = "_spow"
        if helper_name is not None:
            return ast.copy_location(
                ast.Call(
                    func=ast.Name(id=helper_name, ctx=ast.Load()),
                    args=[node.left, node.right],
                    keywords=[],
                ),
                node,
            )
        return node


def validate_and_compile_expr(
    raw: Any,
    allowed_vars: Set[str],
    ctx: str,
) -> Tuple[Callable[[Dict[str, Any]], Any], Set[str]]:
    """
    Validate and compile a constant (int/float/bool) or safe math/boolean string expression.
    Returns: (callable_fn(env_dict) -> float|bool, referenced_variable_names).
    Raises ValueError with clear human-readable diagnostics on invalid syntax or unknown identifiers.
    """
    if isinstance(raw, bool):
        val_b = bool(raw)
        return (lambda _env, v=val_b: v), set()
    if isinstance(raw, (int, float)):
        val_f = float(raw)
        if not math.isfinite(val_f):
            raise ValueError(f"{ctx}: numeric literal must be finite, got {raw!r}.")
        return (lambda _env, v=val_f: v), set()
    if not isinstance(raw, str):
        raise ValueError(
            f"{ctx}: expected a number, boolean, or formula string (e.g. \"clamp(0.8 - enemy_angle_deg / 60, -1, 1)\"), got {type(raw).__name__}."
        )

    expr_str = raw.strip()
    if not expr_str:
        raise ValueError(f"{ctx}: formula string cannot be empty.")
    if len(expr_str) > MAX_EXPR_CHARS:
        raise ValueError(f"{ctx}: formula exceeds {MAX_EXPR_CHARS} characters ({len(expr_str)} chars).")

    # Support C/JS logical shorthands (`&&`, `||`, `!` when not `!=`) gracefully in case a model writes them
    normalized = expr_str.replace("&&", " and ").replace("||", " or ")

    try:
        tree = ast.parse(normalized, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"{ctx}: syntax error in formula {expr_str!r} — {exc.msg} (col {exc.offset}).") from exc

    all_nodes = list(ast.walk(tree))
    if len(all_nodes) > MAX_AST_NODES:
        raise ValueError(f"{ctx}: formula is too complex ({len(all_nodes)} AST nodes > max {MAX_AST_NODES}). Use 'let' variables to split it.")

    referenced: Set[str] = set()
    allowed_binops = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
    allowed_unops = (ast.UAdd, ast.USub, ast.Not)
    allowed_boolops = (ast.And, ast.Or)
    allowed_cmps = (ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq)

    for node in all_nodes:
        if isinstance(node, (ast.Expression, ast.Load)):
            continue
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float, bool)):
                raise ValueError(f"{ctx}: string/bytes constants like {node.value!r} are not allowed inside math formulas.")
            if isinstance(node.value, (int, float)) and not math.isfinite(float(node.value)):
                raise ValueError(f"{ctx}: non-finite number in formula.")
            continue
        if isinstance(node, ast.Name):
            nm = node.id
            if nm in VALID_FUNC_NAMES or nm in VALID_CONST_NAMES:
                continue
            if nm not in allowed_vars:
                candidates = difflib.get_close_matches(nm, allowed_vars | VALID_FUNC_NAMES, n=3)
                hint = f" Did you mean {candidates}?" if candidates else ""
                raise ValueError(f"{ctx}: unknown sensor or variable {nm!r} in {expr_str!r}.{hint}")
            referenced.add(nm)
            continue
        if isinstance(node, ast.BinOp):
            if not isinstance(node.op, allowed_binops):
                raise ValueError(f"{ctx}: unsupported binary operator {type(node.op).__name__} in {expr_str!r}.")
            continue
        if isinstance(node, allowed_binops):
            continue
        if isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, allowed_unops):
                raise ValueError(f"{ctx}: unsupported unary operator {type(node.op).__name__} in {expr_str!r}.")
            continue
        if isinstance(node, allowed_unops):
            continue
        if isinstance(node, ast.BoolOp):
            if not isinstance(node.op, allowed_boolops):
                raise ValueError(f"{ctx}: unsupported boolean operator in {expr_str!r}.")
            continue
        if isinstance(node, allowed_boolops):
            continue
        if isinstance(node, ast.Compare):
            if not all(isinstance(op, allowed_cmps) for op in node.ops):
                raise ValueError(
                    f"{ctx}: unsupported comparison in {expr_str!r} (allowed: <, <=, >, >=, ==, !=)."
                )
            continue
        if isinstance(node, allowed_cmps):
            continue
        if isinstance(node, ast.IfExp):
            continue
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in VALID_FUNC_NAMES:
                fn_repr = node.func.id if isinstance(node.func, ast.Name) else ast.dump(node.func)
                raise ValueError(
                    f"{ctx}: function {fn_repr!r} is not allowed. Allowed functions: {sorted(VALID_FUNC_NAMES)}."
                )
            if node.keywords:
                raise ValueError(f"{ctx}: keyword arguments are not allowed in function calls ({node.func.id}).")
            continue

        raise ValueError(
            f"{ctx}: disallowed syntax construct '{type(node).__name__}' in {expr_str!r}. "
            f"Use pure math/boolean expressions over sensors, 'let' bindings, and 'vars'."
        )

    safe_tree = ast.fix_missing_locations(_SafeMathRewriter().visit(tree))
    code_obj = compile(safe_tree, "<strategy_expr>", "eval")

    def _eval_fn(env: Dict[str, Any]) -> Any:
        res = eval(code_obj, {"__builtins__": {}}, env)
        if isinstance(res, (int, float)) and not isinstance(res, bool):
            rf = float(res)
            return rf if math.isfinite(rf) else 0.0
        return res

    return _eval_fn, referenced
