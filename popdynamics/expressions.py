"""Compile string equations from a runcard into a JAX-compatible callable.

A runcard has to express its right-hand side as text, so this module turns
``"u * (1 - v - mu1 * u)"`` into a function of ``(y, params)``. Evaluation
happens on JAX tracers, so the result works under ``jit``, ``grad``, ``jacfwd``
and ``vmap`` exactly like a hand-written Python model -- the runcard path gives
up no capability.

Expressions are parsed once at build time and checked twice: the syntax tree may
contain only arithmetic (no attribute access, indexing, lambdas or
comprehensions), and every name must resolve to a variable, a parameter or an
allowed function. That is aimed at catching typos with a clear message rather
than at sandboxing a hostile runcard -- runcards are trusted local files, on the
same footing as the Python scripts they replace.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import jax.numpy as jnp
from jax import Array

__all__ = ["ExpressionError", "ALLOWED_FUNCTIONS", "compile_rhs"]


class ExpressionError(ValueError):
    """Raised when a runcard expression is malformed or refers to unknown names."""


# Everything a population model plausibly needs, and nothing else.
ALLOWED_FUNCTIONS: dict[str, Any] = {
    "exp": jnp.exp, "log": jnp.log, "log10": jnp.log10, "log2": jnp.log2,
    "sqrt": jnp.sqrt, "cbrt": jnp.cbrt,
    "sin": jnp.sin, "cos": jnp.cos, "tan": jnp.tan,
    "arcsin": jnp.arcsin, "arccos": jnp.arccos, "arctan": jnp.arctan,
    "arctan2": jnp.arctan2,
    "sinh": jnp.sinh, "cosh": jnp.cosh, "tanh": jnp.tanh,
    "abs": jnp.abs, "sign": jnp.sign,
    "maximum": jnp.maximum, "minimum": jnp.minimum,
    "floor": jnp.floor, "ceil": jnp.ceil,
    "where": jnp.where, "clip": jnp.clip,
    "pi": jnp.pi, "e": jnp.e,
}

# Node types that constitute arithmetic. Anything else is rejected outright.
_ALLOWED_NODES = (
    ast.Expression, ast.Constant, ast.Name, ast.Load, ast.Call,
    ast.BinOp, ast.UnaryOp, ast.IfExp, ast.Compare,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.And, ast.Or, ast.Not, ast.BoolOp,
)


def _parse(expression: str, where: str, known: set[str]) -> Any:
    """Parse one expression, rejecting non-arithmetic syntax and unknown names."""
    if not isinstance(expression, str):
        raise ExpressionError(
            f"{where}: expected an expression string, got {type(expression).__name__}"
        )
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"{where}: could not parse {expression!r} ({exc.msg})") from None

    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ExpressionError(
                f"{where}: {type(node).__name__} is not allowed in {expression!r}; "
                "expressions may only use arithmetic, comparisons and the "
                "supported functions"
            )
        if isinstance(node, ast.Call) and not isinstance(node.func, ast.Name):
            raise ExpressionError(f"{where}: only plain function calls are allowed in {expression!r}")

    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    unknown = used - known
    if unknown:
        raise ExpressionError(
            f"{where}: unknown name(s) {sorted(unknown)} in {expression!r}. "
            f"Known here: {sorted(known - set(ALLOWED_FUNCTIONS))}, "
            f"plus functions {sorted(ALLOWED_FUNCTIONS)}"
        )
    return compile(tree, filename="<runcard>", mode="eval")


def compile_rhs(
    variables: Sequence[str],
    equations: Mapping[str, str],
    parameters: Mapping[str, Any] | None = None,
) -> Callable[[Array, Mapping[str, Any]], Array]:
    """Build ``rhs(y, params)`` from one expression per variable.

    Parameters
    ----------
    variables
        Ordered state variable names; ``y`` is indexed in this order.
    equations
        Maps each variable name to the expression for its time derivative.
    parameters
        Parameter names available to the expressions. Only the *names* matter
        here -- values are supplied at call time through ``params``.
    """
    variables = tuple(variables)
    parameters = dict(parameters or {})

    if not variables:
        raise ExpressionError("no variables given")
    if len(set(variables)) != len(variables):
        raise ExpressionError(f"duplicate variable names in {list(variables)}")

    missing = [v for v in variables if v not in equations]
    if missing:
        raise ExpressionError(f"no equation given for variable(s) {missing}")
    extra = [k for k in equations if k not in variables]
    if extra:
        raise ExpressionError(
            f"equation(s) for unknown variable(s) {sorted(extra)}; "
            f"declared variables are {list(variables)}"
        )
    clash = sorted(set(variables) & set(parameters))
    if clash:
        raise ExpressionError(f"name(s) {clash} used as both a variable and a parameter")

    known = set(variables) | set(parameters) | set(ALLOWED_FUNCTIONS)
    codes = [
        _parse(equations[name], f"equation for {name!r}", known) for name in variables
    ]

    def rhs(y: Array, params: Mapping[str, Any]) -> Array:
        scope = dict(ALLOWED_FUNCTIONS)
        scope.update(params)
        scope.update({name: y[i] for i, name in enumerate(variables)})
        return jnp.stack(
            [jnp.asarray(eval(code, {"__builtins__": {}}, scope), dtype=float) for code in codes]
        )

    rhs.expressions = {name: equations[name] for name in variables}  # for reporting
    return rhs
