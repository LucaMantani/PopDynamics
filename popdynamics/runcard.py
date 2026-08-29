"""Parse and validate a runcard into a :class:`~popdynamics.system.System` plus
a list of things to draw.

This module holds no analysis logic. It turns YAML into the same objects the
Python API uses, so both front-ends necessarily agree. Validation is strict and
happens up front: an unknown key or a malformed value should be reported by
name, before any integration starts, rather than surfacing later as a traceback
from inside JAX.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from popdynamics import models
from popdynamics.expressions import ExpressionError, compile_rhs
from popdynamics.system import System

__all__ = ["Runcard", "RuncardError", "PlotSpec", "load", "from_dict", "PLOT_TYPES"]

PLOT_TYPES = ("phase_portrait", "timeseries")

# How many state variables each plot type needs; None means any number.
PLOT_DIMENSIONS: dict[str, int | None] = {
    "phase_portrait": 2,
    "timeseries": None,
}

_TOP_LEVEL = {"name", "system", "domain", "analysis", "plots"}
_SYSTEM_KEYS = {"variables", "equations", "parameters", "model"}
_ANALYSIS_KEYS = {"fixed_points"}


class RuncardError(ValueError):
    """Raised when a runcard is structurally invalid."""


@dataclass(frozen=True)
class PlotSpec:
    """One requested figure."""

    type: str
    output: str
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Runcard:
    """A parsed runcard: a system, a viewing domain, and what to produce."""

    name: str
    system: System
    domain: dict[str, tuple[float, float]]
    plots: tuple[PlotSpec, ...] = ()
    analysis: dict[str, Any] = field(default_factory=dict)
    source: str = "<dict>"

    @property
    def bounds(self) -> list[tuple[float, float]]:
        """Domain as an ordered ``(lo, hi)`` list, matching the variable order."""
        return [self.domain[name] for name in self.system.names]


def load(path: str | Path) -> Runcard:
    """Read and validate a runcard from a YAML file."""
    path = Path(path)
    try:
        text = path.read_text()
    except OSError as exc:
        raise RuncardError(f"could not read runcard {path}: {exc}") from None
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise RuncardError(f"{path}: invalid YAML ({exc})") from None
    if data is None:
        raise RuncardError(f"{path}: runcard is empty")
    return from_dict(data, source=str(path))


def from_dict(data: Any, *, source: str = "<dict>") -> Runcard:
    """Validate an already-loaded runcard mapping."""
    if not isinstance(data, dict):
        raise RuncardError(f"{source}: runcard must be a mapping, got {type(data).__name__}")
    _reject_unknown(data, _TOP_LEVEL, source, "top-level key")

    if "system" not in data:
        raise RuncardError(f"{source}: missing required key 'system'")
    system = _build_system(data["system"], source)
    name = str(data.get("name", Path(source).stem))
    domain = _build_domain(data.get("domain"), system, source)
    analysis = _build_analysis(data.get("analysis"), source)
    plots = _build_plots(data.get("plots"), system, source)
    return Runcard(
        name=name, system=system, domain=domain, plots=plots,
        analysis=analysis, source=source,
    )


def _reject_unknown(mapping: dict, allowed: set[str], source: str, what: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise RuncardError(
            f"{source}: unknown {what}(s) {unknown}; allowed are {sorted(allowed)}"
        )


def _build_system(spec: Any, source: str) -> System:
    if not isinstance(spec, dict):
        raise RuncardError(f"{source}: 'system' must be a mapping, got {type(spec).__name__}")
    _reject_unknown(spec, _SYSTEM_KEYS, source, "key under 'system'")

    parameters = spec.get("parameters") or {}
    if not isinstance(parameters, dict):
        raise RuncardError(f"{source}: 'system.parameters' must be a mapping")

    if "model" in spec:
        if "equations" in spec or "variables" in spec:
            raise RuncardError(
                f"{source}: give either 'system.model' or "
                f"'system.variables'/'system.equations', not both"
            )
        return _build_from_model(spec["model"], parameters, source)

    if "equations" not in spec:
        raise RuncardError(
            f"{source}: 'system' needs either 'model' or 'equations'"
        )
    equations = spec["equations"]
    if not isinstance(equations, dict):
        raise RuncardError(f"{source}: 'system.equations' must be a mapping of variable to expression")

    # Variables may be listed explicitly to fix their order; otherwise the
    # equation order decides it, which YAML preserves.
    variables = spec.get("variables") or list(equations)
    if not isinstance(variables, list) or not all(isinstance(v, str) for v in variables):
        raise RuncardError(f"{source}: 'system.variables' must be a list of names")

    try:
        rhs = compile_rhs(variables, equations, parameters)
    except ExpressionError as exc:
        raise RuncardError(f"{source}: {exc}") from None
    return System(rhs=rhs, names=tuple(variables), params=dict(parameters))


def _build_from_model(name: Any, parameters: dict, source: str) -> System:
    if not isinstance(name, str) or name not in models.__all__:
        raise RuncardError(
            f"{source}: unknown model {name!r}; available models are {sorted(models.__all__)}"
        )
    factory = getattr(models, name)
    accepted = set(inspect.signature(factory).parameters)
    unknown = sorted(set(parameters) - accepted)
    if unknown:
        raise RuncardError(
            f"{source}: model {name!r} has no parameter(s) {unknown}; "
            f"it accepts {sorted(accepted)}"
        )
    return factory(**parameters)


def _build_domain(spec: Any, system: System, source: str) -> dict[str, tuple[float, float]]:
    if spec is None:
        raise RuncardError(
            f"{source}: missing required key 'domain' (one [lo, hi] per variable)"
        )
    if not isinstance(spec, dict):
        raise RuncardError(f"{source}: 'domain' must be a mapping of variable to [lo, hi]")

    missing = [n for n in system.names if n not in spec]
    if missing:
        raise RuncardError(f"{source}: 'domain' has no range for variable(s) {missing}")
    extra = sorted(set(spec) - set(system.names))
    if extra:
        raise RuncardError(
            f"{source}: 'domain' names unknown variable(s) {extra}; "
            f"the system has {list(system.names)}"
        )

    domain = {}
    for name in system.names:
        pair = spec[name]
        if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
            raise RuncardError(f"{source}: 'domain.{name}' must be [lo, hi], got {pair!r}")
        lo, hi = (float(pair[0]), float(pair[1]))
        if hi <= lo:
            raise RuncardError(f"{source}: 'domain.{name}' must be increasing, got [{lo}, {hi}]")
        domain[name] = (lo, hi)
    return domain


def _build_analysis(spec: Any, source: str) -> dict[str, Any]:
    if spec is None:
        return {}
    if not isinstance(spec, dict):
        raise RuncardError(f"{source}: 'analysis' must be a mapping")
    _reject_unknown(spec, _ANALYSIS_KEYS, source, "key under 'analysis'")
    return dict(spec)


def _build_plots(spec: Any, system: System, source: str) -> tuple[PlotSpec, ...]:
    if spec is None:
        return ()
    if not isinstance(spec, list):
        raise RuncardError(f"{source}: 'plots' must be a list")

    plots = []
    for i, entry in enumerate(spec):
        where = f"{source}: plots[{i}]"
        if not isinstance(entry, dict):
            raise RuncardError(f"{where} must be a mapping, got {type(entry).__name__}")
        if "type" not in entry:
            raise RuncardError(f"{where} is missing 'type' (one of {list(PLOT_TYPES)})")
        kind = entry["type"]
        if kind not in PLOT_TYPES:
            raise RuncardError(f"{where}: unknown plot type {kind!r}; allowed are {list(PLOT_TYPES)}")
        needed = PLOT_DIMENSIONS[kind]
        if needed is not None and system.ndim != needed:
            raise RuncardError(
                f"{where}: a {kind!r} plot needs a {needed}-variable system, but this "
                f"one has {system.ndim} ({', '.join(system.names)})"
            )
        options = {k: v for k, v in entry.items() if k not in ("type", "output")}
        output = entry.get("output") or f"{Path(source).stem}_{kind}_{i}.png"
        plots.append(PlotSpec(type=kind, output=str(output), options=options))
    return tuple(plots)
