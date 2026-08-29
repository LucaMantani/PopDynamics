"""``pop-dyn``: run a runcard and write out its figures and analysis report.

A thin driver. It parses arguments, dispatches each requested plot to the same
public plotting functions the Python API exposes, and formats the report -- no
analysis of its own, so the two front-ends cannot disagree.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # a CLI writes files; it never opens a window
import matplotlib.pyplot as plt

from popdynamics.analysis import classify, fixed_points
from popdynamics.plotting import plot_phase_portrait, plot_timeseries
from popdynamics.runcard import Runcard, RuncardError, load
from popdynamics.system import System

__all__ = ["main", "run"]


def _as_options(value: Any, name: str) -> dict[str, Any] | None:
    """A layer may be given as ``true``/``false`` or as a mapping of options."""
    if value is None or value is True:
        return {}
    if value is False:
        return None
    if isinstance(value, dict):
        return dict(value)
    raise RuncardError(f"'{name}' must be true, false, or a mapping of options, got {value!r}")


def _phase_portrait(runcard: Runcard, options: dict[str, Any]):
    (xlim, ylim) = runcard.bounds
    trajectories = options.get("trajectories") or {}
    if not isinstance(trajectories, dict):
        raise RuncardError("'trajectories' must be a mapping with 'initial_conditions'")

    field = _as_options(options.get("vector_field"), "vector_field")
    nulls = _as_options(options.get("nullclines"), "nullclines")
    points = _as_options(options.get("fixed_points"), "fixed_points")

    figsize = tuple(options.get("figsize", (6.4, 5.4)))
    _, ax = plt.subplots(figsize=figsize)
    plot_phase_portrait(
        runcard.system, xlim, ylim, ax,
        initial_conditions=trajectories.get("initial_conditions"),
        t_span=tuple(trajectories.get("t_span", (0.0, 50.0))),
        vector_field=field is not None,
        nullclines=nulls is not None,
        show_fixed_points=points is not None,
        title=options.get("title", runcard.name),
        field_kwargs=field or {},
        nullcline_kwargs=nulls or {},
        fixed_point_kwargs=points or {},
        trajectory_kwargs=trajectories.get("options", {}),
    )
    return ax.figure


def _timeseries(runcard: Runcard, options: dict[str, Any]):
    y0s = options.get("initial_conditions")
    if not y0s:
        raise RuncardError("a 'timeseries' plot needs 'initial_conditions'")
    t_span = tuple(options.get("t_span", (0.0, 50.0)))
    figsize = tuple(options.get("figsize", (7.0, 4.2)))

    _, ax = plt.subplots(figsize=figsize)
    for i, y0 in enumerate(y0s):
        # Only the first orbit contributes a legend, or every variable would be
        # listed once per initial condition.
        plot_timeseries(runcard.system.integrate(y0, t_span), ax=ax, legend=(i == 0))
    ax.set_title(options.get("title", runcard.name))
    return ax.figure


_DISPATCH = {"phase_portrait": _phase_portrait, "timeseries": _timeseries}


def _report(runcard: Runcard) -> str:
    """Table of fixed points with eigenvalues and stability classification."""
    system = runcard.system
    box = ", ".join(
        f"{n} in [{lo:g}, {hi:g}]" for n, (lo, hi) in zip(system.names, runcard.bounds)
    )
    lines = [f"{runcard.name}", "", f"Fixed points ({box}):"]
    points = fixed_points(system, runcard.bounds)
    if len(points) == 0:
        lines.append("  none found in this domain")
    for point in points:
        lines.append(f"  {classify(system, point)}")
    return "\n".join(lines)


def run(runcard: Runcard, outdir: Path) -> list[Path]:
    """Execute a runcard, returning every path written."""
    outdir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    if runcard.analysis.get("fixed_points"):
        text = _report(runcard)
        print(text)
        print()
        path = outdir / f"{runcard.name}_report.txt"
        path.write_text(text + "\n")
        written.append(path)

    for spec in runcard.plots:
        figure = _DISPATCH[spec.type](runcard, spec.options)
        figure.tight_layout()
        path = outdir / spec.output
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=150)
        plt.close(figure)
        written.append(path)

    for path in written:
        print(f"wrote {path}")
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pop-dyn",
        description="Run a population-dynamics runcard and write its figures.",
    )
    parser.add_argument("runcard", type=Path, help="path to a runcard YAML file")
    parser.add_argument(
        "-o", "--outdir", type=Path, default=None,
        help="directory for outputs (default: the runcard's own directory)",
    )
    args = parser.parse_args(argv)

    try:
        card = load(args.runcard)
        run(card, args.outdir if args.outdir is not None else args.runcard.parent)
    except RuncardError as exc:
        # A bad runcard is a user error, not a crash: report it as one line.
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
