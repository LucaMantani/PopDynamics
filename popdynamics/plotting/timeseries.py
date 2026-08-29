"""Time evolution of the state variables."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from popdynamics.system import Trajectory

__all__ = ["plot_timeseries"]


def plot_timeseries(
    traj: Trajectory,
    ax: Axes | None = None,
    *,
    variables: tuple[str, ...] | None = None,
    legend: bool = True,
    **kwargs: Any,
) -> Axes:
    """Plot each state variable against time.

    Parameters
    ----------
    traj
        A solved trajectory, from :func:`popdynamics.integrate`.
    ax
        Axes to draw on; a new figure is created when omitted.
    variables
        Subset of variables to draw. Defaults to all of them.
    **kwargs
        Forwarded to ``ax.plot``.
    """
    if ax is None:
        _, ax = plt.subplots()

    names = traj.names if variables is None else tuple(variables)
    for name in names:
        ax.plot(traj.ts, traj[name], label=name, **kwargs)

    ax.set_xlabel("$t$")
    ax.set_ylabel("population")
    ax.set_xlim(float(traj.ts[0]), float(traj.ts[-1]))
    if legend:
        ax.legend()
    return ax
