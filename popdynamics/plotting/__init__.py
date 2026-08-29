"""Plotting helpers.

Every function accepts an optional ``ax=`` and returns it, so they compose onto
a shared axes rather than each owning a figure.
"""

from popdynamics.plotting.phase import (
    plot_fixed_points,
    plot_nullclines,
    plot_phase_portrait,
    plot_trajectories,
    plot_vector_field,
)
from popdynamics.plotting.timeseries import plot_timeseries

__all__ = [
    "plot_timeseries",
    "plot_vector_field",
    "plot_nullclines",
    "plot_fixed_points",
    "plot_trajectories",
    "plot_phase_portrait",
]
