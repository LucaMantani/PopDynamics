"""Plotting helpers.

Every function accepts an optional ``ax=`` and returns it, so they compose onto
a shared axes rather than each owning a figure.
"""

from popdynamics.plotting.timeseries import plot_timeseries

__all__ = ["plot_timeseries"]
