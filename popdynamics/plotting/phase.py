"""Phase-plane plotting: vector fields, nullclines, fixed points, trajectories.

Each function draws one layer onto an axes and returns it, so they compose;
:func:`plot_phase_portrait` is a convenience that calls them in order.

The layers are deliberately ranked by visual weight, so the picture reads from
background to foreground: a recessive grey vector field, two hues for the
nullclines, and near-black ink for trajectories and fixed points -- the things
you actually look at. Stability is carried by marker *shape* as well as fill, so
it survives greyscale printing and colour-vision deficiency.
"""

from __future__ import annotations

from typing import Any, Sequence

import jax.numpy as jnp
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from popdynamics.analysis import classify, fixed_points
from popdynamics.system import System

__all__ = [
    "plot_fixed_points",
    "plot_nullclines",
    "plot_phase_portrait",
    "plot_trajectories",
    "plot_vector_field",
]

# Categorical slots 1 and 2 of the reference palette, used only for the two
# nullclines. Validated all-pairs on a light surface: CVD deltaE 24.7 (target 8),
# normal-vision 33.6 (floor 15), both above 3:1 contrast.
NULLCLINE_COLORS = ("#2a78d6", "#eb6834")
FIELD_COLOR = "#a8a7a0"  # recessive: direction is carried by geometry, not hue
INK = "#0b0b0b"
SURFACE = "#ffffff"

# Single-hue sequential ramp (blue, light -> dark) for speed magnitude. Arrows
# are discrete marks rather than a continuous field, so the ramp starts at step
# 250 instead of 100: the lighter steps recede into the surface and the slow
# arrows near a fixed point -- exactly the ones worth seeing -- disappear.
SPEED_CMAP = LinearSegmentedColormap.from_list(
    "popdyn_speed",
    ["#86b6ef", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)


def _require_2d(system: System) -> None:
    if system.ndim != 2:
        raise ValueError(
            f"phase-plane plots need a 2-variable system, got {system.ndim} "
            f"({', '.join(system.names)}). Use plot_timeseries for higher dimensions."
        )


def _grid(system: System, xlim, ylim, n: int):
    """Evaluate the right-hand side on an ``n x n`` mesh in one batched call."""
    x = jnp.linspace(float(xlim[0]), float(xlim[1]), n)
    y = jnp.linspace(float(ylim[0]), float(ylim[1]), n)
    X, Y = jnp.meshgrid(x, y)  # (n, n), row index is y
    points = jnp.stack([X.ravel(), Y.ravel()], axis=-1)
    F = system.f_fn_batched(points, system.params).reshape(n, n, 2)
    return np.asarray(X), np.asarray(Y), np.asarray(F[..., 0]), np.asarray(F[..., 1])


def _style(ax: Axes, system: System) -> None:
    ax.set_xlabel(f"${system.names[0]}$")
    ax.set_ylabel(f"${system.names[1]}$")
    ax.grid(True, color="#e8e7e3", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9c8c2")


def plot_vector_field(
    system: System,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    ax: Axes | None = None,
    *,
    density: int = 21,
    normalize: bool = True,
    color_by_speed: bool = False,
    streamlines: bool = False,
    **kwargs: Any,
) -> Axes:
    """Draw the vector field ``(du/dt, dv/dt)`` over the plane.

    Arrows are normalised to a common length by default so direction stays
    readable where the flow is slow; that discards magnitude, which
    ``color_by_speed=True`` puts back as a colour ramp. ``streamlines=True``
    switches to a streamplot instead, which shows magnitude through line density.
    """
    _require_2d(system)
    if ax is None:
        _, ax = plt.subplots()

    X, Y, U, V = _grid(system, xlim, ylim, density)
    speed = np.hypot(U, V)

    if streamlines:
        ax.streamplot(
            X[0],
            Y[:, 0],
            U,
            V,
            color=speed if color_by_speed else FIELD_COLOR,
            cmap=SPEED_CMAP if color_by_speed else None,
            linewidth=0.9,
            arrowsize=0.9,
            density=kwargs.pop("stream_density", 1.2),
            zorder=1,
            **kwargs,
        )
    else:
        if normalize:
            # Guard the zero-speed cells at fixed points, where the direction is
            # undefined and dividing through would give NaN.
            scale = np.where(speed > 0, speed, 1.0)
            U, V = U / scale, V / scale
        args = (speed,) if color_by_speed else ()
        ax.quiver(
            X,
            Y,
            U,
            V,
            *args,
            cmap=SPEED_CMAP if color_by_speed else None,
            color=None if color_by_speed else FIELD_COLOR,
            # angles="xy" takes the direction from the data-space displacement,
            # so arrows stay visually tangent to trajectories even when the two
            # axes span very different ranges.
            angles="xy",
            scale_units="width",
            scale=density * 1.5 if normalize else None,
            width=0.0035,
            pivot="mid",
            zorder=1,
            **kwargs,
        )
    _style(ax, system)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    return ax


def plot_nullclines(
    system: System,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    ax: Axes | None = None,
    *,
    resolution: int = 401,
    colors: Sequence[str] = NULLCLINE_COLORS,
    legend: bool = True,
    **kwargs: Any,
) -> Axes:
    """Draw the curves where each component of the flow vanishes.

    On the ``u``-nullcline the flow is purely vertical, on the ``v``-nullcline
    purely horizontal, so fixed points are exactly where the two intersect.
    """
    _require_2d(system)
    if ax is None:
        _, ax = plt.subplots()

    X, Y, U, V = _grid(system, xlim, ylim, resolution)
    handles = []
    for i, (component, name) in enumerate(zip((U, V), system.names)):
        color = colors[i % len(colors)]
        ax.contour(
            X,
            Y,
            component,
            levels=[0.0],
            colors=[color],
            linewidths=kwargs.pop("linewidths", 2.0),
            zorder=2,
        )
        # Contours produce no legend handles of their own.
        handles.append(
            Line2D([], [], color=color, lw=2.0, label=rf"$\dot{{{name}}} = 0$")
        )

    _style(ax, system)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    if legend:
        _merge_legend(ax, handles)
    return ax


# Stability -> (matplotlib marker, fill style, legend label). Shape carries the
# distinction, so the encoding survives greyscale and colour-vision deficiency.
_MARKERS = {
    "stable": ("o", "full", "stable"),
    "unstable": ("o", "none", "unstable"),
    "saddle": ("X", "full", "saddle"),
    "marginal": ("o", "left", "marginal / centre"),
}


def plot_fixed_points(
    system: System,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    ax: Axes | None = None,
    *,
    color: str = INK,
    markersize: float = 10.0,
    legend: bool = True,
    annotate: bool = False,
    **kwargs: Any,
) -> Axes:
    """Find the fixed points inside the view and mark them by stability.

    Extra keyword arguments go to :func:`popdynamics.fixed_points`, so the seed
    density and tolerances of the search are reachable from here.
    """
    _require_2d(system)
    if ax is None:
        _, ax = plt.subplots()

    points = fixed_points(system, [tuple(xlim), tuple(ylim)], **kwargs)
    seen: dict[str, str] = {}
    for point in points:
        fp = classify(system, point)
        marker, fill, label = _MARKERS[fp.stability]
        ax.plot(
            float(point[0]),
            float(point[1]),
            marker=marker,
            fillstyle=fill,
            markersize=markersize,
            markerfacecolor=color,
            markeredgecolor=color,
            markeredgewidth=1.6,
            linestyle="none",
            zorder=5,
            # A fixed point sitting exactly on the view boundary is still a fixed
            # point: draw it whole rather than letting the axes clip it in half.
            clip_on=False,
            # A surface-coloured halo keeps markers legible on top of arrows.
            path_effects=[pe.Stroke(linewidth=3.0, foreground=SURFACE), pe.Normal()],
        )
        seen[fp.stability] = label
        if annotate:
            ax.annotate(
                fp.label,
                (float(point[0]), float(point[1])),
                textcoords="offset points",
                xytext=(9, 6),
                fontsize=8,
                color="#52514e",
                zorder=6,
            )

    _style(ax, system)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    if legend and seen:
        handles = [
            Line2D(
                [],
                [],
                color=color,
                marker=_MARKERS[s][0],
                fillstyle=_MARKERS[s][1],
                markerfacecolor=color,
                markeredgecolor=color,
                markersize=markersize * 0.8,
                linestyle="none",
                label=seen[s],
            )
            for s in _MARKERS
            if s in seen
        ]
        _merge_legend(ax, handles)
    return ax


def plot_trajectories(
    system: System,
    initial_conditions: Sequence[Sequence[float]],
    t_span: tuple[float, float] = (0.0, 50.0),
    ax: Axes | None = None,
    *,
    color: str = INK,
    linewidth: float = 1.6,
    alpha: float = 0.85,
    arrows: bool = True,
    start_markers: bool = True,
    **kwargs: Any,
) -> Axes:
    """Integrate each initial condition and draw its orbit in the plane.

    All orbits share one colour: they are the same kind of object, and colouring
    them individually would imply a distinction that does not exist.
    """
    _require_2d(system)
    if ax is None:
        _, ax = plt.subplots()

    y0s = np.atleast_2d(np.asarray(initial_conditions, dtype=float))
    for y0 in y0s:
        traj = system.integrate(y0, t_span, **kwargs)
        u, v = np.asarray(traj.ys[:, 0]), np.asarray(traj.ys[:, 1])
        ax.plot(
            u,
            v,
            color=color,
            lw=linewidth,
            alpha=alpha,
            zorder=3,
            solid_capstyle="round",
        )
        if start_markers:
            ax.plot(
                u[0],
                v[0],
                marker="o",
                markersize=3.5,
                color=color,
                alpha=alpha,
                zorder=4,
            )
        if arrows and len(u) > 2:
            _direction_arrow(ax, u, v, color, alpha, linewidth)

    _style(ax, system)
    return ax


def _direction_arrow(ax: Axes, u, v, color: str, alpha: float, lw: float = 1.6) -> None:
    """Mark the direction of travel partway along an orbit."""
    i = len(u) // 3
    # Step forward until the points differ, so a slow orbit still gets an arrow.
    j = next((k for k in range(i + 1, len(u)) if (u[k], v[k]) != (u[i], v[i])), None)
    if j is None:
        return
    ax.annotate(
        "",
        xy=(u[j], v[j]),
        xytext=(u[i], v[i]),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            alpha=alpha,
            lw=lw,
            mutation_scale=15,
            shrinkA=0,
            shrinkB=0,
        ),
        zorder=4,
    )


def _merge_legend(ax: Axes, handles: list[Line2D]) -> None:
    """Add handles to the axes legend without dropping what is already there."""
    existing, _ = ax.get_legend_handles_labels()
    if ax.get_legend() is not None:
        existing = list(ax.get_legend().legend_handles)
    combined, labels = [], []
    for h in list(existing) + list(handles):
        label = h.get_label()
        if label and not label.startswith("_") and label not in labels:
            combined.append(h)
            labels.append(label)
    ax.legend(
        combined,
        labels,
        loc="best",
        frameon=True,
        framealpha=0.9,
        edgecolor="#dedcd6",
        fontsize=9,
    )


def plot_phase_portrait(
    system: System,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    ax: Axes | None = None,
    *,
    initial_conditions: Sequence[Sequence[float]] | None = None,
    t_span: tuple[float, float] = (0.0, 50.0),
    vector_field: bool = True,
    nullclines: bool = True,
    show_fixed_points: bool = True,
    title: str | None = None,
    **kwargs: Any,
) -> Axes:
    """Draw a full phase portrait: vector field, nullclines, orbits, fixed points.

    Layers are drawn back to front so the fixed points always land on top.
    """
    _require_2d(system)
    if ax is None:
        _, ax = plt.subplots(figsize=(6.0, 5.2))

    if vector_field:
        plot_vector_field(system, xlim, ylim, ax, **kwargs.pop("field_kwargs", {}))
    if nullclines:
        plot_nullclines(
            system, xlim, ylim, ax, legend=False, **kwargs.pop("nullcline_kwargs", {})
        )
    if initial_conditions is not None:
        plot_trajectories(
            system,
            initial_conditions,
            t_span,
            ax,
            **kwargs.pop("trajectory_kwargs", {}),
        )
    if show_fixed_points:
        plot_fixed_points(
            system, xlim, ylim, ax, legend=False, **kwargs.pop("fixed_point_kwargs", {})
        )

    # One legend at the end, so the layers do not each rebuild it.
    handles = []
    if nullclines:
        handles += [
            Line2D(
                [],
                [],
                color=NULLCLINE_COLORS[i % 2],
                lw=2.0,
                label=rf"$\dot{{{name}}} = 0$",
            )
            for i, name in enumerate(system.names)
        ]
    if show_fixed_points:
        present = {
            classify(system, p).stability
            for p in fixed_points(system, [tuple(xlim), tuple(ylim)])
        }
        handles += [
            Line2D(
                [],
                [],
                color=INK,
                marker=_MARKERS[s][0],
                fillstyle=_MARKERS[s][1],
                markerfacecolor=INK,
                markeredgecolor=INK,
                markersize=7.2,
                linestyle="none",
                label=_MARKERS[s][2],
            )
            for s in _MARKERS
            if s in present
        ]
    if handles:
        _merge_legend(ax, handles)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    if title:
        ax.set_title(title)
    return ax
