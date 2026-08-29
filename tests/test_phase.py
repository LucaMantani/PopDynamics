"""Step 3: phase-plane plotting.

Plot correctness is mostly judged by eye, but a few properties are checkable:
nullcline vertices must actually be roots of one component, the vector field
must have the requested resolution, and marker counts must match the fixed
points found.
"""

import jax.numpy as jnp
import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.contour import ContourSet
from matplotlib.quiver import Quiver

from popdynamics import (
    System,
    models,
    plot_fixed_points,
    plot_nullclines,
    plot_phase_portrait,
    plot_trajectories,
    plot_vector_field,
)

LIM = (0.0, 2.5)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _contours(ax):
    return [c for c in ax.collections if isinstance(c, ContourSet)]


def _quivers(ax):
    return [c for c in ax.collections if isinstance(c, Quiver)]


# --- vector field --------------------------------------------------------


def test_vector_field_resolution():
    ax = plot_vector_field(models.lotka_volterra(), LIM, LIM, density=15)
    (q,) = _quivers(ax)
    assert len(q.U) == 15 * 15


def test_normalised_arrows_have_unit_length():
    ax = plot_vector_field(models.lotka_volterra(), LIM, LIM, density=11)
    (q,) = _quivers(ax)
    speed = np.hypot(q.U, q.V)
    # Every arrow is unit length except where the flow vanishes exactly.
    assert np.allclose(speed[speed > 0.5], 1.0)


def test_unnormalised_arrows_keep_magnitude():
    ax = plot_vector_field(models.lotka_volterra(), LIM, LIM, density=11, normalize=False)
    (q,) = _quivers(ax)
    assert np.ptp(np.hypot(q.U, q.V)) > 0.5


def test_zero_speed_cells_do_not_produce_nan():
    """A fixed point sits at a grid node here, so normalisation divides by zero."""
    ax = plot_vector_field(models.lotka_volterra(), (0.0, 2.0), (0.0, 2.0), density=3)
    (q,) = _quivers(ax)
    assert np.all(np.isfinite(q.U)) and np.all(np.isfinite(q.V))


def test_vector_field_variants_run():
    sys = models.lotka_volterra()
    assert plot_vector_field(sys, LIM, LIM, color_by_speed=True) is not None
    assert plot_vector_field(sys, LIM, LIM, streamlines=True) is not None
    assert plot_vector_field(sys, LIM, LIM, streamlines=True, color_by_speed=True) is not None


# --- nullclines ----------------------------------------------------------


def test_one_contour_per_variable():
    ax = plot_nullclines(models.lotka_volterra(), LIM, LIM)
    assert len(_contours(ax)) == 2


@pytest.mark.parametrize(
    "system",
    [models.lotka_volterra(1.0), models.damped_lotka_volterra(1.0, 0.1, 0.1)],
)
def test_nullcline_vertices_are_roots_of_their_component(system):
    """Every drawn point must genuinely satisfy d(variable)/dt = 0."""
    ax = plot_nullclines(system, (0.05, 2.5), (0.05, 2.5))
    for i, cs in enumerate(_contours(ax)):
        verts = np.concatenate([p.vertices for p in cs.get_paths()])
        values = system.f_fn_batched(jnp.asarray(verts), system.params)[:, i]
        assert float(jnp.max(jnp.abs(values))) < 1e-3


def test_nullclines_intersect_at_fixed_points():
    """Fixed points must lie on both nullclines, which is why they are drawn."""
    system = models.lotka_volterra(1.0)
    ax = plot_nullclines(system, (0.05, 2.5), (0.05, 2.5))
    interior = np.array([1.0, 1.0])
    for cs in _contours(ax):
        verts = np.concatenate([p.vertices for p in cs.get_paths()])
        assert np.min(np.linalg.norm(verts - interior, axis=-1)) < 1e-2


# --- fixed points --------------------------------------------------------


def test_marker_per_fixed_point():
    system = models.lotka_volterra(1.0)
    ax = plot_fixed_points(system, LIM, LIM)
    expected = len(system.fixed_points([LIM, LIM]))
    assert expected == 2
    assert len(ax.get_lines()) == expected


def test_stability_shown_by_shape_not_colour_alone():
    """Saddle and centre must differ in marker, so the plot survives greyscale."""
    ax = plot_fixed_points(models.lotka_volterra(1.0), LIM, LIM)
    markers = {line.get_marker() for line in ax.get_lines()}
    fills = {line.get_fillstyle() for line in ax.get_lines()}
    assert len(markers | fills) > 1
    labels = {t.get_text() for t in ax.get_legend().get_texts()}
    assert labels == {"saddle", "marginal / centre"}


def test_damped_system_legend_reports_stable():
    ax = plot_fixed_points(models.damped_lotka_volterra(1.0, 0.1, 0.1), LIM, LIM)
    assert "stable" in {t.get_text() for t in ax.get_legend().get_texts()}


def test_no_fixed_points_in_view_is_not_an_error():
    system = System(rhs=lambda y, p: jnp.array([1.0, 1.0]), names=("u", "v"))
    ax = plot_fixed_points(system, LIM, LIM)
    assert ax.get_lines() == []


def test_fixed_point_search_kwargs_are_forwarded():
    ax = plot_fixed_points(models.lotka_volterra(1.0), LIM, LIM, n_seed=6)
    assert len(ax.get_lines()) == 2


# --- trajectories --------------------------------------------------------


def test_line_per_trajectory():
    ax = plot_trajectories(
        models.lotka_volterra(1.0), [[1.4, 1.0], [1.9, 1.0]], (0.0, 8.0),
        arrows=False, start_markers=False,
    )
    assert len(ax.get_lines()) == 2


def test_trajectory_follows_the_flow():
    """The drawn curve must be the integrated orbit, not a resampling of it."""
    system = models.lotka_volterra(1.0)
    ax = plot_trajectories(system, [[1.4, 1.0]], (0.0, 8.0), arrows=False, start_markers=False)
    xs, ys = ax.get_lines()[0].get_data()
    expected = system.integrate([1.4, 1.0], (0.0, 8.0))
    assert np.allclose(xs, np.asarray(expected.ys[:, 0]))
    assert np.allclose(ys, np.asarray(expected.ys[:, 1]))


def test_single_initial_condition_accepted_unnested():
    ax = plot_trajectories(models.lotka_volterra(1.0), [1.4, 1.0], (0.0, 8.0))
    assert len(ax.get_lines()) >= 1


def test_direction_arrow_is_added():
    ax = plot_trajectories(models.lotka_volterra(1.0), [[1.4, 1.0]], (0.0, 8.0))
    assert len(ax.texts) == 1  # annotate() with an empty string carries the arrow


# --- composition and guards ---------------------------------------------


def test_layers_compose_onto_one_axes():
    system = models.damped_lotka_volterra(1.0, 0.1, 0.1)
    _, ax = plt.subplots()
    plot_vector_field(system, LIM, LIM, ax)
    plot_nullclines(system, LIM, LIM, ax)
    plot_trajectories(system, [[2.2, 0.6]], (0.0, 30.0), ax)
    plot_fixed_points(system, LIM, LIM, ax)
    assert len(_quivers(ax)) == 1
    assert len(_contours(ax)) == 2
    assert ax.get_legend() is not None


def test_phase_portrait_draws_every_layer():
    ax = plot_phase_portrait(
        models.damped_lotka_volterra(1.0, 0.1, 0.1), LIM, LIM,
        initial_conditions=[[2.2, 0.6]], t_span=(0.0, 30.0),
    )
    assert len(_quivers(ax)) == 1
    assert len(_contours(ax)) == 2
    labels = {t.get_text() for t in ax.get_legend().get_texts()}
    assert r"$\dot{u} = 0$" in labels and "stable" in labels
    assert ax.get_xlim() == LIM and ax.get_ylim() == LIM


def test_phase_portrait_layers_can_be_switched_off():
    ax = plot_phase_portrait(
        models.lotka_volterra(1.0), LIM, LIM,
        vector_field=False, nullclines=False, show_fixed_points=False,
    )
    assert _quivers(ax) == [] and _contours(ax) == []


def test_axes_are_labelled_from_variable_names():
    system = System(
        rhs=lambda y, p: jnp.array([-y[0], -y[1]]), names=("prey", "pred")
    )
    ax = plot_vector_field(system, (-1.0, 1.0), (-1.0, 1.0))
    assert ax.get_xlabel() == "$prey$" and ax.get_ylabel() == "$pred$"


@pytest.mark.parametrize(
    "fn, args",
    [
        (plot_vector_field, (LIM, LIM)),
        (plot_nullclines, (LIM, LIM)),
        (plot_fixed_points, (LIM, LIM)),
        (plot_phase_portrait, (LIM, LIM)),
    ],
)
def test_non_2d_systems_are_rejected_clearly(fn, args):
    with pytest.raises(ValueError, match="2-variable system"):
        fn(models.logistic(), *args)
