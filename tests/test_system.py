"""Step 1: System construction, parameter handling, integration, plotting."""

import jax.numpy as jnp
import matplotlib
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from popdynamics import System, integrate, models, plot_timeseries


def test_ndim_and_names():
    sys = models.lotka_volterra(alpha=1.0)
    assert sys.ndim == 2
    assert sys.names == ("u", "v")


def test_rhs_matches_hand_evaluation():
    sys = models.lotka_volterra(alpha=2.0)
    # at (u, v) = (2, 3): du/dt = 2*(1-3) = -4, dv/dt = 2*3*(2-1) = 6
    assert jnp.allclose(sys.f([2.0, 3.0]), jnp.array([-4.0, 6.0]))


def test_param_override_does_not_mutate():
    sys = models.lotka_volterra(alpha=1.0)
    override = sys.f([2.0, 3.0], alpha=2.0)
    assert jnp.allclose(override, jnp.array([-4.0, 6.0]))
    assert sys.params["alpha"] == 1.0  # original untouched


def test_with_params_returns_copy():
    sys = models.lotka_volterra(alpha=1.0)
    other = sys.with_params(alpha=5.0)
    assert other.params["alpha"] == 5.0
    assert sys.params["alpha"] == 1.0


def test_with_params_rejects_unknown():
    sys = models.lotka_volterra()
    with pytest.raises(KeyError, match="beta"):
        sys.with_params(beta=1.0)


def test_wrong_state_shape_is_rejected():
    sys = models.lotka_volterra()
    with pytest.raises(ValueError, match="expected"):
        sys.f([1.0, 2.0, 3.0])


def test_duplicate_names_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        System(rhs=lambda y, p: y, names=("u", "u"))


# --- integration ---------------------------------------------------------


@pytest.mark.parametrize("N0", [0.01, 0.5, 1.5, 3.0])
def test_logistic_converges_to_carrying_capacity(N0):
    sys = models.logistic(r=1.0, K=2.0)
    traj = sys.integrate([N0], (0.0, 40.0))
    assert jnp.allclose(traj.y_final, jnp.array([2.0]), atol=1e-6)


def test_logistic_matches_closed_form():
    r, K, N0 = 1.3, 2.0, 0.4
    traj = models.logistic(r=r, K=K).integrate([N0], (0.0, 5.0))
    exact = K / (1.0 + (K / N0 - 1.0) * jnp.exp(-r * traj.ts))
    assert jnp.allclose(traj["N"], exact, atol=1e-7)


def test_lotka_volterra_orbit_is_closed():
    """The conserved H should not drift, so the orbit must return to its start."""
    sys = models.lotka_volterra(alpha=1.0)
    traj = sys.integrate([1.5, 1.0], (0.0, 100.0), n_points=4001)

    H = traj["v"] - jnp.log(traj["v"]) + (traj["u"] - jnp.log(traj["u"]))
    assert jnp.max(jnp.abs(H - H[0])) < 1e-6

    # amplitude must not decay over ~15 periods
    first, last = traj["u"][:400], traj["u"][-400:]
    assert abs(float(jnp.max(first) - jnp.max(last))) < 1e-4


def test_fixed_point_stays_fixed():
    traj = models.lotka_volterra().integrate([1.0, 1.0], (0.0, 50.0))
    assert jnp.allclose(traj.ys, 1.0, atol=1e-8)


def test_trajectory_indexing():
    traj = models.lotka_volterra().integrate([1.5, 1.0], (0.0, 10.0))
    assert traj["u"].shape == traj.ts.shape
    assert jnp.allclose(traj["u"], traj.ys[:, 0])
    with pytest.raises(KeyError, match="no variable"):
        traj["w"]


def test_integrate_function_and_method_agree():
    sys = models.logistic()
    a = integrate(sys, [0.1], (0.0, 5.0))
    b = sys.integrate([0.1], (0.0, 5.0))
    assert jnp.allclose(a.ys, b.ys)


def test_bad_t_span_rejected():
    with pytest.raises(ValueError, match="increasing"):
        models.logistic().integrate([0.1], (5.0, 0.0))


# --- plotting ------------------------------------------------------------


def test_plot_timeseries_returns_axes():
    traj = models.lotka_volterra().integrate([1.5, 1.0], (0.0, 20.0))
    ax = plot_timeseries(traj)
    assert len(ax.get_lines()) == 2
    plt.close("all")


def test_plot_timeseries_composes_onto_given_axes():
    traj = models.lotka_volterra().integrate([1.5, 1.0], (0.0, 20.0))
    _, ax = plt.subplots()
    plot_timeseries(traj, ax=ax, variables=("u",))
    plot_timeseries(traj, ax=ax, variables=("v",), linestyle="--")
    assert len(ax.get_lines()) == 2
    plt.close("all")
