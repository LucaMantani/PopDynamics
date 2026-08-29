"""Step 6: the model library.

Every model has fixed points and stability conditions derivable in closed form,
so these assert against the algebra rather than against recorded output.
"""

import inspect

import jax.numpy as jnp
import pytest

from popdynamics import models
from popdynamics.runcard import from_dict

QUADRANT = [(0.0, 4.0), (0.0, 4.0)]


def _interior(system, bounds=QUADRANT, tol=1e-6):
    """The fixed point with both coordinates strictly positive, if there is one."""
    found = [
        p for p in system.fixed_points(bounds)
        if float(p[0]) > tol and float(p[1]) > tol
    ]
    return found[0] if found else None


# --- the library as a whole ---------------------------------------------


def test_every_exported_model_builds_with_defaults():
    for name in models.__all__:
        system = getattr(models, name)()
        assert system.ndim == len(system.names) >= 1
        assert jnp.all(jnp.isfinite(system.f([0.5] * system.ndim)))


def test_every_model_parameter_is_reachable_and_used():
    """Factory arguments must land in params, or with_params cannot reach them."""
    for name in models.__all__:
        factory = getattr(models, name)
        system = factory()
        assert set(inspect.signature(factory).parameters) == set(system.params)


def test_every_model_is_available_to_a_runcard():
    """The 'model:' shorthand resolves against models.__all__."""
    for name in models.__all__:
        system = getattr(models, name)()
        card = from_dict({
            "system": {"model": name},
            "domain": {n: [0.0, 2.0] for n in system.names},
        })
        assert card.system.names == system.names


# --- competition ---------------------------------------------------------


@pytest.mark.parametrize("a12, a21", [(0.5, 0.5), (0.3, 0.8), (1.5, 1.5), (2.0, 1.2)])
def test_competition_interior_point_matches_the_algebra(a12, a21):
    """u* = (1-a12)/(1-a12 a21), v* = (1-a21)/(1-a12 a21)."""
    system = models.competition(a12=a12, a21=a21)
    denominator = 1.0 - a12 * a21
    expected = jnp.array([(1 - a12) / denominator, (1 - a21) / denominator])
    assert jnp.allclose(_interior(system), expected, atol=1e-8)


def test_weak_competition_gives_stable_coexistence():
    system = models.competition(a12=0.5, a21=0.5)
    fp = system.classify(_interior(system))
    assert fp.label == "stable node"
    assert jnp.allclose(fp.point, jnp.array([2 / 3, 2 / 3]), atol=1e-8)


def test_strong_competition_gives_a_saddle_and_two_winners():
    """Bistability: the interior point is a saddle, both axes points stable."""
    system = models.competition(a12=1.5, a21=1.5)
    assert system.classify(_interior(system)).label == "saddle"
    for point in ([1.0, 0.0], [0.0, 1.0]):
        assert system.classify(point).stability == "stable"


def test_competition_stability_follows_the_determinant_sign():
    """det J = rho u* v* (1 - a12 a21), and Tr J < 0, so a12 a21 < 1 is the test."""
    for a12, a21, expected in [(0.5, 0.5, "stable"), (1.5, 1.5, "saddle")]:
        system = models.competition(a12=a12, a21=a21, rho=1.3)
        fp = system.classify(_interior(system))
        u, v = float(fp.point[0]), float(fp.point[1])
        assert fp.trace == pytest.approx(-(u + 1.3 * v))
        assert fp.det == pytest.approx(1.3 * u * v * (1 - a12 * a21))
        assert fp.stability == expected


def test_asymmetric_competition_has_no_interior_equilibrium():
    """a12 < 1 < a21 puts the crossing outside the positive quadrant: u wins."""
    system = models.competition(a12=0.5, a21=1.5)
    assert _interior(system) is None
    assert system.classify([1.0, 0.0]).stability == "stable"
    assert system.classify([0.0, 1.0]).stability == "saddle"


def test_rho_rescales_time_without_moving_the_fixed_points():
    slow = models.competition(a12=0.5, a21=0.5, rho=0.2)
    fast = models.competition(a12=0.5, a21=0.5, rho=3.0)
    assert jnp.allclose(_interior(slow), _interior(fast), atol=1e-8)


# --- mutualism -----------------------------------------------------------


@pytest.mark.parametrize("a12, a21", [(0.5, 0.5), (0.2, 0.9), (0.8, 0.4)])
def test_mutualism_interior_point_matches_the_algebra(a12, a21):
    """u* = (1+a12)/(1-a12 a21), v* = (1+a21)/(1-a12 a21)."""
    system = models.mutualism(a12=a12, a21=a21)
    denominator = 1.0 - a12 * a21
    expected = jnp.array([(1 + a12) / denominator, (1 + a21) / denominator])
    assert jnp.allclose(_interior(system, [(0.0, 12.0)] * 2), expected, atol=1e-8)


def test_mutualism_is_stable_while_the_product_is_below_one():
    system = models.mutualism(a12=0.5, a21=0.5)
    fp = system.classify(_interior(system))
    assert jnp.allclose(fp.point, jnp.array([2.0, 2.0]), atol=1e-8)
    assert fp.label == "stable node"


def test_runaway_mutualism_has_no_positive_equilibrium():
    """a12 a21 > 1: mutual benefit outruns self-limitation, nothing to settle at."""
    system = models.mutualism(a12=1.5, a21=1.5)
    assert _interior(system, [(0.0, 50.0)] * 2) is None
    # ...and the flow points outward everywhere in the positive quadrant
    assert jnp.all(system.f([5.0, 5.0]) > 0)


def test_mutualism_helps_where_competition_hurts():
    """Same form, flipped signs: each species raises the other's equilibrium."""
    together = _interior(models.mutualism(a12=0.5, a21=0.5))
    assert float(together[0]) > 1.0  # above the single-species carrying capacity


# --- r_plus_sN -----------------------------------------------------------


def test_r_plus_sN_carrying_capacity():
    system = models.r_plus_sN(r=1.0, s=-0.5)  # K = -r/s = 2
    assert system.classify([2.0]).stability == "stable"
    assert system.classify([0.0]).stability == "unstable"


def test_r_plus_sN_allee_threshold():
    system = models.r_plus_sN(r=-1.0, s=1.0)  # threshold at -r/s = 1
    assert system.classify([1.0]).stability == "unstable"
    assert system.classify([0.0]).stability == "stable"


def test_r_plus_sN_agrees_with_logistic():
    """r N + s N^2 with s = -r/K is the logistic equation."""
    r, K = 1.3, 2.0
    quadratic = models.r_plus_sN(r=r, s=-r / K)
    assert jnp.allclose(quadratic.f([0.7]), models.logistic(r=r, K=K).f([0.7]))
