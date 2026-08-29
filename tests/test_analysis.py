"""Step 2: Jacobians, fixed points and linear stability.

Every assertion here is checked against a result derivable by hand, so a
regression shows up as a disagreement with the theory, not with a snapshot.
"""

import jax.numpy as jnp
import pytest

from popdynamics import System, classify, eigenvalues, fixed_points, jacobian, models

# --- Jacobians -----------------------------------------------------------


def test_logistic_jacobian_matches_derivative():
    r, K = 1.3, 2.0
    sys = models.logistic(r=r, K=K)
    # f(N) = rN(1 - N/K)  =>  f'(N) = r(1 - 2N/K)
    for N in (0.0, 0.5, K, 3.0):
        assert jnp.allclose(jacobian(sys, [N]), jnp.array([[r * (1 - 2 * N / K)]]))


def test_logistic_slopes_at_fixed_points():
    """The one-variable criterion: f'(0) = r > 0, f'(K) = -r < 0."""
    r, K = 1.3, 2.0
    sys = models.logistic(r=r, K=K)
    assert float(jacobian(sys, [0.0])[0, 0]) == pytest.approx(r)
    assert float(jacobian(sys, [K])[0, 0]) == pytest.approx(-r)


def test_lotka_volterra_jacobian_by_hand():
    alpha = 2.0
    sys = models.lotka_volterra(alpha=alpha)
    u, v = 1.5, 0.7
    # J = [[1-v, -u], [alpha v, alpha(u-1)]]
    expected = jnp.array([[1 - v, -u], [alpha * v, alpha * (u - 1)]])
    assert jnp.allclose(jacobian(sys, [u, v]), expected)


# --- eigenvalues ---------------------------------------------------------


@pytest.mark.parametrize("alpha", [0.5, 1.0, 4.0])
def test_lotka_volterra_centre_eigenvalues_are_pure_imaginary(alpha):
    """At (1,1) the eigenvalues are exactly +/- i sqrt(alpha)."""
    lam = eigenvalues(models.lotka_volterra(alpha=alpha), [1.0, 1.0])
    assert jnp.allclose(lam.real, 0.0, atol=1e-12)
    assert jnp.allclose(jnp.sort(jnp.abs(lam.imag)), jnp.sqrt(alpha), atol=1e-12)


def test_eigenvalues_sorted_by_decreasing_real_part():
    lam = eigenvalues(models.lotka_volterra(alpha=2.0), [0.0, 0.0])
    assert lam[0].real > lam[1].real


# --- fixed point search --------------------------------------------------


def test_logistic_fixed_points():
    pts = fixed_points(models.logistic(r=1.0, K=2.0), [(-0.5, 4.0)])
    assert jnp.allclose(jnp.sort(pts[:, 0]), jnp.array([0.0, 2.0]), atol=1e-9)


def test_lotka_volterra_fixed_points():
    pts = fixed_points(models.lotka_volterra(alpha=1.0), [(0.0, 3.0), (0.0, 3.0)])
    assert pts.shape == (2, 2)
    assert jnp.allclose(pts[0], jnp.array([0.0, 0.0]), atol=1e-9)
    assert jnp.allclose(pts[1], jnp.array([1.0, 1.0]), atol=1e-9)


def test_all_found_points_are_actually_roots():
    sys = models.damped_lotka_volterra(alpha=1.0, mu1=0.1, mu2=0.1)
    for p in fixed_points(sys, [(0.0, 3.0), (0.0, 3.0)]):
        assert float(jnp.linalg.norm(sys.f(p))) < 1e-9


def test_search_respects_bounds():
    """(1/mu1, 0) = (10, 0) is a fixed point, but lies outside the box."""
    sys = models.damped_lotka_volterra(alpha=1.0, mu1=0.1, mu2=0.1)
    assert fixed_points(sys, [(0.0, 3.0), (0.0, 3.0)]).shape[0] == 2
    wide = fixed_points(sys, [(-0.5, 12.0), (-0.5, 3.0)])
    assert any(jnp.allclose(p, jnp.array([10.0, 0.0]), atol=1e-6) for p in wide)


def test_no_duplicates_returned():
    sys = models.lotka_volterra()
    pts = fixed_points(sys, [(0.0, 3.0), (0.0, 3.0)], n_seed=25)
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            assert float(jnp.linalg.norm(pts[i] - pts[j])) > 1e-6


def test_empty_result_has_correct_shape():
    """dN/dt = 1 has no fixed point anywhere."""
    sys = System(rhs=lambda y, p: jnp.array([1.0]), names=("N",))
    pts = fixed_points(sys, [(0.0, 1.0)])
    assert pts.shape == (0, 1)


def test_bad_bounds_rejected():
    sys = models.lotka_volterra()
    with pytest.raises(ValueError, match="3 bounds for 2 variables"):
        fixed_points(sys, [(0, 1), (0, 1), (0, 1)])
    with pytest.raises(ValueError, match="increasing"):
        fixed_points(sys, [(1, 0), (0, 1)])


# --- classification ------------------------------------------------------


def test_logistic_stability():
    sys = models.logistic(r=1.0, K=2.0)
    assert classify(sys, [0.0]).stability == "unstable"
    assert classify(sys, [2.0]).stability == "stable"


def test_lotka_volterra_origin_is_a_saddle():
    fp = classify(models.lotka_volterra(alpha=1.0), [0.0, 0.0])
    assert fp.stability == "saddle"
    assert fp.kind == "saddle"
    assert fp.label == "saddle"


def test_lotka_volterra_interior_is_a_centre():
    """Zero real part must report as marginal, not be rounded to stable."""
    fp = classify(models.lotka_volterra(alpha=1.0), [1.0, 1.0])
    assert fp.stability == "marginal"
    assert fp.kind == "center"
    assert fp.label == "center"


def test_damped_lotka_volterra_interior_is_a_stable_spiral():
    mu1, mu2, alpha = 0.1, 0.1, 1.0
    sys = models.damped_lotka_volterra(alpha=alpha, mu1=mu1, mu2=mu2)
    # u* = (1+mu2)/(1+mu1 mu2),  v* = (1-mu1)/(1+mu1 mu2)
    u_star = (1 + mu2) / (1 + mu1 * mu2)
    v_star = (1 - mu1) / (1 + mu1 * mu2)

    pts = fixed_points(sys, [(0.0, 3.0), (0.0, 3.0)])
    interior = pts[jnp.argmax(pts[:, 0])]
    assert jnp.allclose(interior, jnp.array([u_star, v_star]), atol=1e-9)

    fp = classify(sys, interior)
    assert fp.label == "stable spiral"
    assert jnp.all(fp.eigenvalues.real < 0)
    assert jnp.any(jnp.abs(fp.eigenvalues.imag) > 0)


def test_trace_determinant_shortcut():
    """Tr J < 0 and det J > 0 must agree with the eigenvalue verdict."""
    sys = models.damped_lotka_volterra(alpha=1.0, mu1=0.1, mu2=0.1)
    pts = fixed_points(sys, [(0.0, 3.0), (0.0, 3.0)])
    fp = classify(sys, pts[jnp.argmax(pts[:, 0])])
    assert fp.trace < 0 and fp.det > 0
    assert fp.stability == "stable"
    # Tr J = sum of eigenvalues, det J = product of eigenvalues
    assert fp.trace == pytest.approx(float(jnp.sum(fp.eigenvalues).real))
    assert fp.det == pytest.approx(float(jnp.prod(fp.eigenvalues).real))


def test_stable_node_has_no_rotation():
    """Two real negative eigenvalues: monotone approach, not a spiral."""
    sys = System(
        rhs=lambda y, p: jnp.array([-2.0 * y[0], -1.0 * y[1]]), names=("x", "y")
    )
    fp = classify(sys, [0.0, 0.0])
    assert fp.label == "stable node"


def test_str_is_readable():
    fp = classify(models.lotka_volterra(), [0.0, 0.0])
    text = str(fp)
    assert "u=0" in text and "saddle" in text and "lambda" in text


# --- method / function equivalence ---------------------------------------


def test_methods_delegate_to_module_functions():
    """Both spellings must stay in step; the methods are thin delegations."""
    sys = models.damped_lotka_volterra(alpha=1.0, mu1=0.1, mu2=0.1)
    bounds = [(0.0, 3.0), (0.0, 3.0)]
    y = [1.2, 0.8]

    assert jnp.allclose(sys.jacobian(y), jacobian(sys, y))
    assert jnp.allclose(sys.eigenvalues(y), eigenvalues(sys, y))
    assert jnp.allclose(sys.fixed_points(bounds), fixed_points(sys, bounds))
    assert sys.classify(y).label == classify(sys, y).label


def test_method_kwargs_are_forwarded():
    sys = models.lotka_volterra()
    bounds = [(0.0, 3.0), (0.0, 3.0)]
    assert jnp.allclose(
        sys.fixed_points(bounds, n_seed=25), fixed_points(sys, bounds, n_seed=25)
    )


# --- compilation caching -------------------------------------------------


def test_compiled_callables_are_reused():
    """Built once per system, not once per call."""
    sys = models.lotka_volterra()
    assert sys.jacobian_fn is sys.jacobian_fn
    assert sys.f_fn is sys.f_fn
    assert sys.jacobian_fn_batched is sys.jacobian_fn_batched


def test_with_params_shares_the_compilation_cache():
    """A parameter sweep must not recompile: params are a traced argument."""
    sys = models.lotka_volterra(alpha=1.0)
    other = sys.with_params(alpha=2.0)
    assert other.jacobian_fn is sys.jacobian_fn


def test_shared_cache_still_respects_new_parameters():
    """Sharing the cache must not leak the old parameter values."""
    sys = models.lotka_volterra(alpha=1.0)
    other = sys.with_params(alpha=3.0)
    # J = [[1-v, -u], [alpha v, alpha(u-1)]]; only the bottom row sees alpha
    assert float(jacobian(sys, [1.5, 0.7])[1, 0]) == pytest.approx(0.7)
    assert float(jacobian(other, [1.5, 0.7])[1, 0]) == pytest.approx(3 * 0.7)


def test_batched_jacobian_matches_pointwise():
    sys = models.damped_lotka_volterra(alpha=1.0, mu1=0.1, mu2=0.1)
    ys = jnp.array([[0.5, 0.5], [1.0, 1.0], [2.0, 1.5]])
    batched = jacobian(sys, ys)
    assert batched.shape == (3, 2, 2)
    for i, y in enumerate(ys):
        assert jnp.allclose(batched[i], jacobian(sys, y))


def test_batched_eigenvalues_match_pointwise():
    sys = models.damped_lotka_volterra(alpha=1.0, mu1=0.1, mu2=0.1)
    ys = jnp.array([[0.5, 0.5], [1.0, 1.0], [2.0, 1.5]])
    batched = eigenvalues(sys, ys)
    assert batched.shape == (3, 2)
    for i, y in enumerate(ys):
        assert jnp.allclose(batched[i], eigenvalues(sys, y))


def test_batched_eigenvalues_are_sorted_per_row():
    sys = models.lotka_volterra()
    lam = eigenvalues(sys, jnp.array([[0.0, 0.0], [2.0, 0.5]]))
    assert jnp.all(lam[:, 0].real >= lam[:, 1].real)


def test_batch_shape_mismatch_rejected():
    sys = models.lotka_volterra()
    with pytest.raises(ValueError, match=r"expected \(n, 2\)"):
        jacobian(sys, jnp.zeros((4, 3)))
