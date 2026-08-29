"""Fixed points and linear stability analysis.

A fixed point solves ``f(y*) = 0``, and its stability is decided by the
eigenvalues of the Jacobian ``J_ij = df_i/du_j`` evaluated there.

Jacobians come from ``jax.jacfwd``, so they are exact to machine precision
rather than finite-difference approximations. Fixed points are found by running
Newton's method from a grid of seeds, which makes enumeration a *search*: a
point outside ``bounds``, or one whose basin no seed lands in, will be missed.
Widen ``bounds`` or raise ``n_seed`` if you suspect a point is being skipped.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
from jax import Array

from popdynamics.system import System

__all__ = [
    "FixedPoint",
    "classify",
    "eigenvalues",
    "fixed_points",
    "jacobian",
]

# Real/imaginary parts below this count as zero when classifying. A genuine
# centre (undamped Lotka-Volterra) lands around 1e-16 with x64 enabled, so
# there is a wide margin between "numerically zero" and "physically small".
_CLASSIFY_TOL = 1e-9


def _check_state(system: System, y: Array) -> None:
    if y.shape != (system.ndim,):
        raise ValueError(
            f"state has shape {y.shape}, expected ({system.ndim},) "
            f"for variables {system.names}"
        )


def _check_batch(system: System, ys: Array) -> None:
    if ys.shape[-1] != system.ndim:
        raise ValueError(
            f"batch has shape {ys.shape}, expected (n, {system.ndim}) "
            f"for variables {system.names}"
        )


def jacobian(system: System, y: Array | Sequence[float]) -> Array:
    """Jacobian ``df_i/du_j`` of the right-hand side, evaluated at ``y``.

    Exact via forward-mode autodiff -- no finite-difference step to tune. The
    differentiated function is compiled once per system and cached, so
    evaluating at many points costs one compilation rather than one per point.

    Accepts a single state of shape ``(ndim,)`` returning ``(ndim, ndim)``, or a
    batch of shape ``(n, ndim)`` returning ``(n, ndim, ndim)``. Prefer the
    batched form in a loop -- it is vectorised, not just cached.
    """
    y = jnp.asarray(y, dtype=float)
    if y.ndim == 2:
        _check_batch(system, y)
        return system.jacobian_fn_batched(y, system.params)
    _check_state(system, y)
    return system.jacobian_fn(y, system.params)


def eigenvalues(system: System, y: Array | Sequence[float]) -> Array:
    """Eigenvalues of the Jacobian at ``y``, sorted by decreasing real part.

    Sorting makes the leading (least stable) eigenvalue come first, and makes
    the ordering reproducible across runs. Like :func:`jacobian`, this accepts a
    batch of states of shape ``(n, ndim)``.
    """
    J = jacobian(system, y)
    if J.ndim == 3:
        lam = jnp.linalg.eigvals(J)
        return jnp.take_along_axis(lam, jnp.argsort(-lam.real, axis=-1), axis=-1)
    lam = jnp.linalg.eigvals(J)
    return lam[jnp.argsort(-lam.real)]


def fixed_points(
    system: System,
    bounds: Sequence[tuple[float, float]],
    *,
    n_seed: int = 15,
    dedupe_tol: float = 1e-6,
    residual_tol: float = 1e-8,
    max_steps: int = 64,
) -> Array:
    """Find fixed points inside ``bounds`` by Newton from a grid of seeds.

    Parameters
    ----------
    bounds
        One ``(lo, hi)`` pair per state variable, defining both where seeds are
        placed and which roots are kept.
    n_seed
        Seeds per dimension, so ``n_seed ** ndim`` in total.
    dedupe_tol
        Roots closer together than this are treated as the same point.
    residual_tol
        A converged root is only accepted if ``|f(y*)|`` is below this.

    Returns
    -------
    Array
        Shape ``(n_found, ndim)``, sorted lexicographically for reproducibility.
        Empty with shape ``(0, ndim)`` when nothing is found.
    """
    bounds = [(float(lo), float(hi)) for lo, hi in bounds]
    if len(bounds) != system.ndim:
        raise ValueError(
            f"got {len(bounds)} bounds for {system.ndim} variables {system.names}"
        )
    for name, (lo, hi) in zip(system.names, bounds):
        if hi <= lo:
            raise ValueError(
                f"bounds for {name!r} must be increasing, got ({lo}, {hi})"
            )
    if n_seed < 2:
        raise ValueError(f"n_seed must be at least 2, got {n_seed}")

    # Seeds sit strictly inside each interval. Landing a seed exactly on a
    # boundary is a common way to start Newton at a singular Jacobian.
    axes = [jnp.linspace(lo, hi, n_seed + 2)[1:-1] for lo, hi in bounds]
    seeds = jnp.stack(jnp.meshgrid(*axes, indexing="ij"), axis=-1).reshape(
        -1, system.ndim
    )

    solver = optx.Newton(rtol=1e-12, atol=1e-14)

    def solve(y0: Array) -> tuple[Array, Array]:
        sol = optx.root_find(
            lambda y, args: system.f(y),
            solver,
            y0,
            throw=False,
            max_steps=max_steps,
        )
        return sol.value, sol.result == optx.RESULTS.successful

    roots, converged = jax.vmap(solve)(seeds)

    # Newton can converge to a point outside the region of interest, or stall
    # somewhere with a small-but-nonzero residual; both must be discarded.
    residuals = jnp.linalg.norm(system.f_fn_batched(roots, system.params), axis=-1)
    lo = jnp.array([b[0] for b in bounds])
    hi = jnp.array([b[1] for b in bounds])
    in_bounds = jnp.all(
        (roots >= lo - dedupe_tol) & (roots <= hi + dedupe_tol), axis=-1
    )
    keep = (
        converged
        & (residuals < residual_tol)
        & in_bounds
        & jnp.all(jnp.isfinite(roots), axis=-1)
    )

    candidates = np.asarray(roots[keep])
    if candidates.size == 0:
        return jnp.zeros((0, system.ndim))

    unique: list[np.ndarray] = []
    for point in candidates:
        if not any(
            np.allclose(point, seen, atol=dedupe_tol, rtol=0.0) for seen in unique
        ):
            unique.append(point)

    out = np.array(unique)
    return jnp.asarray(out[np.lexsort(out.T[::-1])])


@dataclass(frozen=True)
class FixedPoint:
    """A fixed point together with its linear stability analysis."""

    point: Array
    jacobian: Array
    eigenvalues: Array
    stability: str
    """``"stable"``, ``"unstable"``, ``"saddle"`` or ``"marginal"``."""
    kind: str
    """``"node"``, ``"spiral"``, ``"center"``, ``"saddle"`` or ``"degenerate"``."""
    system: System

    @property
    def trace(self) -> float:
        """``Tr J`` -- the sum of the eigenvalues."""
        return float(jnp.trace(self.jacobian).real)

    @property
    def det(self) -> float:
        """``det J`` -- the product of the eigenvalues."""
        return float(jnp.linalg.det(self.jacobian).real)

    @property
    def label(self) -> str:
        """Human-readable classification, e.g. ``"stable spiral"``."""
        if self.kind in ("saddle", "center"):
            return self.kind
        return f"{self.stability} {self.kind}"

    def __str__(self) -> str:
        coords = ", ".join(
            f"{n}={_snap(v):.4g}" for n, v in zip(self.system.names, self.point)
        )
        lam = ", ".join(_format_eigenvalue(z) for z in self.eigenvalues)
        return f"({coords})  {self.label}  [lambda = {lam}]"


def classify(system: System, y: Array | Sequence[float]) -> FixedPoint:
    """Classify the fixed point at ``y`` from its Jacobian's eigenvalues.

    ``y`` is taken on trust as a fixed point; nothing here checks ``f(y) = 0``,
    so the result is the linearisation about ``y`` whether or not it is one.
    """
    y = jnp.asarray(y, dtype=float)
    J = jacobian(system, y)
    lam = eigenvalues(system, y)
    re, im = lam.real, lam.imag

    n_pos = int(jnp.sum(re > _CLASSIFY_TOL))
    n_neg = int(jnp.sum(re < -_CLASSIFY_TOL))
    n_zero = lam.size - n_pos - n_neg

    if n_zero:
        # At least one eigenvalue has vanishing real part: the linearisation
        # cannot decide stability, so say so rather than guessing.
        stability = "marginal"
    elif n_pos == 0:
        stability = "stable"
    elif n_neg == 0:
        stability = "unstable"
    else:
        stability = "saddle"

    rotates = bool(jnp.any(jnp.abs(im) > _CLASSIFY_TOL))
    if stability == "saddle":
        kind = "saddle"
    elif rotates:
        kind = "center" if stability == "marginal" else "spiral"
    elif stability == "marginal":
        kind = "degenerate"
    else:
        kind = "node"

    return FixedPoint(
        point=y,
        jacobian=J,
        eigenvalues=lam,
        stability=stability,
        kind=kind,
        system=system,
    )


def _snap(x: float) -> float:
    """Display helper: Newton lands on values like -6e-33 that mean zero.

    Only the printed value is snapped -- the stored root is left untouched,
    since a genuinely tiny fixed point is a real possibility.
    """
    x = float(x)
    return 0.0 if abs(x) < 1e-12 else x


def _format_eigenvalue(z: complex) -> str:
    z = complex(z)
    if abs(z.imag) <= _CLASSIFY_TOL:
        return f"{z.real:+.4g}"
    return f"{z.real:+.4g}{z.imag:+.4g}i"
