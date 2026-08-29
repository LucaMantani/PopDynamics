"""A library of standard population-dynamics models.

Each factory returns a ready-to-use :class:`~popdynamics.system.System` with
variable names set, so plots are labelled without extra configuration. The
runcard ``model:`` shorthand resolves names against this module.
"""

from __future__ import annotations

import jax.numpy as jnp

from popdynamics.system import System

__all__ = ["logistic", "lotka_volterra", "damped_lotka_volterra"]


def logistic(r: float = 1.0, K: float = 1.0) -> System:
    """Logistic growth ``dN/dt = r N (1 - N/K)``.

    Fixed points at ``N = 0`` (unstable, ``f'(0) = r``) and ``N = K``
    (stable, ``f'(K) = -r``).
    """

    def rhs(y, p):
        (N,) = y
        return jnp.array([p["r"] * N * (1.0 - N / p["K"])])

    return System(rhs=rhs, names=("N",), params={"r": r, "K": K})


def lotka_volterra(alpha: float = 1.0) -> System:
    """Undamped Lotka-Volterra ``du/dt = u(1-v)``, ``dv/dt = alpha v(u-1)``.

    The conservative predator-prey system: closed orbits around a centre at
    ``(1, 1)``, with the conserved quantity
    ``H = v - log v + alpha (u - log u)``.
    """

    def rhs(y, p):
        u, v = y
        return jnp.array([u * (1.0 - v), p["alpha"] * v * (u - 1.0)])

    return System(rhs=rhs, names=("u", "v"), params={"alpha": alpha})


def damped_lotka_volterra(
    alpha: float = 1.0, mu1: float = 0.1, mu2: float = 0.1
) -> System:
    """Logistically-limited predator-prey.

    ``du/dt = u(1 - v - mu1 u)``, ``dv/dt = alpha v(u - 1 - mu2 v)``.

    The self-limitation terms destroy the conserved quantity of
    :func:`lotka_volterra`, turning the centre at ``(1, 1)`` into a stable
    spiral at ``u* = (1+mu2)/(1+mu1 mu2)``, ``v* = (1-mu1)/(1+mu1 mu2)``.
    """

    def rhs(y, p):
        u, v = y
        return jnp.array(
            [
                u * (1.0 - v - p["mu1"] * u),
                p["alpha"] * v * (u - 1.0 - p["mu2"] * v),
            ]
        )

    return System(
        rhs=rhs, names=("u", "v"), params={"alpha": alpha, "mu1": mu1, "mu2": mu2}
    )
