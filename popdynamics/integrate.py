"""Trajectory integration, a thin wrapper over diffrax."""

from __future__ import annotations

from typing import Any

import diffrax
import jax.numpy as jnp

from popdynamics.system import System, Trajectory

__all__ = ["integrate"]


def integrate(
    system: System,
    y0: Any,
    t_span: tuple[float, float],
    *,
    n_points: int = 501,
    solver: diffrax.AbstractSolver | None = None,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    max_steps: int = 100_000,
) -> Trajectory:
    """Integrate ``system`` from initial condition ``y0`` over ``t_span``.

    Uses an adaptive Tsit5 step by default, sampling the dense solution at
    ``n_points`` evenly spaced times so plots are smooth regardless of the
    steps the solver actually took.
    """
    y0 = jnp.asarray(y0, dtype=float)
    if y0.shape != (system.ndim,):
        raise ValueError(
            f"initial condition has shape {y0.shape}, expected ({system.ndim},) "
            f"for variables {system.names}"
        )
    t0, t1 = float(t_span[0]), float(t_span[1])
    if t1 <= t0:
        raise ValueError(f"t_span must be increasing, got {t_span}")

    ts = jnp.linspace(t0, t1, n_points)
    term = diffrax.ODETerm(
        lambda t, y, args: jnp.asarray(system.rhs(y, args), dtype=float)
    )

    sol = diffrax.diffeqsolve(
        term,
        solver if solver is not None else diffrax.Tsit5(),
        t0=t0,
        t1=t1,
        dt0=None,
        y0=y0,
        args=system.params,
        stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
        saveat=diffrax.SaveAt(ts=ts),
        max_steps=max_steps,
    )
    return Trajectory(ts=sol.ts, ys=sol.ys, system=system)
