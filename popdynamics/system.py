"""The central :class:`System` object and the :class:`Trajectory` it produces."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import jax.numpy as jnp
from jax import Array


@dataclass(frozen=True)
class System:
    """An autonomous system of first-order ODEs ``dy/dt = f(y; params)``.

    Autonomy is assumed throughout the package: fixed points, nullclines,
    conserved quantities and Lyapunov functions are only well defined when the
    right-hand side has no explicit time dependence.

    Parameters
    ----------
    rhs
        Callable ``f(y, params) -> array`` of shape ``(ndim,)``. It must be
        written in ``jax.numpy`` so it can be differentiated and vectorised.
    names
        Name of each state variable, e.g. ``("u", "v")``. Used for axis labels
        and to index into a :class:`Trajectory`.
    params
        Parameter values passed through to ``rhs`` as its second argument.
    """

    rhs: Callable[[Array, Mapping[str, Any]], Array]
    names: tuple[str, ...]
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "names", tuple(self.names))
        object.__setattr__(self, "params", dict(self.params))
        if not self.names:
            raise ValueError("System needs at least one variable name")
        if len(set(self.names)) != len(self.names):
            raise ValueError(f"duplicate variable names in {self.names!r}")

    @property
    def ndim(self) -> int:
        """Number of state variables."""
        return len(self.names)

    def f(self, y: Any, **overrides: Any) -> Array:
        """Evaluate the right-hand side at state ``y``.

        Keyword arguments temporarily override parameter values, so
        ``sys.f(y, alpha=2.0)`` needs no copy of the system.
        """
        y = jnp.asarray(y, dtype=float)
        if y.shape != (self.ndim,):
            raise ValueError(
                f"state has shape {y.shape}, expected ({self.ndim},) "
                f"for variables {self.names}"
            )
        params = {**self.params, **overrides} if overrides else self.params
        dydt = jnp.asarray(self.rhs(y, params), dtype=float)
        if dydt.shape != (self.ndim,):
            raise ValueError(
                f"rhs returned shape {dydt.shape}, expected ({self.ndim},)"
            )
        return dydt

    def with_params(self, **overrides: Any) -> System:
        """Return a copy of this system with some parameters changed.

        The system itself is immutable, which keeps parameter sweeps free of
        aliasing bugs::

            [sys.with_params(alpha=a) for a in jnp.linspace(0.5, 2.0, 10)]
        """
        unknown = set(overrides) - set(self.params)
        if unknown:
            raise KeyError(
                f"unknown parameter(s) {sorted(unknown)}; "
                f"this system has {sorted(self.params)}"
            )
        return replace(self, params={**self.params, **overrides})

    def integrate(self, y0: Any, t_span: tuple[float, float], **kwargs: Any):
        """Integrate from ``y0`` over ``t_span``. See :func:`popdynamics.integrate`."""
        from popdynamics.integrate import integrate

        return integrate(self, y0, t_span, **kwargs)


@dataclass(frozen=True)
class Trajectory:
    """A solved trajectory: times ``ts``, states ``ys`` of shape ``(len(ts), ndim)``."""

    ts: Array
    ys: Array
    system: System

    @property
    def names(self) -> tuple[str, ...]:
        return self.system.names

    @property
    def y_final(self) -> Array:
        return self.ys[-1]

    def __getitem__(self, name: str) -> Array:
        """Time series of one variable by name, e.g. ``traj["u"]``."""
        try:
            i = self.names.index(name)
        except ValueError:
            raise KeyError(
                f"no variable {name!r}; this system has {self.names}"
            ) from None
        return self.ys[:, i]
