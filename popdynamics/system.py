"""The central :class:`System` object and the :class:`Trajectory` it produces."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array


@dataclass(frozen=True)
class System:
    """An autonomous system of first-order ODEs ``dy/dt = f(y; params)``.

    Autonomy is assumed throughout the package: fixed points and nullclines
    are only well defined when the right-hand side has no explicit time
    dependence.

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
        # Compiled callables are built once and reused. The cache is not a
        # dataclass field, so it takes no part in equality and is rebuilt fresh
        # by ``dataclasses.replace`` -- which is what with_params relies on.
        object.__setattr__(self, "_cache", {})
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

    def _raw_f(self, y: Array, params: Mapping[str, Any]) -> Array:
        return jnp.asarray(self.rhs(y, params), dtype=float)

    def _compiled(self, key: str, build: Callable[[], Callable]) -> Callable:
        cache = self._cache
        if key not in cache:
            cache[key] = build()
        return cache[key]

    # Each of these is traced and compiled on first use, then reused for every
    # subsequent call. Parameters are passed in as an argument rather than
    # closed over, so a parameter sweep reuses one compilation.

    @property
    def f_fn(self) -> Callable[[Array, Mapping[str, Any]], Array]:
        """Compiled ``f(y, params)``, evaluating the right-hand side."""
        return self._compiled("f", lambda: jax.jit(self._raw_f))

    @property
    def f_fn_batched(self) -> Callable[[Array, Mapping[str, Any]], Array]:
        """Compiled ``f`` mapped over a leading batch axis: ``(n, ndim) -> (n, ndim)``."""
        return self._compiled(
            "f_batched",
            lambda: jax.jit(jax.vmap(self._raw_f, in_axes=(0, None))),
        )

    @property
    def jacobian_fn(self) -> Callable[[Array, Mapping[str, Any]], Array]:
        """Compiled Jacobian ``J(y, params)`` of shape ``(ndim, ndim)``."""
        return self._compiled(
            "jacobian",
            lambda: jax.jit(jax.jacfwd(self._raw_f, argnums=0)),
        )

    @property
    def jacobian_fn_batched(self) -> Callable[[Array, Mapping[str, Any]], Array]:
        """Compiled Jacobian over a batch: ``(n, ndim) -> (n, ndim, ndim)``."""
        return self._compiled(
            "jacobian_batched",
            lambda: jax.jit(
                jax.vmap(jax.jacfwd(self._raw_f, argnums=0), in_axes=(0, None))
            ),
        )

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
        dydt = self.f_fn(y, params)
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
        other = replace(self, params={**self.params, **overrides})
        # The compiled callables take params as a traced argument rather than
        # closing over them, so they stay valid here. Sharing the cache is what
        # makes a parameter sweep compile once instead of once per value.
        object.__setattr__(other, "_cache", self._cache)
        return other

    def integrate(self, y0: Any, t_span: tuple[float, float], **kwargs: Any):
        """Integrate from ``y0`` over ``t_span``. See :func:`popdynamics.integrate`."""
        from popdynamics.integrate import integrate

        return integrate(self, y0, t_span, **kwargs)

    # Analysis. These delegate to popdynamics.analysis, which holds the actual
    # implementations; the imports are deferred because that module imports
    # System in turn. Both spellings work -- ``sys.jacobian(y)`` reads better in
    # a notebook, ``jacobian(sys, y)`` composes better under jax transforms.

    def jacobian(self, y: Any) -> Array:
        """Jacobian of the right-hand side at ``y``. See :func:`popdynamics.jacobian`."""
        from popdynamics.analysis import jacobian

        return jacobian(self, y)

    def eigenvalues(self, y: Any) -> Array:
        """Eigenvalues of the Jacobian at ``y``. See :func:`popdynamics.eigenvalues`."""
        from popdynamics.analysis import eigenvalues

        return eigenvalues(self, y)

    def fixed_points(self, bounds: Any, **kwargs: Any) -> Array:
        """Search for fixed points in ``bounds``. See :func:`popdynamics.fixed_points`."""
        from popdynamics.analysis import fixed_points

        return fixed_points(self, bounds, **kwargs)

    def classify(self, y: Any):
        """Classify the fixed point at ``y``. See :func:`popdynamics.classify`."""
        from popdynamics.analysis import classify

        return classify(self, y)


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
