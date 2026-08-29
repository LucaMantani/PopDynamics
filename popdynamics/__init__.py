"""PopDynamics: a JAX-based framework for studying population dynamics.

Define a system of first-order autonomous ODEs once, then get time evolution,
phase portraits with nullclines and vector fields, fixed points classified by
linear stability, and Lyapunov-function checking.
"""

# This must run before any other jax import so every array defaults to float64.
# Both Newton convergence and telling a genuinely-zero eigenvalue real part from
# a small negative one need more precision than jax's float32 default.
import jax as _jax

_jax.config.update("jax_enable_x64", True)

from popdynamics import models
from popdynamics.integrate import integrate
from popdynamics.plotting import plot_timeseries
from popdynamics.system import System, Trajectory

__all__ = [
    "System",
    "Trajectory",
    "integrate",
    "plot_timeseries",
    "models",
]
