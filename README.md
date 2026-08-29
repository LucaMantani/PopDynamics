# PopDynamics

Define a system of first-order autonomous ODEs and get time evolution, phase
portraits with nullclines and vector fields, and fixed points classified by
linear stability. Jacobians come from JAX autodiff, so they are exact rather
than finite-difference approximations.

## Installation

```
conda create -n pop-dynamics
conda activate pop-dynamics
conda install python
pip install -e .
```

## Defining a system

A `System` is a right-hand side `f(y, params)` written in `jax.numpy`, plus the
variable names and parameter values.

```python
import jax.numpy as jnp
from popdynamics import System

def rhs(y, p):
    u, v = y
    return jnp.array([u * (1 - v), p["alpha"] * v * (u - 1)])

sys = System(rhs=rhs, names=("u", "v"), params={"alpha": 1.0})
```

`popdynamics.models` has ready-made ones: `logistic`, `r_plus_sN`,
`lotka_volterra`, `damped_lotka_volterra`, `competition`, `mutualism`.

```python
from popdynamics import models
sys = models.damped_lotka_volterra(alpha=1.0, mu1=0.1, mu2=0.1)
```

Systems are immutable; `sys.with_params(alpha=2.0)` returns a copy, which makes
parameter sweeps safe.

## Analysis

```python
sys.f([1.2, 0.8])                        # evaluate the right-hand side
sys.jacobian([1.2, 0.8])                 # exact, via jax.jacfwd
sys.eigenvalues([1.2, 0.8])              # sorted by decreasing real part

points = sys.fixed_points([(0, 3), (0, 3)])
for p in points:
    print(sys.classify(p))
    # (u=1.089, v=0.8911)  stable spiral  [lambda = -0.099+0.985i, -0.099-0.985i]
```

`classify` returns a `FixedPoint` carrying the eigenvalues, `trace`, `det`, a
`stability` verdict (`stable` / `unstable` / `saddle` / `marginal`) and a `kind`
(`node` / `spiral` / `center` / `saddle` / `degenerate`).

Fixed points are found by Newton's method from a grid of seeds, so enumeration
is a search, not a proof: a point outside `bounds`, or one whose basin no seed
lands in, will be missed. Widen `bounds` or raise `n_seed` if you suspect one is
being skipped.

Zero real parts are reported as `marginal` rather than rounded into
stable/unstable — the linearisation genuinely cannot decide there.

## Plotting

Every plotting function takes an optional `ax=` and returns it, so they compose.
`plot_phase_portrait` calls the others in order.

```python
import matplotlib.pyplot as plt
from popdynamics import plot_phase_portrait

plot_phase_portrait(
    sys, xlim=(0, 2.6), ylim=(0, 2.6),
    initial_conditions=[[2.4, 0.6], [0.4, 1.9]], t_span=(0, 60),
)
plt.show()
```

The layers are also available individually: `plot_vector_field`,
`plot_nullclines`, `plot_fixed_points`, `plot_trajectories`, and
`plot_timeseries` for time evolution.

## Runcards

The same thing from a YAML file, so a figure is reproducible from something you
can commit next to it:

```
pop-dyn runcards/damped_lv.yaml
```

```yaml
name: damped_lv

system:
  variables: [u, v]
  equations:
    u: u * (1 - v - mu1 * u)
    v: alpha * v * (u - 1 - mu2 * v)
  parameters: {alpha: 1.0, mu1: 0.1, mu2: 0.1}

domain:
  u: [0.0, 2.6]
  v: [0.0, 2.6]

analysis:
  fixed_points: true      # prints and saves a table of points and eigenvalues

plots:
  - type: phase_portrait  # or: timeseries
    output: damped_lv_phase.pdf
    trajectories:
      initial_conditions: [[2.4, 0.6], [0.4, 1.9]]
      t_span: [0, 60]
```

Instead of `variables`/`equations`, `system.model: damped_lotka_volterra` pulls
a system from `popdynamics.models` by name, with `parameters` overriding its
defaults.

Equations are arithmetic over the variables, the parameters, and a fixed set of
functions (`exp`, `log`, `sqrt`, `sin`, `tanh`, `abs`, …); anything else is
rejected by name when the runcard is read. They are evaluated on JAX tracers, so
autodiff works through them exactly as through hand-written Python.

Figures default to PDF. `-o DIR` sets the output directory; without it, outputs
land beside the runcard.

## Tests

```
pytest
```
