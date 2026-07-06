# RadiShPICR Simulation Context

Read this file first before writing a RadiShPICR validation test, smoke run, or
simulation script.  The goal is to avoid rediscovering the package structure and
to avoid copying stale run-directory scripts without checking the current source
API.

## Current Source-Backed Package Layout

The current importable source under `RadiShPICR/` is organized around these
pieces:

- `RadiShPICR.particles`: the `particle_species` container used by the current
  timestepper and tests.
- `RadiShPICR.ConstraintBasedRelativity.grid`: radial grid construction through
  `build_radial_grid(...)`.
- `RadiShPICR.ConstraintBasedRelativity.solve_metric`: construction of the
  grid-level metric and field tuple through `calculate_metric(...)`.
- `RadiShPICR.ConstraintBasedRelativity.geodesic` and
  `RadiShPICR.ConstraintBasedRelativity.lorentz_force`: particle acceleration
  terms consumed by the constraint-based timestepper.
- `RadiShPICR.ConstraintBasedRelativity.evolve`: constraint-based time
  integration routines, currently `step(...)` and `step_rk4(...)`.
- `RadiShPICR.Z4C`: Z4C metric containers, matter terms, and time-evolution
  helpers.  This formulation has its own RK4 routines in
  `RadiShPICR.Z4C.time_evolve`.
- `RadiShPICR.evolve`: compatibility imports for the constraint-based
  `step(...)` and `step_rk4(...)`.  Prefer formulation-local imports in new
  source.
- `RadiShPICR.diagnostics`: file writers for phase-space and metric-field
  snapshots.

Treat the package source and tests as the truth before using any script under
`runs/` as a template.

## Current Simulation Contract

A minimal source-compatible simulation is initialized in this order:

1. Build a radial grid with `build_radial_grid(epsilon, r_max, num_interior_points)`.
2. Create one `particle_species` object with JAX arrays for `r`, `ur`, `phi`,
   and `uphi`.
3. Compute the initial grid-level state with
   `calculate_metric(particles, grid.r_full, grid.dr)`.
4. Advance particles with `step(...)` for Euler-style stepping or `step_rk4(...)`
   for the current RK4 option.
5. Recompute `U_state = calculate_metric(...)` after particle motion whenever a
   diagnostic or validation check needs the current fields.

Current `particle_species` fields and methods are:

- Stored fields: `name`, `charges`, `masses`, `weight`, `r`, `ur`, `phi`,
  `uphi`, `shape_mode`.
- `get_positions()` returns `(r, phi)`.
- `get_velocities()` returns `(ur, uphi)`.
- `get_mass()` returns `masses * weight`.
- `get_charge()` returns `charges * weight`.
- `get_shape()` returns `shape_mode`.

Current `U_state` is a plain tuple, not a result class:

```python
(
    A,
    phi,
    alpha,
    Krr,
    beta_over_r,
    Er,
    source_terms,
    r_grid,
)
```

with

```python
source_terms = (mass_density, charge_density, Srr, Sr)
```

The timestepper recomputes the metric and source terms inside each stage.  Do
not hide or bypass that recomputation when writing validation tests for the
time integrator.

## Minimal Source-Compatible Skeleton

Use this shape for small validation scripts before adding run-specific
diagnostics:

```python
import jax.numpy as jnp

from RadiShPICR.ConstraintBasedRelativity import (
    build_radial_grid,
    calculate_metric,
    step_rk4,
)
from RadiShPICR.particles import particle_species


dt = 1.0e-3
num_steps = 10

grid = build_radial_grid(
    epsilon=1.0e-3,
    r_max=1.0,
    num_interior_points=64,
)

particles = particle_species(
    name="test",
    charge=-1.0,
    mass=1.0,
    weight=1.0,
    r=jnp.asarray([0.25, 0.75]),
    ur=jnp.asarray([0.05, -0.05]),
    phi=jnp.asarray([0.0, 0.0]),
    uphi=jnp.asarray([0.0, 0.0]),
    shape_mode="quadratic",
)

U_state = calculate_metric(particles, grid.r_full, grid.dr)

for step_index in range(num_steps):
    particles = step_rk4(particles, grid.r_full, grid.dr, dt)
    U_state = calculate_metric(particles, grid.r_full, grid.dr)
```

For multiple species, first check the current deposition and force routines.
Some current source paths and tests exercise a single `particle_species` object,
while older run scripts pass lists of species from custom initializers.

## Diagnostics

The current diagnostics use the same plain particle object and `U_state` tuple:

```python
from RadiShPICR.diagnostics import write_metric_fields, write_phase_space

write_phase_space(particles, output_folder, step=step_index, time=time)
write_metric_fields(U_state, output_folder, step=step_index, time=time)
```

`write_phase_space(...)` writes the current `particles.r` and `particles.ur`.
`write_metric_fields(...)` writes `r`, `A`, `phi`, `alpha`, `Krr`,
`beta_over_r`, `Er`, `mass_density`, `charge_density`, `Srr`, and `Sr`.

## Stale Run-Script Warning

Several scripts under `runs/` document the intended simulation lifecycle, but
they currently reference APIs that are not backed by `.py` source files in this
checkout.  Do not copy them into new validation work without first verifying
the imports.

In particular, verify any use of:

- `RadiShPICR.evolve.advance_one_step`
- `RadiShPICR.forces.*`
- `RadiShPICR.relativity.*`
- `RadiShPICR.EM.*`
- `MetricState`
- `compute_metric`
- `compute_A_solver_residual`
- particle attributes such as `u_r`, `u_phi`, `mass`, or `get_name()`

The current source-backed particle attributes are `ur`, `uphi`, `masses`, and
`name`.  The currently tracked `RadiShPICR/relativity/` and `RadiShPICR/EM/`
directories may contain bytecode caches, but no corresponding tracked `.py`
source files.  Treat those run scripts as historical or intended workflow
references until their imports are source-backed again.

## Validation Guidance

For source-level changes, start with:

```bash
cd code/RadiShPICR
PYTHONPATH=. python3 -m pytest tests/test_ustate_api.py tests/test_ustate_diagnostics.py -q -p no:cacheprovider
```

If this shell lacks `pytest` or `jax`, use a compile-only fallback and report
the environment limitation explicitly:

```bash
cd code/RadiShPICR
python3 -m py_compile \
    RadiShPICR/evolve.py \
    RadiShPICR/particles/particle_species.py \
    RadiShPICR/ConstraintBasedRelativity/evolve.py \
    RadiShPICR/ConstraintBasedRelativity/solve_metric.py \
    RadiShPICR/Z4C/time_evolve.py \
    RadiShPICR/diagnostics/metric_fields.py \
    RadiShPICR/diagnostics/phase_space.py
```

When writing a validation script, first verify the imports it uses with
`PYTHONPATH=code/RadiShPICR` from the repository root or `PYTHONPATH=.` from
`code/RadiShPICR`.
