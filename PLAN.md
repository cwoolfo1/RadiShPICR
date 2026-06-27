# Z4C Formulation Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` or `superpowers:executing-plans` to
> implement this plan task-by-task. Keep this checklist synchronized with the
> source after each completed task.

**Goal:** Complete the spherical Z4C formulation so it can evolve particle
matter and metric fields alongside the existing constrained formulation, then
allow simulation entry points to select either formulation.

**Architecture:** Keep `ConstraintBasedRelativity` and `Z4C` as separate
formulation modules with formulation-specific metric, matter, and geodesic
implementations. The top-level timestepper will dispatch between them while
preserving the current constrained behavior as the default. Shared particle
storage, radial shape interpolation, center absorption, and electrostatic
forcing remain outside the formulation-specific metric code.

**Tech Stack:** Python, JAX, `jax.numpy`, PyTest, existing RadiShPICR particle
and radial shape APIs.

---

## Current Status

The Z4C implementation is approximately 60 percent complete.

Completed:

- [x] Define the `Z4C_Metric` state containing lapse, shift, conformal metric,
  conformal factor, extrinsic curvature, constraint variables, damping
  parameters, and radial grid.
- [x] Implement the spatial-metric, extrinsic-curvature, constraint,
  lapse, and shift right-hand-side functions.
- [x] Define the five-component `MatterTerms` container.
- [x] Provide zero-valued vacuum matter terms.
- [x] Assemble the Z4C right-hand sides into
  `metric_time_derivatives(metric, matter_terms)`.
- [x] Implement classic RK4 metric stepping through
  `RadiShPICR.Z4C.time_evolve.rk4_step(metric, matter_terms, dt)`.
- [x] Test metric layout preservation, fixed grid/damping fields, vacuum flat
  space, and RK4 stage weights.

Remaining:

- [ ] Establish regular-center and radial-boundary behavior for Z4C finite
  differences and every explicit `1/r` and `1/r**2` term.
- [ ] Compute all Z4C matter source terms from particles.
- [ ] Compute particle geodesic derivatives from a `Z4C_Metric`.
- [ ] Add Z4C particle RK4 and coupled particle/metric stepping.
- [ ] Add a public constrained/Z4C formulation selector.
- [ ] Add Z4C diagnostics, import surfaces, and end-to-end validation.

## Numerical Contracts

These contracts must remain explicit throughout the migration:

- Particle fields remain `r`, `phi`, `ur`, and `uphi`; `ur` and `uphi` are
  covariant spatial momenta.
- The physical inverse spatial metric represented by `Z4C_Metric` is
  `gamma_rr_inv = chi / conformal_grr` and
  `gamma_phiphi_inv = chi / (r**2 * conformal_gt)`.
- Particle Lorentz factor is
  `W = sqrt(1 + gamma_rr_inv*ur**2 + gamma_phiphi_inv*uphi**2)`.
- Matter terms are deposited with the particle's configured radial shape.
- The first coupled Z4C implementation holds matter terms constant during the
  four metric RK4 substages.
- `kappa`, `eta`, `r`, and `dr` are fixed Z4C parameters/grid data, not evolved
  variables.
- The existing constrained APIs remain the default until the Z4C path passes
  focused and smoke validation.

## Phase 1: Stabilize Z4C Metric Primitives

### Task 1: Add a supported Z4C package surface

**Files:**

- Create: `RadiShPICR/Z4C/__init__.py`
- Modify: `tests/test_import_order.py`

- [ ] Add failing import tests for `Z4C_Metric`, `MatterTerms`,
  `initialize_flat_metric`, `compute_matter_terms`, and `rk4_step`.
- [ ] Export only source-backed Z4C APIs from `RadiShPICR.Z4C`.
- [ ] Add `initialize_flat_metric(r, dr, kappa=0.0, eta=0.0)` in
  `z4c_metric.py`. It must initialize `alpha`, `conformal_grr`,
  `conformal_gt`, and `chi` to one and all evolved curvature/constraint/shift
  fields to zero.
- [ ] Verify the new imports do not introduce a cycle with
  `ConstraintBasedRelativity` or `RadiShPICR.evolve`.

Validation:

```bash
PYTHONPATH=. python3 -m pytest tests/test_import_order.py tests/test_z4c_time_evolve.py -q -p no:cacheprovider
```

### Task 2: Define regular-center and boundary derivatives

**Files:**

- Modify: `RadiShPICR/Z4C/derivatives.py`
- Modify: `RadiShPICR/Z4C/spatial_metric.py`
- Modify: `RadiShPICR/Z4C/extrinsic_curvature.py`
- Modify: `RadiShPICR/Z4C/constraint_terms.py`
- Test: `tests/test_z4c_derivatives.py`

- [ ] Replace the current periodic `jnp.roll` boundary behavior with explicit
  radial finite-difference stencils. Use fourth-order centered differences where
  the full stencil is available, parity-reflected ghost values at the center,
  and fourth-order backward one-sided differences at the outer two points.
  Boundary closures must not wrap the outer boundary into the center.
- [ ] Use even center parity for `alpha`, `conformal_grr`, `conformal_gt`,
  `chi`, `Kh`, `Arr`, `At`, `theta`, `kappa`, and `eta`. Use odd center parity
  for `beta`, `Zr`, and `Gamma`.
- [ ] Use even center parity for `rho`, `Srr`, `Stt`, and `St`; use odd center
  parity for radial momentum density `Sr`.
- [ ] Replace raw center divisions with analytic center limits or parity-based
  regularized expressions. Do not merely pad `r` if that changes the continuum
  center equation.
- [ ] Add tests proving constant fields have zero derivatives, polynomial
  derivatives converge at the expected interior order, no periodic wrap occurs,
  and flat vacuum RHS values are finite on a grid containing `r=0`.

Validation:

```bash
PYTHONPATH=. python3 -m pytest tests/test_z4c_derivatives.py tests/test_z4c_time_evolve.py -q -p no:cacheprovider
```

## Phase 2: Particle-Sourced Energy-Momentum Tensor

### Task 3: Implement local Z4C particle moments

**Files:**

- Modify: `RadiShPICR/Z4C/energy_momentum_tensor.py`
- Test: `tests/test_z4c_matter_terms.py`

- [ ] Add a helper that computes `gamma_rr_inv`, `gamma_phiphi_inv`, `W`, and
  the proper shell volume at one grid point:

```python
gamma_rr_inv = chi / conformal_grr
gamma_phiphi_inv = chi / (safe_r**2 * conformal_gt)
W = jnp.sqrt(
    1.0
    + gamma_rr_inv * particles.ur**2
    + gamma_phiphi_inv * particles.uphi**2
)
cell_volume = (
    4.0
    * jnp.pi
    * jnp.sqrt(conformal_grr)
    * conformal_gt
    * safe_r**2
    * dr
    / chi**1.5
)
```

- [ ] Add `matter_terms_at_point(particles, metric, radial_coordinate)`.
- [ ] Deposit the covariant particle moments using existing shape weights:

```python
rho = sum(mass * weight * W / cell_volume)
Sr = sum(mass * weight * ur / cell_volume)
St = sum(mass * weight * uphi / cell_volume)
Srr = sum(mass * weight * ur**2 / (cell_volume * W))
Stt = sum(mass * weight * uphi**2 / (cell_volume * W))
```

  In these expressions, `weight` denotes the radial shape weight; particle
  macro-weighting remains included through `particles.get_mass()`.
- [ ] Preserve finite center-cell volume using the same half-cell radius
  convention used by current constrained deposition until Task 2 supplies an
  analytic center-volume rule.
- [ ] Test vacuum, rest-particle, radial-momentum, and angular-momentum cases
  independently.

### Task 4: Assemble grid-level Z4C matter terms

**Files:**

- Modify: `RadiShPICR/Z4C/energy_momentum_tensor.py`
- Test: `tests/test_z4c_matter_terms.py`

- [ ] Add `compute_matter_terms(particles, metric) -> MatterTerms`.
- [ ] Evaluate all five moments on `metric.r` using JAX-compatible array code;
  prefer `jax.vmap` over Python loops.
- [ ] Require each returned component to have the same shape and dtype as
  `metric.r`.
- [ ] Add eager-versus-`jax.jit` parity tests.
- [ ] Add a shape-mode regression showing quadratic particles use the existing
  quadratic deposition weights.

Validation:

```bash
PYTHONPATH=. python3 -m pytest tests/test_z4c_matter_terms.py -q -p no:cacheprovider
```

## Phase 3: Z4C Particle Forces

### Task 5: Implement Z4C metric interpolation and geodesic terms

**Files:**

- Create: `RadiShPICR/Z4C/geodesic.py`
- Test: `tests/test_z4c_geodesic.py`

- [ ] Add `compute_geodesic_terms(particles, metric)` returning
  `(dr_dt, dphi_dt, dur_dt)`.
- [ ] Compute grid-level `gamma_rr_inv` and `gamma_phiphi_inv`, their radial
  derivatives, `dalpha_dr`, and `dbeta_dr`.
- [ ] Interpolate all required fields and derivatives with
  `interpolate_field_to_particles(...)` and each species' `shape_mode`.
- [ ] Implement:

```python
dr_dt = alpha * gamma_rr_inv * ur / W - beta
dphi_dt = alpha * gamma_phiphi_inv * uphi / W
dur_dt = -W * dalpha_dr + ur * dbeta_dr
dur_dt -= alpha * (
    ur**2 * d_gamma_rr_inv_dr
    + uphi**2 * d_gamma_phiphi_inv_dr
) / (2.0 * W)
```

- [ ] Keep the electromagnetic force outside this function. Z4C geodesics
  describe gravity and coordinate motion only.
- [ ] Test flat radial motion, flat angular motion, lapse gradients, shift
  advection, and shape-aware interpolation.
- [ ] Add eager-versus-`jax.jit` parity coverage.

Validation:

```bash
PYTHONPATH=. python3 -m pytest tests/test_z4c_geodesic.py -q -p no:cacheprovider
```

### Task 6: Define electromagnetic coupling for Z4C

**Files:**

- Create: `RadiShPICR/Z4C/lorentz_force.py`
- Test: `tests/test_z4c_geodesic.py`

- [ ] Keep the current constrained Lorentz force unchanged.
- [ ] Define the Z4C stepper input so the electric field is passed separately
  from `Z4C_Metric`; do not add `Er` to the metric container.
- [ ] Add a small force function with the explicit contract
  `compute_lorentz_terms(particles, metric, Er) -> dur_dt`.
- [ ] Interpolate `alpha` and `Er` to particles with the configured shape and
  preserve the current force normalization
  `alpha*q*Er/m`.
- [ ] Test zero charge, zero field, constant field, and shape-aware
  interpolation.

## Phase 4: Coupled Z4C Time Integration

### Task 7: Add Z4C particle RK4

**Files:**

- Create: `RadiShPICR/Z4C/evolve.py`
- Test: `tests/test_z4c_evolve.py`

- [ ] Add `step_particles_rk4(particles, metric, Er, dt)`.
- [ ] Reuse the current center-absorption semantics: stage particles at
  `r <= 0` are clamped before metric interpolation and have all derivatives
  zeroed.
- [ ] Use `compute_geodesic_terms(...)` for `dr_dt`, `dphi_dt`, and the
  gravitational `dur_dt`; add the separate Lorentz contribution to `dur_dt`.
- [ ] Preserve `uphi` until an angular force equation is explicitly added.
- [ ] Test classic RK4 weighting, flat free motion, constant electric force,
  and center-crossing behavior.

### Task 8: Add one coupled Z4C simulation step

**Files:**

- Modify: `RadiShPICR/Z4C/evolve.py`
- Test: `tests/test_z4c_evolve.py`

- [ ] Add:

```python
def step_z4c(particles, metric, Er, dt):
    matter_terms = compute_matter_terms(particles, metric)
    updated_metric = rk4_step(metric, matter_terms, dt)
    updated_particles = step_particles_rk4(particles, metric, Er, dt)
    return updated_particles, updated_metric
```

- [ ] Document and test the first-version splitting contract: matter and metric
  fields are fixed during their respective RK4 substeps.
- [ ] Do not silently introduce stage-wise particle/matter/metric coupling in
  this task. That is a later accuracy upgrade requiring a separate convergence
  study.
- [ ] Test that matter terms are computed once and the same matter tuple reaches
  all four metric stages.

Validation:

```bash
PYTHONPATH=. python3 -m pytest tests/test_z4c_evolve.py tests/test_z4c_time_evolve.py -q -p no:cacheprovider
```

## Phase 5: User-Selectable Relativity

### Task 9: Preserve explicit constrained and Z4C entry points

**Files:**

- Modify: `RadiShPICR/evolve.py`
- Modify: `RadiShPICR/ConstraintBasedRelativity/__init__.py`
- Modify: `RadiShPICR/Z4C/__init__.py`
- Test: `tests/test_relativity_selection.py`

- [ ] Rename the internal current implementations to
  `step_constrained(...)` and `step_rk4_constrained(...)` while retaining
  `step(...)` and `step_rk4(...)` wrappers for compatibility.
- [ ] Export `step_z4c(...)` from `RadiShPICR.Z4C`.
- [ ] Keep constrained results bitwise/roundoff equivalent to the pre-dispatch
  implementation.

### Task 10: Add the formulation selector

**Files:**

- Modify: `RadiShPICR/evolve.py`
- Test: `tests/test_relativity_selection.py`
- Modify: `AGENTS.md`

- [ ] Keep the existing `step(...)` and `step_rk4(...)` signatures as
  constrained-only compatibility APIs.
- [ ] Add a new public dispatcher with a consistent return contract:

```python
def advance_one_step(particles, relativity_state, dt, relativity="constrained"):
    ...
    return particles, relativity_state
```

- [ ] Accept exactly `"constrained"` and `"z4c"` in the first version.
- [ ] For constrained relativity, define `relativity_state` as a plain tuple
  `(r_grid, dr)`; call `step_rk4_constrained(...)`, then return the unchanged
  tuple with the updated particles.
- [ ] For Z4C relativity, define `relativity_state` as `(metric, Er)`; call
  `step_z4c(...)`, then return `(updated_metric, Er)` with the updated
  particles.
- [ ] Preserve `"constrained"` as the dispatcher default, while existing
  callers of `step(...)` and `step_rk4(...)` remain completely unchanged.
- [ ] Do not synthesize a Z4C metric from constrained `U_state` inside the
  dispatcher.
- [ ] Add dispatch tests for both formulations and one focused invalid-name
  check.
- [ ] Update `AGENTS.md` with the initialization and call order for both
  formulations.

Validation:

```bash
PYTHONPATH=. python3 -m pytest tests/test_relativity_selection.py tests/test_ustate_api.py -q -p no:cacheprovider
```

## Phase 6: Diagnostics and End-to-End Validation

### Task 11: Add Z4C metric diagnostics

**Files:**

- Create: `RadiShPICR/diagnostics/z4c_metric_fields.py`
- Modify: `RadiShPICR/diagnostics/__init__.py`
- Test: `tests/test_z4c_diagnostics.py`

- [ ] Add `write_z4c_metric_fields(metric, matter_terms, output_folder, step,
  time=None)`.
- [ ] Save all evolved metric fields, fixed damping parameters, radial grid,
  and all five matter terms with stable names matching `Z4C_Metric` and
  `MatterTerms`.
- [ ] Keep existing constrained `write_metric_fields(...)` output unchanged.
- [ ] Test saved keys, array shapes, step, and time.

### Task 12: Add physics and convergence smoke tests

**Files:**

- Create: `tests/test_z4c_smoke.py`
- Create: `../../runs/Z4C_Smoke/run_z4c_smoke.py`

- [ ] Run flat vacuum for multiple steps and require finite, stationary fields.
- [ ] Run vacuum gauge perturbations and monitor finite constraint variables.
- [ ] Run one low-density neutral-particle case and require finite matter,
  metric, and particle arrays.
- [ ] Compare Z4C geodesic motion against constrained motion for a shared static
  metric represented in both formulations.
- [ ] Perform a timestep refinement check for the isolated metric RK4 and
  particle RK4 paths.
- [ ] Record that full coupled fourth-order convergence is not expected while
  matter and metric are frozen during opposite substages.

Validation:

```bash
PYTHONPATH=. python3 -m pytest tests/test_z4c_*.py tests/test_relativity_selection.py -q -p no:cacheprovider
PYTHONPATH=. python3 -m pytest tests -q -p no:cacheprovider
```

## Completion Criteria

The Z4C path is ready for user selection only when:

- [ ] Particle matter deposition produces all five finite Z4C source arrays.
- [ ] Z4C geodesic and Lorentz terms are shape-aware and JIT-compatible.
- [ ] Center and outer-boundary behavior no longer use periodic wraparound.
- [ ] Coupled Z4C particle/metric stepping runs for more than one step without
  non-finite state.
- [ ] Existing constrained tests pass without changed numerical behavior.
- [ ] The public selector defaults to constrained relativity and routes Z4C
  explicitly.
- [ ] Diagnostics distinguish constrained `U_state` output from Z4C metric and
  matter output.
- [ ] `PLAN.md` is updated after every completed task so checked items match
  source and tests.
