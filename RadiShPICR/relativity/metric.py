from typing import NamedTuple

import jax.numpy as jnp

from RadiShPICR.EM.EM_energy_momentum_tensor import compute_EM_energy_density, compute_EM_stress
from RadiShPICR.EM.gauss_law import compute_charge_density_and_radial_electric_field
from RadiShPICR.deposition import compute_mass_density
from RadiShPICR.deposition.particle_shapes import last_shape_support_index
from RadiShPICR.relativity.A import solve_metric_A
from RadiShPICR.relativity.curvature import compute_extrinsic_curvature
from RadiShPICR.relativity.energy_momentum_tensor import compute_Sr, compute_Srr
from RadiShPICR.relativity.lapse import compute_lapse
from RadiShPICR.relativity.schwarzschild import schwarzschild_A
from RadiShPICR.relativity.shift import compute_shift
from RadiShPICR.relativity.solvers import solve_metric_A_broyden
from RadiShPICR.relativity.utils import exact_exterior_points_from_last_matter_index


class MetricState(NamedTuple):
    """Grid fields that define the spherical spacetime seen by particles."""

    rho: jnp.ndarray
    A: jnp.ndarray
    lapse: jnp.ndarray
    shift: jnp.ndarray
    extrinsic_curvature: jnp.ndarray
    S_r: jnp.ndarray
    S_rr: jnp.ndarray
    exact_exterior_points: jnp.ndarray


def _particle_list(particles):
    if isinstance(particles, (list, tuple)):
        return particles
    return [particles]


def compute_metric(
    particles,
    grid,
    schwarzschild_mass,
    initial_A_guess=None,
    shape_mode="nearest",
    metric_A_solver="newton",
    EM=True,
):
    """Solve the metric fields sourced by the current particle distribution."""

    species_list = _particle_list(particles)
    last_matter_index = last_shape_support_index(
        species_list[0].r,
        grid,
        shape_mode=shape_mode,
    )
    for species in species_list[1:]:
        species_last_matter_index = last_shape_support_index(
            species.r,
            grid,
            shape_mode=shape_mode,
        )
        last_matter_index = jnp.maximum(last_matter_index, species_last_matter_index)
    exact_exterior_points = exact_exterior_points_from_last_matter_index(
        last_matter_index,
        grid,
    )

    if initial_A_guess is None:
        prepared_initial_A_guess = schwarzschild_A(
            grid.r_full,
            schwarzschild_mass,
            grid.epsilon,
        )
        # The vacuum solution is the least-biased starting point for A.
    else:
        prepared_initial_A_guess = jnp.asarray(initial_A_guess, dtype=grid.r_full.dtype)
        if prepared_initial_A_guess.shape != grid.r_full.shape:
            raise ValueError(
                "initial_A_guess must have the same shape as grid.r_full: "
                f"expected {grid.r_full.shape}, got {prepared_initial_A_guess.shape}."
            )

    if metric_A_solver == "newton":
        A, converged, residual = solve_metric_A(
            particles,
            grid,
            schwarzschild_mass,
            initial_A_guess=prepared_initial_A_guess,
            shape_mode=shape_mode,
            EM=EM,
        )
        solver_name = "solve_metric_A"
    elif metric_A_solver == "broyden":
        A, converged, residual = solve_metric_A_broyden(
            particles,
            grid,
            schwarzschild_mass,
            initial_A_guess=prepared_initial_A_guess,
            shape_mode=shape_mode,
            EM=EM,
        )
        solver_name = "solve_metric_A_broyden"
    else:
        raise ValueError("metric_A_solver must be 'newton' or 'broyden'.")

    if not converged:
        raise RuntimeError(
            f"{solver_name} did not converge: ||R||_inf exceeded tolerance. "
            f"Last residual: {residual}"
        )

    particle_rho = jnp.zeros_like(grid.r_full)
    for species in species_list:
        particle_rho = particle_rho + compute_mass_density(
            species,
            A,
            grid,
            shape_mode=shape_mode,
        )

    rho = particle_rho
    if EM:
        _, electric_field = compute_charge_density_and_radial_electric_field(
            species_list,
            A,
            grid,
            shape_mode=shape_mode,
        )
        rho = rho + compute_EM_energy_density(electric_field)

    S_r = jnp.zeros_like(grid.r_full)
    S_rr = jnp.zeros_like(grid.r_full)
    for species in species_list:
        S_r = S_r + compute_Sr(species, A, grid, shape_mode=shape_mode)
        S_rr = S_rr + compute_Srr(species, A, grid, shape_mode=shape_mode)
    if EM:
        S_rr = S_rr + compute_EM_stress(electric_field, A)

    # Birkhoff's theorem fixes the discrete exterior to the vacuum solution.
    rho = jnp.where(exact_exterior_points, 0.0, rho)
    S_r = jnp.where(exact_exterior_points, 0.0, S_r)
    S_rr = jnp.where(exact_exterior_points, 0.0, S_rr)

    lapse = compute_lapse(
        A,
        S_rr,
        schwarzschild_mass,
        grid,
        exact_exterior_points=exact_exterior_points,
    )
    extrinsic_curvature = compute_extrinsic_curvature(
        A,
        S_r,
        schwarzschild_mass,
        grid,
        exact_exterior_points=exact_exterior_points,
    )
    shift = compute_shift(
        lapse,
        extrinsic_curvature,
        grid,
        exact_exterior_points=exact_exterior_points,
    )

    return MetricState(
        rho=rho,
        A=A,
        lapse=lapse,
        shift=shift,
        extrinsic_curvature=extrinsic_curvature,
        S_r=S_r,
        S_rr=S_rr,
        exact_exterior_points=exact_exterior_points,
    )
