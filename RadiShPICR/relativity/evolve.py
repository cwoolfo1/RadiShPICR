import jax
import jax.numpy as jnp

from RadiShPICR.deposition import compute_mass_density
from RadiShPICR.deposition.particle_shapes import last_shape_support_index
from RadiShPICR.relativity.A import solve_metric_A
from RadiShPICR.relativity.curvature import compute_extrinsic_curvature
from RadiShPICR.relativity.matter_source_terms import compute_Sr, compute_Srr
from RadiShPICR.relativity.schwarzschild import schwarzschild_A
from RadiShPICR.relativity.shift_and_lapse import compute_lapse, compute_shift
from RadiShPICR.relativity.states import FieldState
from RadiShPICR.relativity.utils import (
    compute_metric_radial_derivative,
    exact_exterior_points_from_last_matter_index,
    safe_metric_A,
    safe_radius,
)



def compute_fields(
    particles,
    grid,
    schwarzschild_mass,
    dt=None,
    initial_A_guess=None,
    shape_mode="nearest",
):

    max_index_cell = last_shape_support_index(particles.r, grid, shape_mode=shape_mode)
    # compute the outermost grid index touched by the particle deposition support
    exact_exterior_points = exact_exterior_points_from_last_matter_index(max_index_cell, grid)
    # compute the exact Schwarzschild exterior points from the outermost matter support

    
    if initial_A_guess is None:
        prepared_initial_A_guess = schwarzschild_A(grid.r_full, schwarzschild_mass, grid.epsilon)
        # if no initial guess is provided for A, use the exact Schwarzschild solution as the initial guess
    else:
        prepared_initial_A_guess = jnp.asarray(initial_A_guess, dtype=grid.r_full.dtype)
        # treat the supplied warm start as a raw A-array guess instead of re-running the Euler predictor on it
        if prepared_initial_A_guess.shape != grid.r_full.shape:
            raise ValueError(
                "initial_A_guess must have the same shape as grid.r_full: "
                f"expected {grid.r_full.shape}, got {prepared_initial_A_guess.shape}."
            )
        # require the warm-start array to live on the same grid as the Hamiltonian solve

    A, converged, residual = solve_metric_A(
        particles,
        grid,
        schwarzschild_mass,
        initial_A_guess=prepared_initial_A_guess,
        shape_mode=shape_mode,
    )
    # iterate on the initial guess for A using the Newton solver to enforce the Hamiltonian constraint
    
    if not converged:
        raise RuntimeError(
            f"solve_metric_A did not converge: ||R||_inf exceeded tolerance. Last residual: {residual}"
        )
    # if the solver did not converge, raise an error with the last residual for debugging


    rho = compute_mass_density(particles, A, grid, shape_mode=shape_mode)
    S_r = compute_Sr(particles, A, grid, shape_mode=shape_mode)
    S_rr = compute_Srr(particles, A, grid, shape_mode=shape_mode)
    # compute the matter source terms from the particle properties and the metric field A

    rho = jnp.where(exact_exterior_points, 0.0, rho)
    S_r = jnp.where(exact_exterior_points, 0.0, S_r)
    S_rr = jnp.where(exact_exterior_points, 0.0, S_rr)
    # apply the condition that the source terms are zero in the vacuum region outside the matter support


    lapse = compute_lapse( A, S_rr, schwarzschild_mass, grid, exact_exterior_points=exact_exterior_points)
    # compute the lapse function from A and S_rr

    extrinsic_curvature = compute_extrinsic_curvature(A, S_r, schwarzschild_mass, grid, exact_exterior_points=exact_exterior_points)
    # compute the extrinsic curvature from A and S_r

    shift = compute_shift(lapse, extrinsic_curvature, grid, exact_exterior_points=exact_exterior_points )
    # compute the shift from the lapse and extrinsic curvature

    return FieldState( rho=rho, A=A, lapse=lapse, shift=shift, extrinsic_curvature=extrinsic_curvature,
        S_r=S_r, S_rr=S_rr )

@jax.jit
def Euler_Step_A(fields, grid, dt, schwarzschild_mass):

    safe_r = safe_radius(grid.r_full, grid.epsilon)
    # ensure the radius is not completely 0 for numerical stability

    dA_dr = compute_metric_radial_derivative(fields.A, schwarzschild_mass, grid)
    # compute the radial derivative of A with the regular center and Schwarzschild outer boundary conditions

    dt_value = jnp.asarray(dt, dtype=fields.A.dtype)
    metric_rhs = fields.shift * (dA_dr + fields.A / safe_r)
    # compute the right-hand side of the A evolution equation, which is the shift times (dA/dr + A/r) in spherical symmetry

    A_new = fields.A + dt_value * metric_rhs
    # compute the new A values from the Euler step

    A_new = safe_metric_A(A_new)
    # ensure the new A values are safe for numerical stability

    outer_A = schwarzschild_A( grid.r_full[-1], schwarzschild_mass, grid.epsilon )
    # compute the exact Schwarzschild A for the outer boundary condition, which is the correct vacuum solution

    A_new = A_new.at[-1].set( outer_A )
    # set the outer boundary to the exact Schwarzschild value, which is the correct vacuum solution
    A_new = A_new.at[0].set( fields.A[0] )
    # keep the regular center fixed during the predictor step so the Neumann center condition stays explicit

    return A_new
