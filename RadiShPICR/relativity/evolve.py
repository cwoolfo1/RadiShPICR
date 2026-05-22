from functools import partial

import jax
import jax.numpy as jnp

from src.curvature import compute_extrinsic_curvature
from src.states import FieldState
from src.schwarzschild import schwarzschild_A
from src.utils import safe_radius, safe_metric_A, compute_metric_radial_derivative
from src.A import solve_metric_A
from src.matter_source_terms import compute_mass_density, compute_Sr, compute_Srr
from src.particle_shapes import last_shape_support_index
from src.particles import compute_particle_derivatives
from src.shift_and_lapse import compute_lapse, compute_shift
from src.utils import (
    exact_exterior_points_from_last_matter_index,
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
    S_r = compute_Sr(
        particles.mass,
        particles.u_r,
        particles.r,
        A,
        grid,
        shape_mode=shape_mode,
    )
    S_rr = compute_Srr(
        particles.mass,
        particles.u_r,
        particles.u_phi,
        particles.r,
        A,
        grid,
        shape_mode=shape_mode,
    )
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

@partial(jax.jit, static_argnames=("shape_mode",))
def euler_step(
    particles, fields, grid, dt, schwarzschild_mass, shape_mode="nearest"):

    derivatives = compute_particle_derivatives(
        particles,
        fields,
        grid,
        schwarzschild_mass,
        shape_mode=shape_mode,
    )
    # compute the time derivatives of the particle quantities
    dt_value = jnp.asarray(dt, dtype=particles.r.dtype)
    updated_r = particles.r + dt_value * derivatives.dr_dt
    updated_phi = particles.phi + dt_value * derivatives.dphi_dt
    updated_u_r = particles.u_r + dt_value * derivatives.du_r_dt
    # time step the particle quantities with a simple Euler step using the computed derivatives
    return particles.with_updated_orbital_state(updated_r, updated_phi, updated_u_r)


@partial(jax.jit, static_argnames=("shape_mode",))
def rk4_step(particles, fields, grid, dt, schwarzschild_mass, shape_mode="nearest"):

    dt_value = jnp.asarray(dt, dtype=particles.r.dtype)

    k1 = compute_particle_derivatives(
        particles,
        fields,
        grid,
        schwarzschild_mass,
        shape_mode=shape_mode,
    )
    state_k2 = particles.with_updated_orbital_state(
        particles.r + 0.5 * dt_value * k1.dr_dt,
        particles.phi + 0.5 * dt_value * k1.dphi_dt,
        particles.u_r + 0.5 * dt_value * k1.du_r_dt,
    ) # compute the intermediate state for the second RK4 step using the k1 derivatives

    k2 = compute_particle_derivatives(
        state_k2,
        fields,
        grid,
        schwarzschild_mass,
        shape_mode=shape_mode,
    )
    state_k3 = particles.with_updated_orbital_state(
        particles.r + 0.5 * dt_value * k2.dr_dt,
        particles.phi + 0.5 * dt_value * k2.dphi_dt,
        particles.u_r + 0.5 * dt_value * k2.du_r_dt,
    ) # compute the intermediate state for the third RK4 step using the k2 derivatives

    k3 = compute_particle_derivatives(
        state_k3,
        fields,
        grid,
        schwarzschild_mass,
        shape_mode=shape_mode,
    )
    state_k4 = particles.with_updated_orbital_state(
        particles.r + dt_value * k3.dr_dt,
        particles.phi + dt_value * k3.dphi_dt,
        particles.u_r + dt_value * k3.du_r_dt,
    ) # compute the intermediate state for the fourth RK4 step using the k3 derivatives

    k4 = compute_particle_derivatives(
        state_k4,
        fields,
        grid,
        schwarzschild_mass,
        shape_mode=shape_mode,
    )

    updated_r = particles.r + (dt_value / 6.0) * (
        k1.dr_dt + 2.0 * k2.dr_dt + 2.0 * k3.dr_dt + k4.dr_dt
    )
    updated_phi = particles.phi + (dt_value / 6.0) * (
        k1.dphi_dt + 2.0 * k2.dphi_dt + 2.0 * k3.dphi_dt + k4.dphi_dt
    )
    updated_u_r = particles.u_r + (dt_value / 6.0) * (
        k1.du_r_dt + 2.0 * k2.du_r_dt + 2.0 * k3.du_r_dt + k4.du_r_dt
    ) # compute the final updated particle quantities using the RK4 formula with the computed k1, k2, k3, and k4 derivatives

    return particles.with_updated_orbital_state(updated_r, updated_phi, updated_u_r)


def advance_one_step(
    particles, grid, dt, schwarzschild_mass = None, initial_A_guess = None,
    previous_fields = None, fixed_fields = None, integrator = "rk4",
    remove_escaped_particles = True, shape_mode="nearest"):

    if schwarzschild_mass is None:
        raise ValueError( "advance_one_step requires an explicit schwarzschild_mass. ")

    current_mass = float(schwarzschild_mass)

    if fixed_fields is None:
        if previous_fields is None:
            prepared_initial_A_guess = initial_A_guess
            # pass through any raw A-array warm start exactly as supplied by the caller
        else:
            prepared_initial_A_guess = Euler_Step_A(
                previous_fields,
                grid,
                dt,
                current_mass,
            )
            # use the paper-style Euler predictor when the caller supplies the previous full field state
        fields = compute_fields(
            particles,
            grid,
            schwarzschild_mass,
            dt=dt,
            initial_A_guess=prepared_initial_A_guess,
            shape_mode=shape_mode,
        )
        # if no fixed fields are provided, update the fields
    else:
        fields = fixed_fields
        # use static fields if provided.

    if integrator == "euler":
        updated_particles = euler_step(
            particles,
            fields,
            grid,
            dt,
            current_mass,
            shape_mode=shape_mode,
        )
    else:
        updated_particles = rk4_step(
            particles,
            fields,
            grid,
            dt,
            current_mass,
            shape_mode=shape_mode,
        )

    # the standard polar update now keeps every particle instead of deleting trajectories at the grid edges
    return updated_particles, fields
