from __future__ import annotations

from functools import partial

from deposition import charge_density, mass_density
import jax
import jax.numpy as jnp

from RadiShPICR.deposition.particle_shapes import interpolate_field_to_particles
from RadiShPICR.relativity.utils import (
    safe_radius
)

from RadiShPICR.relativity.solve_metric import (
    dr_A,
    dr_beta_over_r,
    dr_Er,
    dr_alpha
    )


@partial(jax.jit, static_argnames=("shape_mode",))
def compute_geodesic_terms(particles, 
    metric_terms,
    r_grid,
    dr):

    A_values, phi_values, alpha_values, beta_over_r_values, Krr_values, Er_values, source_terms = metric_terms
    mass_density, charge_density, Srr, Sr = source_terms

    shape_mode = particles.get_shape()
    # 0 for nearest, 2 for quadratic

    r, phi = particles.get_positions()
    ur, uphi = particles.get_velocities()
    # get particle information

    A_at_particle = interpolate_field_to_particles(
        A_values,
        r,
        r_grid,
        shape_mode=shape_mode,
    )

    lapse_at_particle = interpolate_field_to_particles(
        alpha_values,
        r,
        r_grid,
        shape_mode=shape_mode,
    )

    beta = beta_over_r_values * r_grid
    # convert beta_over_r back to beta on the grid for interpolation

    shift_at_particle = interpolate_field_to_particles(
        beta,
        r,
        r_grid,
        shape_mode=shape_mode,
    )

    phi_at_particle = interpolate_field_to_particles(
        phi_values,
        r,
        r_grid,
        shape_mode=shape_mode,
    )

    # interpolate metric fields to particle positions


    U_state = (A_values, phi_values, alpha_values, Krr_values, beta_over_r_values, Er_values, source_terms, r_grid)
    dA_dr = dr_A(U_state)
    dalpha_dr = dr_alpha(U_state)
    d_shift_dr = dr_beta_over_r(U_state) * r_grid + beta_over_r_values
    # compute d_shift_dr on the grid using the product rule
    # use definitions of metric derivatives to compute them on the grid

    dA_dr_at_particle = interpolate_field_to_particles(
        dA_dr,
        r,
        r_grid,
        shape_mode=shape_mode,
    )

    d_lapse_dr_at_particle = interpolate_field_to_particles(
        dalpha_dr,
        r,
        r_grid,
        shape_mode=shape_mode,
    )

    d_shift_dr_at_particle = interpolate_field_to_particles(
        d_shift_dr,
        r,
        r_grid,
        shape_mode=shape_mode,
    )
    # interpolate metric derivatives to particle positions
        

    safe_r_particle = safe_radius(r, grid.epsilon)
    # W is the normal-observer Lorentz factor for 1D radial motion plus the
    # conserved azimuthal momentum in the spherical spatial metric.
    W = jnp.sqrt(
        1.0
        + particles.u_r**2 / A_at_particle**2
        + particles.u_phi**2 / (safe_r_particle**2 * A_at_particle**2)
    )

    # The coordinate radial velocity is the lapse-scaled physical radial
    # momentum minus the shift advection.
    dr_dt = lapse_at_particle * particles.u_r / (A_at_particle**2 * W) - shift_at_particle

    # The radial geodesic equation contains lapse force, shift-gradient
    # advection, radial metric-gradient force, and centrifugal curvature force.
    du_r_dt = -W * d_lapse_dr_at_particle + particles.u_r * d_shift_dr_at_particle

    du_r_dt = du_r_dt + (
        lapse_at_particle * particles.u_r**2 * dA_dr_at_particle / (A_at_particle**3 * W)
    )

    du_r_dt = du_r_dt + (
        lapse_at_particle * particles.u_phi**2
        / W * (
            1.0 / (safe_r_particle**3 * A_at_particle**2)
            + dA_dr_at_particle / (safe_r_particle**2 * A_at_particle**3)
        )
    )

    return dr_dt, du_r_dt
