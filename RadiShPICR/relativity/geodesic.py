from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from RadiShPICR.deposition.particle_shapes import interpolate_field_to_particles
from RadiShPICR.relativity.utils import (
    centered_first_derivative,
    compute_metric_radial_derivative,
    safe_radius,
)


def compute_dphi_dt(
    lapse_at_particle,
    azimuthal_momentum,
    metric_A_at_particle,
    radial_position,
    lorentz_factor,
):
    return lapse_at_particle * azimuthal_momentum / (
        radial_position**2 * metric_A_at_particle**2 * lorentz_factor
    )


def interpolate_to_particle(field, radial_positions, grid, shape_mode="nearest"):
    return interpolate_field_to_particles(
        field,
        radial_positions,
        grid,
        shape_mode=shape_mode,
    )


@partial(jax.jit, static_argnames=("shape_mode",))
def compute_geodesic_terms(
    particles,
    metric,
    grid,
    schwarzschild_mass,
    shape_mode="nearest",
):
    A_at_particle = interpolate_to_particle(metric.A, particles.r, grid, shape_mode=shape_mode)
    lapse_at_particle = interpolate_to_particle(
        metric.lapse,
        particles.r,
        grid,
        shape_mode=shape_mode,
    )
    shift_at_particle = interpolate_to_particle(
        metric.shift,
        particles.r,
        grid,
        shape_mode=shape_mode,
    )

    dA_dr = compute_metric_radial_derivative(
        metric.A,
        schwarzschild_mass,
        grid,
        exact_exterior_points=None,
    )
    d_lapse_dr = centered_first_derivative(metric.lapse, grid.dr)
    d_shift_dr = centered_first_derivative(metric.shift, grid.dr)
    dA_dr_at_particle = interpolate_to_particle(
        dA_dr,
        particles.r,
        grid,
        shape_mode=shape_mode,
    )
    d_lapse_dr_at_particle = interpolate_to_particle(
        d_lapse_dr,
        particles.r,
        grid,
        shape_mode=shape_mode,
    )
    d_shift_dr_at_particle = interpolate_to_particle(
        d_shift_dr,
        particles.r,
        grid,
        shape_mode=shape_mode,
    )

    safe_r_particle = safe_radius(particles.r, grid.epsilon)
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
    dphi_dt = compute_dphi_dt(
        lapse_at_particle,
        particles.u_phi,
        A_at_particle,
        safe_r_particle,
        W,
    )

    # The radial geodesic equation contains lapse force, shift-gradient
    # advection, radial metric-gradient force, and centrifugal curvature force.
    du_r_dt = -W * d_lapse_dr_at_particle + particles.u_r * d_shift_dr_at_particle
    du_r_dt = du_r_dt + (
        lapse_at_particle * particles.u_r**2 * dA_dr_at_particle / (A_at_particle**3 * W)
    )
    du_r_dt = du_r_dt + (
        lapse_at_particle
        * particles.u_phi**2
        / W
        * (
            1.0 / (safe_r_particle**3 * A_at_particle**2)
            + dA_dr_at_particle / (safe_r_particle**2 * A_at_particle**3)
        )
    )

    return dr_dt, dphi_dt, du_r_dt
