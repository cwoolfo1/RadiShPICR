from __future__ import annotations
from functools import partial
from typing import NamedTuple
import jax
import jax.numpy as jnp


from RadiShPICR.relativity.matter_source_terms import interpolate_to_particle
from RadiShPICR.relativity.utils import (
    centered_first_derivative,
    compute_metric_radial_derivative,
    safe_radius,
)


class ParticleDerivativeState(NamedTuple):
    """Time derivatives for the evolved orbit variables."""

    dr_dt: jnp.ndarray
    dphi_dt: jnp.ndarray
    du_r_dt: jnp.ndarray


def compute_dphi_dt( lapse_at_particle, azimuthal_momentum,
    metric_A_at_particle, radial_position, lorentz_factor):

    return lapse_at_particle * azimuthal_momentum / (
        radial_position**2 * metric_A_at_particle**2 * lorentz_factor
    )


@partial(jax.jit, static_argnames=("shape_mode",))
def compute_particle_derivatives(
    particles,
    fields,
    grid,
    schwarzschild_mass,
    shape_mode="nearest",
):

    A_at_particle = interpolate_to_particle(fields.A, particles.r, grid, shape_mode=shape_mode)
    lapse_at_particle = interpolate_to_particle(fields.lapse, particles.r, grid, shape_mode=shape_mode)
    shift_at_particle = interpolate_to_particle(fields.shift, particles.r, grid, shape_mode=shape_mode)
    # interpolate quantities to particle positions

    dA_dr = compute_metric_radial_derivative( fields.A, schwarzschild_mass, grid, exact_exterior_points = None )
    # compute the radial derivative of the metric function A, which is needed for the particle acceleration

    d_lapse_dr = centered_first_derivative(fields.lapse, grid.dr)
    d_shift_dr = centered_first_derivative(fields.shift, grid.dr)
    dA_dr_at_particle = interpolate_to_particle(dA_dr, particles.r, grid, shape_mode=shape_mode)
    d_lapse_dr_at_particle = interpolate_to_particle(d_lapse_dr, particles.r, grid, shape_mode=shape_mode)
    d_shift_dr_at_particle = interpolate_to_particle(d_shift_dr, particles.r, grid, shape_mode=shape_mode)
    # compute the radial derivatives of the metric functions and interpolate to particle positions

    safe_r_particle = safe_radius(particles.r, grid.epsilon)
    # ensure r values are not too close to zero to avoid numerical issues

    W = jnp.sqrt( 1.0 + particles.u_r**2 / A_at_particle**2
        + particles.u_phi**2 / (safe_r_particle**2 * A_at_particle**2) )
    # compute the lorzentz factor for the particles

    dr_dt = lapse_at_particle * particles.u_r / (A_at_particle**2 * W) - shift_at_particle
    # compute the radial velocity of the particles using the geodesic equations in the given metric

    dphi_dt = compute_dphi_dt( lapse_at_particle, particles.u_phi, A_at_particle, 
            safe_r_particle, W )
    # compute the azimuthal velocity of the particles using the geodesic equations in the given metric

    du_r_dt = -W * d_lapse_dr_at_particle + particles.u_r * d_shift_dr_at_particle
    du_r_dt = du_r_dt + ( lapse_at_particle * particles.u_r**2 * dA_dr_at_particle / (A_at_particle**3 * W))
    du_r_dt = du_r_dt + (lapse_at_particle * particles.u_phi**2 / W
        * (1.0 / (safe_r_particle**3 * A_at_particle**2)
            + dA_dr_at_particle / (safe_r_particle**2 * A_at_particle**3)) )
    # compute the radial acceleration of the particles using the geodesic equations in the given metric

    return ParticleDerivativeState(dr_dt=dr_dt, dphi_dt=dphi_dt, du_r_dt=du_r_dt)
