from functools import partial

import jax
import jax.numpy as jnp

from RadiShPICR.relativity.particle_shapes import (
    interpolate_field_to_particles,
    radial_shape_stencil,
)
from RadiShPICR.relativity.utils import safe_radius


def interpolate_to_particle(field, radial_positions, grid, shape_mode="nearest"):
    """Interpolate a grid field from the physical points to the particles."""

    return interpolate_field_to_particles(field, radial_positions, grid, shape_mode=shape_mode)


@partial(jax.jit, static_argnames=("shape_mode",))
def _deposit_particle_values(radial_positions, particle_values, grid, shape_mode="nearest"):
    """Deposit one scalar value per particle to the radial grid."""

    source = jnp.zeros_like(grid.r_full)
    indices, weights = radial_shape_stencil(radial_positions, grid, shape_mode=shape_mode)
    source = source.at[indices].add(weights * particle_values[jnp.newaxis, :])
    source = source.at[0].set(0.0)
    source = source.at[-1].set(0.0)
    return source


@partial(jax.jit, static_argnames=("shape_mode",))
def compute_Sr(particles, A, grid, shape_mode="nearest"):
    """Momentum density S_r from particles on the polar radial grid."""

    masses = particles.get_mass()
    radial_velocities = particles.u_r
    radial_positions = particles.r
    A_at_particle = interpolate_to_particle(A, radial_positions, grid, shape_mode=shape_mode)
    # interpolate the A metric field to the particle positions, because the source term depends on A at the particle position
    safe_r_particle = safe_radius(radial_positions, grid.epsilon)
    # ensure the radial positions of the particles are safe for division
    contribution = masses * radial_velocities
    # the contribution to S_r from each particle is m * v^r, but we still need to divide by the volume of the cell and the appropriate factors of A and r to get the correct source term for the momentum constraint.
    
    contribution = contribution / (4.0 * jnp.pi * A_at_particle**3 * safe_r_particle**2 * grid.dr)
    # define the contribution to S_r from each particle, including the necessary factors of A and r for the momentum constraint source term, and dividing by the cell volume to get a density rather than a total contribution.

    return _deposit_particle_values(radial_positions, contribution, grid, shape_mode=shape_mode)


@partial(jax.jit, static_argnames=("shape_mode",))
def compute_Srr(particles, A, grid, shape_mode="nearest"):
    """Radial stress S_rr from particles on the polar radial grid."""

    masses = particles.get_mass()
    radial_velocities = particles.u_r
    angular_velocities = particles.u_phi
    radial_positions = particles.r
    A_at_particle = interpolate_to_particle(A, radial_positions, grid, shape_mode=shape_mode)
    # interpolate A to the particle positions
    safe_r_particle = safe_radius(radial_positions, grid.epsilon)
    # ensure the radius is not completely 0 for numerical stability
    W = jnp.sqrt(
        1.0
        + radial_velocities**2 / A_at_particle**2
        + angular_velocities**2 / (safe_r_particle**2 * A_at_particle**2)
    )
    # compute the lorenz factor between a fluid and normal observer for each particle

    contribution = masses * radial_velocities**2
    contribution = contribution / (
        4.0 * jnp.pi * A_at_particle**3 * safe_r_particle**2 * grid.dr * W
    )
    # add per particle contributions to Srr

    return _deposit_particle_values(radial_positions, contribution, grid, shape_mode=shape_mode)
