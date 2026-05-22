from functools import partial

import jax
import jax.numpy as jnp

from RadiShPICR.deposition.particle_shapes import (
    interpolate_field_to_particles,
    radial_shape_stencil,
)
from RadiShPICR.relativity.utils import safe_radius


def interpolate_to_particle(field, radial_positions, grid, shape_mode="nearest"):
    """Interpolate one grid field from radial grid points to particles."""

    return interpolate_field_to_particles(field, radial_positions, grid, shape_mode=shape_mode)


@partial(jax.jit, static_argnames=("shape_mode",))
def _deposit_particle_values(radial_positions, particle_values, grid, shape_mode="nearest"):
    """Deposit one scalar particle contribution onto the radial grid."""

    source = jnp.zeros_like(grid.r_full)
    indices, weights = radial_shape_stencil(radial_positions, grid, shape_mode=shape_mode)

    # Each particle contributes to the cells in its shape-function support.
    source = source.at[indices].add(weights * particle_values[jnp.newaxis, :])

    # The first and last cells are boundary cells, so they do not carry matter.
    source = source.at[0].set(0.0)
    source = source.at[-1].set(0.0)
    return source


@partial(jax.jit, static_argnames=("shape_mode",))
def compute_Sr(particles, A, grid, shape_mode="nearest"):
    """Compute the radial momentum density ``S_r`` on the polar radial grid."""

    radial_positions = particles.r
    radial_momenta = particles.u_r
    particle_masses = particles.get_mass()

    A_at_particle = interpolate_to_particle(A, radial_positions, grid, shape_mode=shape_mode)
    safe_r_particle = safe_radius(radial_positions, grid.epsilon)

    # The spherical cell volume contributes 4 pi r^2 dr, and the polar metric
    # determinant contributes A^3 for the conformally flat spatial metric.
    particle_contribution = particle_masses * radial_momenta
    particle_contribution = particle_contribution / (
        4.0 * jnp.pi * A_at_particle**3 * safe_r_particle**2 * grid.dr
    )

    return _deposit_particle_values(
        radial_positions,
        particle_contribution,
        grid,
        shape_mode=shape_mode,
    )


@partial(jax.jit, static_argnames=("shape_mode",))
def compute_Srr(particles, A, grid, shape_mode="nearest"):
    """Compute the radial stress ``S_rr`` on the polar radial grid."""

    radial_positions = particles.r
    radial_momenta = particles.u_r
    azimuthal_momenta = particles.u_phi
    particle_masses = particles.get_mass()

    A_at_particle = interpolate_to_particle(A, radial_positions, grid, shape_mode=shape_mode)
    safe_r_particle = safe_radius(radial_positions, grid.epsilon)

    # W is the Lorentz factor between the particle four-momentum and the normal
    # observer in the spherical isotropic spatial metric.
    lorentz_factor = jnp.sqrt(
        1.0
        + radial_momenta**2 / A_at_particle**2
        + azimuthal_momenta**2 / (safe_r_particle**2 * A_at_particle**2)
    )

    # S_rr is the radial-radial projection of the stress tensor per spherical
    # grid volume, including the same metric volume factor as S_r.
    particle_contribution = particle_masses * radial_momenta**2
    particle_contribution = particle_contribution / (
        4.0
        * jnp.pi
        * A_at_particle**3
        * safe_r_particle**2
        * grid.dr
        * lorentz_factor
    )

    return _deposit_particle_values(
        radial_positions,
        particle_contribution,
        grid,
        shape_mode=shape_mode,
    )
