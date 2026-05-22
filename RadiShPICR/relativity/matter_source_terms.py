from functools import partial

import jax
import jax.numpy as jnp

from src.utils import safe_radius, safe_metric_A
from src.particle_shapes import interpolate_field_to_particles, radial_shape_stencil


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
def compute_Sr(
    masses, radial_velocities, radial_positions, A, grid, shape_mode="nearest"):
    """Momentum density S_r from particles on the polar radial grid."""

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
def compute_Srr(
    masses, radial_velocities, angular_velocities, radial_positions, A, grid,
    shape_mode="nearest"):
    """Radial stress S_rr from particles on the polar radial grid."""

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


@partial(jax.jit, static_argnames=("shape_mode",))
def _density_deposition_terms(particles, A, grid, shape_mode="nearest"):
    """Shared particle terms for deposited mass density."""

    indices, weights = radial_shape_stencil(particles.r, grid, shape_mode=shape_mode)
    if shape_mode == "nearest":
        A_at_particle = safe_metric_A(A)[indices[0, :]]
    else:
        A_at_particle = interpolate_to_particle(
            safe_metric_A(A),
            particles.r,
            grid,
            shape_mode=shape_mode,
        )
    # evaluate the metric at the particle with the same shape rule used by deposition
    safe_r_particle = safe_radius(particles.r, grid.epsilon)
    # ensure the radial positions of the particles are safe for division
    kinetic_term = particles.u_r**2 + particles.u_phi**2 / safe_r_particle**2
    W = jnp.sqrt(
        1.0
        + kinetic_term / A_at_particle**2
    )
    # define lorenz factor between fluid and normal observer

    return indices, weights, A_at_particle, kinetic_term, W


@partial(jax.jit, static_argnames=("shape_mode",))
def compute_mass_density(particles, A, grid, shape_mode="nearest"):
    """Deposited mass density from particles on the polar-slicing radial grid."""

    rho = jnp.zeros_like(grid.r_full)
    # initialize the density array to 0.

    indices, weights, A_at_particle, _, W = (
        _density_deposition_terms(particles, A, grid, shape_mode=shape_mode)
    )
    # use the same deposition and metric sampling rule as the metric derivative

    dV_rest = 4.0 * jnp.pi * grid.r_full[indices] ** 2 * grid.dr
    # spherical shell volume associated with each receiving grid point
    density_contribution = (
        weights * particles.mass[jnp.newaxis, :] * W[jnp.newaxis, :]
        / (dV_rest * A_at_particle[jnp.newaxis, :] ** 3)
    )
    # define the density contribution from each particle

    rho = rho.at[indices].add(density_contribution)
    # add the particle contributions to the density array using the deposition index

    # The two boundary cells remain vacuum cells in the elliptic solve.
    rho = rho.at[0].set(0.0)
    rho = rho.at[-1].set(0.0)

    return rho


@partial(jax.jit, static_argnames=("shape_mode",))
def compute_mass_density_metric_derivative(particles, A, grid, shape_mode="nearest"):
    """Metric derivative of the deposited mass density, ``d rho / d A``."""

    jacobian = compute_mass_density_metric_jacobian(
        particles,
        A,
        grid,
        shape_mode=shape_mode,
    )
    return jnp.diag(jacobian)


@partial(jax.jit, static_argnames=("shape_mode",))
def compute_mass_density_metric_jacobian(particles, A, grid, shape_mode="nearest"):
    """Jacobian of deposited mass density with respect to grid ``A`` values."""

    drho_dA = jnp.zeros((grid.r_full.shape[0], grid.r_full.shape[0]), dtype=A.dtype)
    # initialize the full metric derivative matrix.

    indices, weights, A_at_particle, kinetic_term, W = (
        _density_deposition_terms(particles, A, grid, shape_mode=shape_mode)
    )
    # evaluate the derivative using the same deposition rule as the density

    dV_rest = 4.0 * jnp.pi * grid.r_full[indices] ** 2 * grid.dr
    derivative_contribution = weights * particles.mass[jnp.newaxis, :] * (
        -kinetic_term[jnp.newaxis, :] / (dV_rest * A_at_particle[jnp.newaxis, :] ** 6 * W[jnp.newaxis, :])
        - 3.0 * W[jnp.newaxis, :] / (dV_rest * A_at_particle[jnp.newaxis, :] ** 4)
    )
    # define the derivative contribution from each particle

    row_indices = indices[:, jnp.newaxis, :]
    column_indices = indices[jnp.newaxis, :, :]
    jacobian_values = derivative_contribution[:, jnp.newaxis, :] * weights[jnp.newaxis, :, :]
    # Chain the derivative through A_at_particle = sum_j weight_j A_j.
    row_indices = jnp.broadcast_to(row_indices, jacobian_values.shape)
    column_indices = jnp.broadcast_to(column_indices, jacobian_values.shape)

    drho_dA = drho_dA.at[
        (row_indices.ravel(), column_indices.ravel())
    ].add(jacobian_values.ravel())
    # add all receiver/source metric derivative pairs.

    drho_dA = drho_dA.at[0, :].set(0.0)
    drho_dA = drho_dA.at[-1, :].set(0.0)
    drho_dA = drho_dA.at[:, 0].set(0.0)
    drho_dA = drho_dA.at[:, -1].set(0.0)
    # The two boundary cells remain vacuum cells in the elliptic solve.

    return drho_dA
