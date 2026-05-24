from functools import partial

import jax
import jax.numpy as jnp

from RadiShPICR.deposition.particle_shapes import (
    interpolate_field_to_particles,
    radial_shape_stencil,
)
from RadiShPICR.relativity.utils import safe_metric_A, safe_radius


@partial(jax.jit, static_argnames=("shape_mode",))
def mass_energy_deposition_terms(particles, A, grid, shape_mode="nearest"):
    """Particle metric and energy factors used by mass-energy deposition."""

    indices, weights = radial_shape_stencil(particles.r, grid, shape_mode=shape_mode)
    if shape_mode == "nearest":
        A_at_particle = safe_metric_A(A)[indices[0, :]]
    else:
        A_at_particle = interpolate_field_to_particles(
            safe_metric_A(A),
            particles.r,
            grid,
            shape_mode=shape_mode,
        )

    safe_r_particle = safe_radius(particles.r, grid.epsilon)
    kinetic_term = particles.u_r**2 + particles.u_phi**2 / safe_r_particle**2
    W = jnp.sqrt(1.0 + kinetic_term / A_at_particle**2)

    return indices, weights, A_at_particle, kinetic_term, W


@partial(jax.jit, static_argnames=("shape_mode",))
def compute_number_density(particles, A, grid, shape_mode="nearest"):
    """Deposited relativistic number density on the polar-slicing radial grid."""

    number_density = jnp.zeros_like(grid.r_full)
    indices, weights, A_at_particle, _, W = mass_energy_deposition_terms(
        particles,
        A,
        grid,
        shape_mode=shape_mode,
    )

    dV_rest = 4.0 * jnp.pi * grid.r_full[indices] ** 2 * grid.dr
    density_contribution = (
        weights
        * W[jnp.newaxis, :]
        / (dV_rest * A_at_particle[jnp.newaxis, :] ** 3)
    )

    number_density = number_density.at[indices].add(density_contribution)
    number_density = number_density.at[0].set(0.0)
    number_density = number_density.at[-1].set(0.0)

    return number_density


@partial(jax.jit, static_argnames=("shape_mode",))
def compute_number_density_metric_derivative(particles, A, grid, shape_mode="nearest"):
    """Diagonal derivative of number density with respect to metric ``A``."""

    jacobian = compute_number_density_metric_jacobian(
        particles,
        A,
        grid,
        shape_mode=shape_mode,
    )
    return jnp.diag(jacobian)


@partial(jax.jit, static_argnames=("shape_mode",))
def compute_number_density_metric_jacobian(particles, A, grid, shape_mode="nearest"):
    """Jacobian of number density with respect to grid ``A`` values."""

    dn_dA = jnp.zeros((grid.r_full.shape[0], grid.r_full.shape[0]), dtype=A.dtype)
    indices, weights, A_at_particle, kinetic_term, W = mass_energy_deposition_terms(
        particles,
        A,
        grid,
        shape_mode=shape_mode,
    )

    dV_rest = 4.0 * jnp.pi * grid.r_full[indices] ** 2 * grid.dr
    derivative_contribution = weights * (
        -kinetic_term[jnp.newaxis, :]
        / (dV_rest * A_at_particle[jnp.newaxis, :] ** 6 * W[jnp.newaxis, :])
        - 3.0
        * W[jnp.newaxis, :]
        / (dV_rest * A_at_particle[jnp.newaxis, :] ** 4)
    )

    row_indices = indices[:, jnp.newaxis, :]
    column_indices = indices[jnp.newaxis, :, :]
    jacobian_values = (
        derivative_contribution[:, jnp.newaxis, :] * weights[jnp.newaxis, :, :]
    )
    row_indices = jnp.broadcast_to(row_indices, jacobian_values.shape)
    column_indices = jnp.broadcast_to(column_indices, jacobian_values.shape)

    dn_dA = dn_dA.at[(row_indices.ravel(), column_indices.ravel())].add(
        jacobian_values.ravel()
    )
    dn_dA = dn_dA.at[0, :].set(0.0)
    dn_dA = dn_dA.at[-1, :].set(0.0)
    dn_dA = dn_dA.at[:, 0].set(0.0)
    dn_dA = dn_dA.at[:, -1].set(0.0)

    return dn_dA


@partial(jax.jit, static_argnames=("shape_mode",))
def compute_mass_density(particles, A, grid, shape_mode="nearest"):
    """Deposited mass-energy density on the polar-slicing radial grid."""

    mass_density = jnp.zeros_like(grid.r_full)
    indices, weights, A_at_particle, _, W = mass_energy_deposition_terms(
        particles,
        A,
        grid,
        shape_mode=shape_mode,
    )

    dV_rest = 4.0 * jnp.pi * grid.r_full[indices] ** 2 * grid.dr
    density_contribution = (
        weights
        * particles.get_mass()
        * W[jnp.newaxis, :]
        / (dV_rest * A_at_particle[jnp.newaxis, :] ** 3)
    )

    mass_density = mass_density.at[indices].add(density_contribution)
    mass_density = mass_density.at[0].set(0.0)
    mass_density = mass_density.at[-1].set(0.0)

    return mass_density


@partial(jax.jit, static_argnames=("shape_mode",))
def compute_mass_density_metric_derivative(particles, A, grid, shape_mode="nearest"):
    """Diagonal derivative of mass-energy density with respect to metric ``A``."""

    jacobian = compute_mass_density_metric_jacobian(
        particles,
        A,
        grid,
        shape_mode=shape_mode,
    )
    return jnp.diag(jacobian)


@partial(jax.jit, static_argnames=("shape_mode",))
def compute_mass_density_metric_jacobian(particles, A, grid, shape_mode="nearest"):
    """Jacobian of mass-energy density with respect to grid ``A`` values."""

    drho_dA = jnp.zeros((grid.r_full.shape[0], grid.r_full.shape[0]), dtype=A.dtype)
    indices, weights, A_at_particle, kinetic_term, W = mass_energy_deposition_terms(
        particles,
        A,
        grid,
        shape_mode=shape_mode,
    )

    dV_rest = 4.0 * jnp.pi * grid.r_full[indices] ** 2 * grid.dr
    derivative_contribution = weights * particles.get_mass() * (
        -kinetic_term[jnp.newaxis, :]
        / (dV_rest * A_at_particle[jnp.newaxis, :] ** 6 * W[jnp.newaxis, :])
        - 3.0
        * W[jnp.newaxis, :]
        / (dV_rest * A_at_particle[jnp.newaxis, :] ** 4)
    )

    row_indices = indices[:, jnp.newaxis, :]
    column_indices = indices[jnp.newaxis, :, :]
    jacobian_values = (
        derivative_contribution[:, jnp.newaxis, :] * weights[jnp.newaxis, :, :]
    )
    row_indices = jnp.broadcast_to(row_indices, jacobian_values.shape)
    column_indices = jnp.broadcast_to(column_indices, jacobian_values.shape)

    drho_dA = drho_dA.at[(row_indices.ravel(), column_indices.ravel())].add(
        jacobian_values.ravel()
    )
    drho_dA = drho_dA.at[0, :].set(0.0)
    drho_dA = drho_dA.at[-1, :].set(0.0)
    drho_dA = drho_dA.at[:, 0].set(0.0)
    drho_dA = drho_dA.at[:, -1].set(0.0)

    return drho_dA
