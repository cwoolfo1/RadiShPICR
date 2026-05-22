from functools import partial

import jax
import jax.numpy as jnp

from RadiShPICR.deposition.particle_shapes import radial_shape_stencil
from RadiShPICR.relativity.utils import safe_metric_A, safe_radius


def _effective_mass(particles):
    return particles.get_mass()


def _effective_charge(particles):
    return particles.get_charge()


def interpolate_to_particle(field, radial_positions, grid, shape_mode="nearest"):
    from RadiShPICR.deposition.particle_shapes import interpolate_field_to_particles

    return interpolate_field_to_particles(
        field,
        radial_positions,
        grid,
        shape_mode=shape_mode,
    )


@partial(jax.jit, static_argnames=("shape_mode",))
def _density_deposition_terms(particles, A, grid, shape_mode="nearest"):
    """Shared particle terms for deposited relativistic number density."""

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

    safe_r_particle = safe_radius(particles.r, grid.epsilon)
    kinetic_term = particles.u_r**2 + particles.u_phi**2 / safe_r_particle**2
    W = jnp.sqrt(1.0 + kinetic_term / A_at_particle**2)

    return indices, weights, A_at_particle, kinetic_term, W


@partial(jax.jit, static_argnames=("shape_mode",))
def compute_number_density(particles, A, grid, shape_mode="nearest"):
    """Deposited relativistic number density on the polar-slicing radial grid."""

    number_density = jnp.zeros_like(grid.r_full)
    indices, weights, A_at_particle, _, W = _density_deposition_terms(
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
    indices, weights, A_at_particle, kinetic_term, W = _density_deposition_terms(
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
    jacobian_values = derivative_contribution[:, jnp.newaxis, :] * weights[jnp.newaxis, :, :]
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
    return _effective_mass(particles) * compute_number_density(
        particles,
        A,
        grid,
        shape_mode=shape_mode,
    )


@partial(jax.jit, static_argnames=("shape_mode",))
def compute_mass_density_metric_derivative(particles, A, grid, shape_mode="nearest"):
    return _effective_mass(particles) * compute_number_density_metric_derivative(
        particles,
        A,
        grid,
        shape_mode=shape_mode,
    )


@partial(jax.jit, static_argnames=("shape_mode",))
def compute_mass_density_metric_jacobian(particles, A, grid, shape_mode="nearest"):
    return _effective_mass(particles) * compute_number_density_metric_jacobian(
        particles,
        A,
        grid,
        shape_mode=shape_mode,
    )


@partial(jax.jit, static_argnames=("shape_mode",))
def compute_charge_density(particles, A, grid, shape_mode="nearest"):
    return _effective_charge(particles) * compute_number_density(
        particles,
        A,
        grid,
        shape_mode=shape_mode,
    )


@partial(jax.jit, static_argnames=("shape_mode",))
def compute_charge_density_metric_derivative(particles, A, grid, shape_mode="nearest"):
    return _effective_charge(particles) * compute_number_density_metric_derivative(
        particles,
        A,
        grid,
        shape_mode=shape_mode,
    )


@partial(jax.jit, static_argnames=("shape_mode",))
def compute_charge_density_metric_jacobian(particles, A, grid, shape_mode="nearest"):
    return _effective_charge(particles) * compute_number_density_metric_jacobian(
        particles,
        A,
        grid,
        shape_mode=shape_mode,
    )
