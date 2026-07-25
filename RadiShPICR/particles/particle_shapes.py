from functools import partial

import jax
import jax.numpy as jnp

# from RadiShPICR.ConstraintBasedRelativity.utils import nearest_interior_index


def _quadratic_shape_weight(delta):
    """Evaluate the one-dimensional quadratic TSC shape function."""

    absolute_delta = jnp.abs(delta)
    center_weight = 0.75 - delta**2
    outer_weight = 0.5 * (1.5 - absolute_delta) ** 2

    return jnp.where(
        absolute_delta <= 0.5,
        center_weight,
        jnp.where(absolute_delta <= 1.5, outer_weight, 0.0),
    )


def nearest_interior_index(radial_positions, grid):
    """Map particles to the nearest interior grid point.

    The two edge cells are reserved as vacuum boundary cells, so matter is only
    deposited on indices `1` through `N-2`.
    """

    floating_index = (radial_positions - grid.r_full[0]) / grid.dr
    nearest = jnp.rint(floating_index).astype(jnp.int32)
    return jnp.clip(nearest, 1, grid.r_full.shape[0] - 2)


@partial(jax.jit, static_argnames=("shape_mode",))
def radial_shape_stencil(radial_positions, grid, shape_mode="nearest"):
    """Return radial deposition/interpolation indices and weights."""

    if shape_mode == "nearest":
        indices = nearest_interior_index(radial_positions, grid)[jnp.newaxis, :]
        weights = jnp.ones_like(indices, dtype=radial_positions.dtype)
        return indices, weights

    floating_index = (radial_positions - grid.r_full[0]) / grid.dr
    anchor = jnp.rint(floating_index).astype(jnp.int32)
    delta = floating_index - anchor.astype(radial_positions.dtype)

    offsets = jnp.asarray([-1, 0, 1], dtype=anchor.dtype)
    stencil_delta = (
        delta[jnp.newaxis, :]
        - offsets[:, jnp.newaxis].astype(radial_positions.dtype)
    )
    raw_weights = _quadratic_shape_weight(stencil_delta)
    raw_indices = anchor[jnp.newaxis, :] + offsets[:, jnp.newaxis]

    first_interior = jnp.asarray(1, dtype=raw_indices.dtype)
    last_interior = jnp.asarray(grid.r_full.shape[0] - 2, dtype=raw_indices.dtype)
    valid_source_cell = jnp.logical_and(
        raw_indices >= first_interior,
        raw_indices <= last_interior,
    )
    interior_weights = jnp.where(valid_source_cell, raw_weights, 0.0)
    weight_sum = jnp.sum(interior_weights, axis=0, keepdims=True)

    fallback_index = nearest_interior_index(radial_positions, grid)[jnp.newaxis, :]
    fallback_indices = jnp.broadcast_to(fallback_index, raw_indices.shape)
    clipped_indices = jnp.clip(raw_indices, first_interior, last_interior)
    has_valid_weight = weight_sum > 0.0

    fallback_weights = jnp.zeros_like(interior_weights)
    fallback_weights = fallback_weights.at[0, :].set(1.0)

    indices = jnp.where(has_valid_weight, clipped_indices, fallback_indices)
    weights = jnp.where(
        has_valid_weight,
        interior_weights / jnp.where(has_valid_weight, weight_sum, 1.0),
        fallback_weights,
    )

    return indices, weights


@partial(jax.jit, static_argnames=("shape_mode",))
def interpolate_field_to_particles(field, radial_positions, grid, shape_mode="nearest"):
    """Interpolate a radial grid field to particle positions."""

    if shape_mode == "nearest":
        return jnp.interp(radial_positions, grid.r_full, field)

    indices, weights = radial_shape_stencil(
        radial_positions,
        grid,
        shape_mode=shape_mode,
    )
    return jnp.sum(field[indices] * weights, axis=0)


def shape_weights_at_point(radial_positions, radial_coordinate, dr, shape_mode="nearest"):
    if shape_mode == "nearest":
        return jnp.where(jnp.abs(radial_positions - radial_coordinate) < 0.5 * dr, 1.0, 0.0)

    delta = (radial_positions - radial_coordinate) / dr
    return _quadratic_shape_weight(delta)
