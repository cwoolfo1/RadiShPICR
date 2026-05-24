from functools import partial

import jax
import jax.numpy as jnp

from RadiShPICR.deposition.particle_shapes import (
    interpolate_field_to_particles,
    radial_shape_stencil,
)
from RadiShPICR.relativity.utils import safe_metric_A


@partial(jax.jit, static_argnames=("shape_mode",))
def compute_charge_density(particles, A, grid, shape_mode="nearest"):
    """Deposit charge density using the metric shell volume."""

    charge_density = jnp.zeros_like(grid.r_full)
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

    dV_metric = 4.0 * jnp.pi * grid.r_full[indices] ** 2 * grid.dr
    density_contribution = (
        weights
        * particles.get_charge()
        / (dV_metric * A_at_particle[jnp.newaxis, :] ** 3)
    )

    charge_density = charge_density.at[indices].add(density_contribution)
    charge_density = charge_density.at[0].set(0.0)
    charge_density = charge_density.at[-1].set(0.0)

    return charge_density
