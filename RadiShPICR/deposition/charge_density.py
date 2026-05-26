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


    # n_A = 1 / (4*pi*r^2*dr) * 1/(W*A^3)
    # definiton of the number density given in the BIBLE!

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
    # interpolate the metric A to the particle positions to calculate the 
    # appropriate volume element for the charge deposition

    dV_metric = 4.0 * jnp.pi * grid.r_full[indices] ** 2 * grid.dr
    # compute the dV for normal spherical cooridinates

    dQ = weights * particles.get_charge() / dV_metric
    # calculate the non-relativistic charge density contribution first

    u_r, u_phi = particles.get_velocity()
    r = particles.r
    # get the particle velocities and position to calculate the relativistic correction factor W

    W = jnp.sqrt( 1.0                                    +  \
        ( u_r / A_at_particle[jnp.newaxis, :] )**2       +  \
        ( u_phi / r / A_at_particle[jnp.newaxis, :] )**2    )
    # define the lorentz factor W from the normalization of 4-velocity.

    density_contribution = dQ / W / A_at_particle[jnp.newaxis, :]**3
    # apply the relativistic correction to the charge density contribution

    charge_density = charge_density.at[indices].add(density_contribution)
    charge_density = charge_density.at[0].set(0.0)
    charge_density = charge_density.at[-1].set(0.0)

    return charge_density
