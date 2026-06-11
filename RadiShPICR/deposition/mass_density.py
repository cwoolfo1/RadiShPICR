from functools import partial

import jax
import jax.numpy as jnp

def mass_density_at_point(particles, A_at_point, point, dr, shape_mode="nearest"):
    # particles is the particle data structure containing particle positions and velocities
    # A_at_point is the metric factor A evaluated at the specified point on the grid
    # point is the radial grid index at which to compute the mass density
    # dr is the grid spacing in the radial direction
    # shape_mode specifies the particle shape function to use for deposition

    index_position = point * dr
    # compute the physical radial position corresponding to the specified grid index

    if shape_mode == "nearest":
        r_particle = particles.r
        weights = jnp.where(jnp.abs(r_particle - index_position) < 0.5 * dr, 1.0, 0.0)
        # if the shape mode is "nearest", compute the weights as 1 for particles within half a grid spacing of the point, and 0 otherwise
    else:
        r_particle = particles.r

        # left side of the stencil : 0.5 * (0.5 - delta) ** 2
        # center of the stencil : 0.75 - delta**2
        # right side of the stencil : 0.5 * (0.5 + delta) ** 2

        deltas = (r_particle - index_position) / dr
        # compute the normalized distance of each particle from the point in units of the grid spacing

        center_particles = jnp.where(jnp.abs(deltas) < 0.5, 0.75 - deltas**2, 0.0)
        # weights for particles within the center of the stencil (within half a grid spacing of the point)
        left_particles = jnp.where(
            (deltas >= -1.5) & (deltas < -0.5), 0.5 * (0.5 - deltas) ** 2, 0.0
        )
        # weights for particles on the left side of the stencil (between 0.5 and 1.5 grid spacings to the left of the point)
        right_particles = jnp.where(
            (deltas > 0.5) & (deltas <= 1.5), 0.5 * (0.5 + deltas) ** 2, 0.0
        )
        # weights for particles on the right side of the stencil (between 0.5 and 1.5 grid spacings to the right of the point)

        weights = center_particles + left_particles + right_particles
        # total weights for each particle based on their position relative to the point and the chosen shape

    lorenz_factors = jnp.sqrt(1.0 + (particles.u_r**2 / A_at_point**2) + (particles.u_phi**2 / (A_at_point**2 * index_position**2)))
    # compute the Lorentz factor for each particle at the point using their velocities and the metric factor A 

    mass = particles.mass
    # get the mass of each particle from the particle data structure
    mass_density_at_point = jnp.sum(mass * weights * lorenz_factors /  ( 4 * jnp.pi * A_at_point**3 * index_position**2 * dr )  )
    # compute the mass density at the point by summing the contributions from all particles, accounting for their mass, shape weights, Lorentz factors, and the volume element in spherical coordinates   


    return mass_density_at_point