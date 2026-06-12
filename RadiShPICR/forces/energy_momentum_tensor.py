import jax.numpy as jnp

from RadiShPICR.deposition.particle_shapes import shape_weights_at_point


def Sr_at_point(particles, A_at_point, radial_coordinate, dr, shape_mode=None):
    r_particle, _ = particles.get_positions()
    ur, _ = particles.get_velocities()
    particle_shape = particles.get_shape() if shape_mode is None else shape_mode

    weights = shape_weights_at_point(r_particle, radial_coordinate, dr, particle_shape)
    safe_r = jnp.maximum(jnp.asarray(radial_coordinate, dtype=r_particle.dtype), 0.5 * dr)
    cell_volume = 4.0 * jnp.pi * A_at_point**3 * safe_r**2 * dr

    return jnp.sum(particles.get_mass() * weights * ur / cell_volume)


def Srr_at_point(particles, A_at_point, radial_coordinate, dr, shape_mode=None):
    r_particle, _ = particles.get_positions()
    ur, uphi = particles.get_velocities()
    particle_shape = particles.get_shape() if shape_mode is None else shape_mode

    weights = shape_weights_at_point(r_particle, radial_coordinate, dr, particle_shape)
    safe_r = jnp.maximum(jnp.asarray(radial_coordinate, dtype=r_particle.dtype), 0.5 * dr)
    lorentz_factor = jnp.sqrt(
        1.0
        + ur**2 / A_at_point**2
        + uphi**2 / (safe_r**2 * A_at_point**2)
    )
    cell_volume = 4.0 * jnp.pi * A_at_point**3 * safe_r**2 * dr

    return jnp.sum(particles.get_mass() * weights * ur**2 / (cell_volume * lorentz_factor))
