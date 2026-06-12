import jax.numpy as jnp


def shape_weights_at_point(radial_positions, radial_coordinate, dr, shape_mode="nearest"):
    if shape_mode == "nearest":
        return jnp.where(jnp.abs(radial_positions - radial_coordinate) < 0.5 * dr, 1.0, 0.0)

    delta = (radial_positions - radial_coordinate) / dr
    center_particles = jnp.where(jnp.abs(delta) < 0.5, 0.75 - delta**2, 0.0)
    left_particles = jnp.where(
        (delta >= -1.5) & (delta < -0.5), 0.5 * (0.5 - delta) ** 2, 0.0
    )
    right_particles = jnp.where(
        (delta > 0.5) & (delta <= 1.5), 0.5 * (0.5 + delta) ** 2, 0.0
    )

    return center_particles + left_particles + right_particles


def charge_density_at_point(particles, A_at_point, radial_coordinate, dr, shape_mode=None):
    r_particle, _ = particles.get_positions()
    ur, uphi = particles.get_velocities()
    particle_shape = particles.get_shape() if shape_mode is None else shape_mode

    weights = shape_weights_at_point(r_particle, radial_coordinate, dr, particle_shape)
    safe_r = jnp.maximum(jnp.asarray(radial_coordinate, dtype=r_particle.dtype), 0.5 * dr)
    lorentz_factors = jnp.sqrt(
        1.0
        + ur**2 / A_at_point**2
        + uphi**2 / (A_at_point**2 * safe_r**2)
    )

    cell_volume = 4.0 * jnp.pi * A_at_point**3 * safe_r**2 * dr
    return jnp.sum(particles.get_charge() * weights * lorentz_factors / cell_volume)
