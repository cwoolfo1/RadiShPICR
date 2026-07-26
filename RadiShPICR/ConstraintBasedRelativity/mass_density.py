import jax.numpy as jnp

from RadiShPICR.particles.particle_shapes import shape_weights_at_point
from RadiShPICR.ConstraintBasedRelativity.utils import pad_value, radial_shell_volume


def mass_density_at_point(particles, A_at_point, radial_coordinate, dr, shape_mode=None):
    r_particle, _ = particles.get_positions()
    ur, uphi = particles.get_velocities()
    particle_shape = particles.get_shape() if shape_mode is None else shape_mode

    weights = shape_weights_at_point(r_particle, radial_coordinate, dr, particle_shape)
    safe_r = jnp.maximum(jnp.asarray(radial_coordinate, dtype=r_particle.dtype), 0.5 * dr)
    A_for_denominators = pad_value(A_at_point)
    lorentz_factors = jnp.sqrt(
        1.0
        + ur**2 / A_for_denominators**2
        + uphi**2 / (A_for_denominators**2 * safe_r**2)
    )

    cell_volume = radial_shell_volume(
        A_for_denominators,
        radial_coordinate,
        dr,
    )
    return jnp.sum(particles.get_mass() * weights * lorentz_factors / cell_volume)
