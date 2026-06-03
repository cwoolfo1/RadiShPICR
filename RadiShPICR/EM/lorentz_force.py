from functools import partial

import jax
import jax.numpy as jnp

from RadiShPICR.deposition.particle_shapes import interpolate_field_to_particles


def interpolate_to_particle(field, radial_positions, grid, shape_mode="nearest"):
    return interpolate_field_to_particles(
        field,
        radial_positions,
        grid,
        shape_mode=shape_mode,
    )


@partial(jax.jit, static_argnames=("shape_mode",))
def compute_radial_lorentz_force_terms(
    particles,
    metric,
    grid,
    electric_field,
    shape_mode="nearest",
):
    """Return the radial electrostatic Lorentz-force term for particles."""

    electric_field_at_particle = interpolate_to_particle(
        electric_field,
        particles.r,
        grid,
        shape_mode=shape_mode,
    )
    lapse_at_particle = interpolate_to_particle(
        metric.lapse,
        particles.r,
        grid,
        shape_mode=shape_mode,
    )
    charge_to_mass = jnp.asarray(particles.charge, dtype=particles.r.dtype) / jnp.asarray(
        particles.mass,
        dtype=particles.r.dtype,
    )

    return lapse_at_particle * charge_to_mass * electric_field_at_particle
