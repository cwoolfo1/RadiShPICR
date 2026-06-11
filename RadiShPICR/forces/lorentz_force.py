from functools import partial

import jax
import jax.numpy as jnp

from RadiShPICR.deposition.particle_shapes import interpolate_field_to_particles

def compute_lorentz_terms(particles, 
    metric_terms,
    r_grid,
    dr):
    A_values, phi_values, alpha_values, beta_over_r_values, Krr_values, Er_values, source_terms = metric_terms
    mass_density, charge_density, Srr, Sr = source_terms

    r, phi = particles.get_positions()
    mass = particles.get_mass()
    charge = particles.get_charge()
    shape_mode = particles.get_shape()

    lapse_at_particle = interpolate_field_to_particles(
        alpha_values,
        r,
        r_grid,
        shape_mode=shape_mode,
    )

    electric_field_at_particle = interpolate_field_to_particles(
        Er_values,
        r,
        r_grid,
        shape_mode=shape_mode,
    )

    lorentz_acceleration_at_particle = charge / mass * electric_field_at_particle

    return lapse_at_particle * lorentz_acceleration_at_particle
