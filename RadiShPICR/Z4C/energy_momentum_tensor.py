from typing import NamedTuple

import jax.numpy as jnp

from RadiShPICR.Z4C.z4c_metric import Z4C_Metric
from RadiShPICR.particles.particle_shapes import interpolate_field_to_particles
from RadiShPICR.particles.particle_shapes import shape_weights_at_point

class MatterTerms(NamedTuple):
    rho: jnp.ndarray
    # energy density
    Srr: jnp.ndarray
    Stt: jnp.ndarray
    # stress tensor components
    Sr: jnp.ndarray
    St: jnp.ndarray
    # momentum density


def initialize_vacuum_matter_terms(metric):
    zeros = jnp.zeros_like(metric.r)

    return MatterTerms(
        rho=zeros,
        Srr=zeros,
        Stt=zeros,
        Sr=zeros,
        St=zeros,
    )


def compute_radial_matter_terms(particles, metric: Z4C_Metric):
    # assuming only radial dependence

    rho = relativistic_mass_energy_density(particles, metric)
    Srr = compute_radial_stress_tensor_component(particles, metric)
    Sr = compute_radial_momentum_density(particles, metric)

    return MatterTerms(
        rho=rho,
        Srr=Srr,
        Stt=jnp.zeros_like(rho),
        Sr=Sr,
        St=jnp.zeros_like(rho),
    )


def relativistic_mass_energy_density(particles, metric: Z4C_Metric):
    r_particle, _ = particles.get_positions()
    ur, uphi = particles.get_velocities()
    particle_shape = particles.get_shape()
    # unpack particle positions, velocities, and shape
    chi = metric.chi
    conformal_grr = metric.conformal_grr
    conformal_gt = metric.conformal_gt
    r = metric.r
    dr = metric.dr
    # unpack metric components


    scaling_factor = jnp.sqrt( 1 / chi**3 )
    # compute the scaling factor for the energy density based on the metric components
    scaling_factor_p = interpolate_field_to_particles(scaling_factor, r_particle, r, dr, particle_shape)
    # interpolate the scaling factor to particle positions
    grr_p = interpolate_field_to_particles(conformal_grr, r_particle, r, dr, particle_shape)
    gt_p = interpolate_field_to_particles(conformal_gt, r_particle, r, dr, particle_shape)
    # interpolate the conformal metric components to particle positions

    particle_volume_element = 4 * jnp.pi * r_particle**2 * scaling_factor_p
    # compute the volume element for each particle based on the interpolated scaling factor

    lorentz_factor = jnp.sqrt( 1 + ur**2 / grr_p  + uphi**2 / (r_particle**2 * gt_p)  )
    # compute the Lorentz factor for each particle based on its velocities and the interpolated metric components

    weights = shape_weights_at_point(r_particle, r, dr, particle_shape)
    # compute the shape weights for each particle based on its position and the grid points

    energy_density = jnp.sum(particles.get_mass() * weights * lorentz_factor / particle_volume_element)
    # compute the total energy density by summing over all particles, weighted by their mass,
    # shape weights, Lorentz factor, and volume element

    return energy_density


def compute_radial_momentum_density(particles, metric: Z4C_Metric):
    r_particle, _ = particles.get_positions()
    ur, uphi = particles.get_velocities()
    particle_shape = particles.get_shape()
    # unpack particle positions, velocities, and shape
    chi = metric.chi
    conformal_grr = metric.conformal_grr
    conformal_gt = metric.conformal_gt
    r = metric.r
    dr = metric.dr
    # unpack metric components

    weights = shape_weights_at_point(r_particle, r, dr, particle_shape)
    # compute the shape weights for each particle based on its position and the grid points

    scaling_factor = jnp.sqrt( 1 / chi**3 )
    # compute the scaling factor for the energy density based on the metric components
    scaling_factor_p = interpolate_field_to_particles(scaling_factor, r_particle, r, dr, particle_shape)
    # interpolate the scaling factor to particle positions

    particle_volume_element = 4 * jnp.pi * r_particle**2 * scaling_factor_p
    # compute the volume element for each particle based on the interpolated scaling factor

    return jnp.sum(particles.get_mass() * weights * ur / particle_volume_element)
    # compute the total radial momentum density by summing over all particles, weighted by their mass,

def compute_radial_stress_tensor_component(particles, metric: Z4C_Metric):
    r_particle, _ = particles.get_positions()
    ur, uphi = particles.get_velocities()
    particle_shape = particles.get_shape()
    # unpack particle positions, velocities, and shape
    chi = metric.chi
    conformal_grr = metric.conformal_grr
    conformal_gt = metric.conformal_gt
    r = metric.r
    dr = metric.dr
    # unpack metric components

    weights = shape_weights_at_point(r_particle, r, dr, particle_shape)
    # compute the shape weights for each particle based on its position and the grid points

    scaling_factor = jnp.sqrt( 1 / chi**3 )
    # compute the scaling factor for the energy density based on the metric components
    scaling_factor_p = interpolate_field_to_particles(scaling_factor, r_particle, r, dr, particle_shape)
    # interpolate the scaling factor to particle positions

    grr_p = interpolate_field_to_particles(conformal_grr, r_particle, r, dr, particle_shape)
    gt_p = interpolate_field_to_particles(conformal_gt, r_particle, r, dr, particle_shape)
    # interpolate the conformal metric components to particle positions

    particle_volume_element = 4 * jnp.pi * r_particle**2 * scaling_factor_p
    # compute the volume element for each particle based on the interpolated scaling factor

    lorentz_factor = jnp.sqrt( 1 + ur**2 / grr_p  + uphi**2 / (r_particle**2 * gt_p)  )
    # compute the Lorentz factor for each particle based on its velocities and the interpolated metric components

    return jnp.sum(particles.get_mass() * weights * ur**2 / (particle_volume_element * lorentz_factor))
    # compute the total radial stress tensor component by summing over all particles, weighted by their mass,