from typing import NamedTuple

import jax.numpy as jnp

from RadiShPICR.ConstraintBasedRelativity.grid import RadialGrid
from RadiShPICR.Z4C.derivatives import first_derivative, second_derivative
from RadiShPICR.Z4C.derivatives import first_derivative
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


def _radial_grid_from_metric(metric: Z4C_Metric):
    return RadialGrid(
        r_full=metric.r,
        r_interior=metric.r,
        dr=metric.dr,
        r_max=metric.r[-1],
    )


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
    grid = _radial_grid_from_metric(metric)


    scaling_factor = jnp.sqrt( 1 / chi**3 )
    # compute the scaling factor for the energy density based on the metric components
    scaling_factor_p = interpolate_field_to_particles(scaling_factor, r_particle, grid, shape_mode=particle_shape)
    # interpolate the scaling factor to particle positions
    grr_p = interpolate_field_to_particles(conformal_grr / chi, r_particle, grid, shape_mode=particle_shape)
    gt_p = interpolate_field_to_particles(conformal_gt / chi, r_particle, grid, shape_mode=particle_shape)
    # interpolate the conformal metric components to particle positions

    particle_volume_element = 4 * jnp.pi * r_particle**2 * scaling_factor_p
    # compute the volume element for each particle based on the interpolated scaling factor

    lorentz_factor = jnp.sqrt( 1 + ur**2 / grr_p  + uphi**2 / (r_particle**2 * gt_p)  )
    # compute the Lorentz factor for each particle based on its velocities and the interpolated metric components

    weights = shape_weights_at_point(r_particle[jnp.newaxis, :], r[:, jnp.newaxis], dr, particle_shape)
    # compute the shape weights for each particle based on its position and the grid points

    energy_density = jnp.sum(particles.get_mass() * weights * lorentz_factor / particle_volume_element, axis=1)
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
    grid = _radial_grid_from_metric(metric)

    weights = shape_weights_at_point(r_particle[jnp.newaxis, :], r[:, jnp.newaxis], dr, particle_shape)
    # compute the shape weights for each particle based on its position and the grid points

    scaling_factor = jnp.sqrt( 1 / chi**3 )
    # compute the scaling factor for the energy density based on the metric components
    scaling_factor_p = interpolate_field_to_particles(scaling_factor, r_particle, grid, shape_mode=particle_shape)
    # interpolate the scaling factor to particle positions

    particle_volume_element = 4 * jnp.pi * r_particle**2 * scaling_factor_p
    # compute the volume element for each particle based on the interpolated scaling factor

    return jnp.sum(particles.get_mass() * weights * ur / particle_volume_element, axis=1)
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
    grid = _radial_grid_from_metric(metric)

    weights = shape_weights_at_point(r_particle[jnp.newaxis, :], r[:, jnp.newaxis], dr, particle_shape)
    # compute the shape weights for each particle based on its position and the grid points

    scaling_factor = jnp.sqrt( 1 / chi**3 )
    # compute the scaling factor for the energy density based on the metric components
    scaling_factor_p = interpolate_field_to_particles(scaling_factor, r_particle, grid, shape_mode=particle_shape)
    # interpolate the scaling factor to particle positions

    grr_p = interpolate_field_to_particles(conformal_grr / chi, r_particle, grid, shape_mode=particle_shape)
    gt_p = interpolate_field_to_particles(conformal_gt / chi, r_particle, grid, shape_mode=particle_shape)
    # interpolate the conformal metric components to particle positions

    particle_volume_element = 4 * jnp.pi * r_particle**2 * scaling_factor_p
    # compute the volume element for each particle based on the interpolated scaling factor

    lorentz_factor = jnp.sqrt( 1 + ur**2 / grr_p  + uphi**2 / (r_particle**2 * gt_p)  )
    # compute the Lorentz factor for each particle based on its velocities and the interpolated metric components

    return jnp.sum(particles.get_mass() * weights * ur**2 / (particle_volume_element * lorentz_factor), axis=1)
    # compute the total radial stress tensor component by summing over all particles, weighted by their mass,




def compute_hamiltonian_constraint(metric: Z4C_Metric):

    # ASSUMES VACUUM and IGNORES THETA FOR NOW. NEEDS TO BE FIXED FOR NON-VACUUM CASES
    
    chi = metric.chi
    grr = metric.conformal_grr
    gt = metric.conformal_gt
    Arr = metric.Arr
    At = metric.At
    K  = metric.Kh
    r  = metric.r

    dchidr = first_derivative(chi, metric.dr, parity=1)
    dgrrdr = first_derivative(grr, metric.dr, parity=1)
    dgtdr = first_derivative(gt, metric.dr, parity=1)
    d2gtdr     = second_derivative(gt, metric.dr, parity=1 )
    d2chidr    = second_derivative(chi, metric.dr, parity=1)


    constraint =  -(Arr**2/grr**2) + (2*d2chidr)/grr - (5*dchidr**2)/(2*(jnp.maximum(chi, 1e-10))*grr) - (2*At**2)/gt**2 + (2*K**2)/3 + \
        dchidr*(-(dgrrdr/grr**2) + (2*dgtdr)/(grr*gt) + 4/(grr*r)) + \
        chi*(dgtdr**2/(2*grr*gt**2) + (dgrrdr*dgtdr)/(grr**2*gt) - (2*d2gtdr)/(grr*gt) - 2/(grr*r**2) + 2/(gt*r**2) + 
        (2*dgrrdr)/(grr**2*r) - (6*dgtdr)/(grr*gt*r))


    return constraint
