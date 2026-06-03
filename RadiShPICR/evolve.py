import jax.numpy as jnp

from RadiShPICR.EM import (
    compute_charge_density_and_radial_electric_field,
    compute_radial_lorentz_force_terms,
)
from RadiShPICR.relativity.A import euler_step_A
from RadiShPICR.relativity.geodesic import compute_geodesic_terms, GeodesicTerms
from RadiShPICR.relativity.metric import compute_metric


def _particle_list(particles):
    if isinstance(particles, (list, tuple)):
        return list(particles)
    return [particles]


def _restore_particle_structure(original_particles, updated_particle_list):
    if isinstance(original_particles, tuple):
        return tuple(updated_particle_list)
    if isinstance(original_particles, list):
        return updated_particle_list
    return updated_particle_list[0]


def _source_particles_at_stage(source_particles, source_particle_index, stage_particles):
    if source_particles is None:
        return None

    staged_source_particles = list(source_particles)
    staged_source_particles[source_particle_index] = stage_particles
    return staged_source_particles


def _compute_particle_derivatives(
    particles,
    fields,
    grid,
    schwarzschild_mass,
    shape_mode="nearest",
    source_particles=None,
    electric_field=None,
    GR = True,
    EM = True,
):
    
    u_r, u_phi = particles.get_velocity()
    zeros = jnp.zeros_like(u_r)
    geodesic_terms = GeodesicTerms(
        dr_dt=zeros,
        dphi_dt=zeros,
        du_r_dt=zeros,
    )
    # initialize empty tuple

    if GR:
        geodesic_terms = compute_geodesic_terms(
            particles,
            fields,
            grid,
            schwarzschild_mass,
            shape_mode=shape_mode,
        )
        # compute the geodesic terms for the particle derivatives


    if EM:
        if electric_field is None:
            if source_particles is None:
                source_particles = [particles]
            _, electric_field = compute_charge_density_and_radial_electric_field(
                _particle_list(source_particles),
                fields.A,
                grid,
                shape_mode=shape_mode,
            )
        
        lorentz_terms = compute_radial_lorentz_force_terms(
            particles,
            fields,
            grid,
            electric_field,
            shape_mode=shape_mode,
        )
        # if EM contributions are included, compute the Lorentz force terms

        geodesic_terms = geodesic_terms._replace(
            du_r_dt=geodesic_terms.du_r_dt + lorentz_terms.du_r_dt,
        )
        # update the geodesic terms to include the Lorentz force contributions to du_r_dt

    return geodesic_terms


def _rk4_step_one_species(
    particles,
    fields,
    grid,
    dt,
    schwarzschild_mass,
    shape_mode="nearest",
    recompute_metric_each_stage=False,
    source_particles=None,
    source_particle_index=None,
):
    dt_value = jnp.asarray(dt, dtype=particles.r.dtype)
    if source_particles is None:
        source_particles = [particles]
        source_particle_index = 0

    k1 = _compute_particle_derivatives(
        particles,
        fields,
        grid,
        schwarzschild_mass,
        shape_mode=shape_mode,
        source_particles=source_particles,
    )
    state_k2 = particles.with_updated_orbital_state(
        particles.r + 0.5 * dt_value * k1.dr_dt,
        particles.phi + 0.5 * dt_value * k1.dphi_dt,
        particles.u_r + 0.5 * dt_value * k1.du_r_dt,
    )
    # Dynamic runs solve the metric constraint at each RK stage state before
    # depositing charge and solving Gauss law for that same stage.
    if recompute_metric_each_stage:
        source_k2 = _source_particles_at_stage(
            source_particles,
            source_particle_index,
            state_k2,
        )
        fields_k2 = compute_metric(
            source_k2,
            grid,
            schwarzschild_mass,
            initial_A_guess=fields.A,
            shape_mode=shape_mode,
        )
    else:
        fields_k2 = fields

    k2 = _compute_particle_derivatives(
        state_k2,
        fields_k2,
        grid,
        schwarzschild_mass,
        shape_mode=shape_mode,
        source_particles=_source_particles_at_stage(
            source_particles,
            source_particle_index,
            state_k2,
        ),
    )
    state_k3 = particles.with_updated_orbital_state(
        particles.r + 0.5 * dt_value * k2.dr_dt,
        particles.phi + 0.5 * dt_value * k2.dphi_dt,
        particles.u_r + 0.5 * dt_value * k2.du_r_dt,
    )
    if recompute_metric_each_stage:
        source_k3 = _source_particles_at_stage(
            source_particles,
            source_particle_index,
            state_k3,
        )
        fields_k3 = compute_metric(
            source_k3,
            grid,
            schwarzschild_mass,
            initial_A_guess=fields_k2.A,
            shape_mode=shape_mode,
        )
    else:
        fields_k3 = fields

    k3 = _compute_particle_derivatives(
        state_k3,
        fields_k3,
        grid,
        schwarzschild_mass,
        shape_mode=shape_mode,
        source_particles=_source_particles_at_stage(
            source_particles,
            source_particle_index,
            state_k3,
        ),
    )
    state_k4 = particles.with_updated_orbital_state(
        particles.r + dt_value * k3.dr_dt,
        particles.phi + dt_value * k3.dphi_dt,
        particles.u_r + dt_value * k3.du_r_dt,
    )
    if recompute_metric_each_stage:
        source_k4 = _source_particles_at_stage(
            source_particles,
            source_particle_index,
            state_k4,
        )
        fields_k4 = compute_metric(
            source_k4,
            grid,
            schwarzschild_mass,
            initial_A_guess=fields_k3.A,
            shape_mode=shape_mode,
        )
    else:
        fields_k4 = fields

    k4 = _compute_particle_derivatives(
        state_k4,
        fields_k4,
        grid,
        schwarzschild_mass,
        shape_mode=shape_mode,
        source_particles=_source_particles_at_stage(
            source_particles,
            source_particle_index,
            state_k4,
        ),
    )

    updated_r = particles.r + (dt_value / 6.0) * (
        k1.dr_dt + 2.0 * k2.dr_dt + 2.0 * k3.dr_dt + k4.dr_dt
    )
    updated_phi = particles.phi + (dt_value / 6.0) * (
        k1.dphi_dt + 2.0 * k2.dphi_dt + 2.0 * k3.dphi_dt + k4.dphi_dt
    )
    updated_u_r = particles.u_r + (dt_value / 6.0) * (
        k1.du_r_dt + 2.0 * k2.du_r_dt + 2.0 * k3.du_r_dt + k4.du_r_dt
    )

    return particles.with_updated_orbital_state(updated_r, updated_phi, updated_u_r)


def rk4_step(
    particles,
    fields,
    grid,
    dt,
    schwarzschild_mass,
    shape_mode="nearest",
    recompute_metric_each_stage=False,
):
    if isinstance(particles, (list, tuple)):
        particle_list = _particle_list(particles)
        updated_particle_list = []
        for species_index, species in enumerate(particle_list):
            updated_particle_list.append(
                _rk4_step_one_species(
                    species,
                    fields,
                    grid,
                    dt,
                    schwarzschild_mass,
                    shape_mode=shape_mode,
                    recompute_metric_each_stage=recompute_metric_each_stage,
                    source_particles=particle_list,
                    source_particle_index=species_index,
                )
            )
        return _restore_particle_structure(particles, updated_particle_list)

    return _rk4_step_one_species(
        particles,
        fields,
        grid,
        dt,
        schwarzschild_mass,
        shape_mode=shape_mode,
        recompute_metric_each_stage=recompute_metric_each_stage,
    )


def advance_one_step(
    particles,
    grid,
    dt,
    schwarzschild_mass=None,
    initial_A_guess=None,
    previous_fields=None,
    fixed_fields=None,
    remove_escaped_particles=True,
    shape_mode="nearest",
):
    if schwarzschild_mass is None:
        raise ValueError("advance_one_step requires an explicit schwarzschild_mass.")

    current_mass = float(schwarzschild_mass)

    if fixed_fields is None:
        if previous_fields is None:
            prepared_initial_A_guess = initial_A_guess
        else:
            prepared_initial_A_guess = euler_step_A(
                previous_fields,
                grid,
                dt,
                current_mass,
            )
        fields = compute_metric(
            particles,
            grid,
            current_mass,
            initial_A_guess=prepared_initial_A_guess,
            shape_mode=shape_mode,
        )
    else:
        fields = fixed_fields

    updated_particles = rk4_step(
        particles,
        fields,
        grid,
        dt,
        current_mass,
        shape_mode=shape_mode,
        recompute_metric_each_stage=fixed_fields is None,
    )

    if fixed_fields is None:
        fields = compute_metric(
            updated_particles,
            grid,
            current_mass,
            initial_A_guess=fields.A,
            shape_mode=shape_mode,
        )

    return updated_particles, fields
