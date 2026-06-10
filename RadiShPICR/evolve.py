import jax.numpy as jnp

from RadiShPICR.EM import (
    compute_charge_density_and_radial_electric_field,
    compute_radial_lorentz_force_terms,
)
from RadiShPICR.relativity.A import euler_step_A
from RadiShPICR.relativity.geodesic import compute_geodesic_terms
from RadiShPICR.relativity.metric import MetricState, compute_metric


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


def _flat_metric_state(grid):
    zeros = jnp.zeros_like(grid.r_full)
    ones = jnp.ones_like(grid.r_full)

    return MetricState(
        rho=zeros,
        A=ones,
        lapse=ones,
        shift=zeros,
        extrinsic_curvature=zeros,
        S_r=zeros,
        S_rr=zeros,
        exact_exterior_points=jnp.ones_like(grid.r_full, dtype=bool),
    )


def _compute_particle_derivatives(
    particles,
    fields,
    grid,
    schwarzschild_mass,
    shape_mode="nearest",
    source_particles=None,
    electric_field=None,
    GR=True,
    EM=True,
):

    u_r, _ = particles.get_velocity()
    zeros = jnp.zeros_like(u_r)
    fields_for_motion = fields
    mass_for_motion = schwarzschild_mass

    if not GR:
        fields_for_motion = _flat_metric_state(grid)
        mass_for_motion = 0.0
        # GR=False keeps the particle orbit equations, but evaluates them in
        # flat space so Schwarzschild boundary derivatives do not enter.

    dr_dt, dphi_dt, du_r_dt = compute_geodesic_terms(
        particles,
        fields_for_motion,
        grid,
        mass_for_motion,
        shape_mode=shape_mode,
    )
    # compute the orbit derivatives for the particle update

    if EM:
        if electric_field is None:
            if source_particles is None:
                source_particles = [particles]
            _, electric_field = compute_charge_density_and_radial_electric_field(
                _particle_list(source_particles),
                fields_for_motion.A,
                grid,
                shape_mode=shape_mode,
            )
        
        lorentz_du_r_dt = compute_radial_lorentz_force_terms(
            particles,
            fields_for_motion,
            grid,
            electric_field,
            shape_mode=shape_mode,
        )
        # if EM contributions are included, compute the Lorentz force terms

        du_r_dt = du_r_dt + lorentz_du_r_dt
        # add the Lorentz force contribution to the radial momentum equation

    return dr_dt, dphi_dt, du_r_dt


def _rk4_step_one_species(
    particles,
    fields,
    grid,
    dt,
    schwarzschild_mass,
    shape_mode="nearest",
    recompute_metric_each_stage=False,
    metric_A_solver="newton",
    source_particles=None,
    source_particle_index=None,
    GR=True,
    EM=True,
):
    dt_value = jnp.asarray(dt, dtype=particles.r.dtype)
    fields_for_step = fields
    if not GR:
        fields_for_step = _flat_metric_state(grid)

    if source_particles is None:
        source_particles = [particles]
        source_particle_index = 0

    k1_dr_dt, k1_dphi_dt, k1_du_r_dt = _compute_particle_derivatives(
        particles,
        fields_for_step,
        grid,
        schwarzschild_mass,
        shape_mode=shape_mode,
        source_particles=source_particles,
        GR=GR,
        EM=EM,
    )
    state_k2 = particles.with_updated_orbital_state(
        particles.r + 0.5 * dt_value * k1_dr_dt,
        particles.phi + 0.5 * dt_value * k1_dphi_dt,
        particles.u_r + 0.5 * dt_value * k1_du_r_dt,
    )
    # Dynamic runs solve the metric constraint at each RK stage state before
    # depositing charge and solving Gauss law for that same stage.
    if GR and recompute_metric_each_stage:
        source_k2 = _source_particles_at_stage(
            source_particles,
            source_particle_index,
            state_k2,
        )
        fields_k2 = compute_metric(
            source_k2,
            grid,
            schwarzschild_mass,
            initial_A_guess=fields_for_step.A,
            shape_mode=shape_mode,
            metric_A_solver=metric_A_solver,
            EM=EM,
        )
    else:
        fields_k2 = fields_for_step

    k2_dr_dt, k2_dphi_dt, k2_du_r_dt = _compute_particle_derivatives(
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
        GR=GR,
        EM=EM,
    )
    state_k3 = particles.with_updated_orbital_state(
        particles.r + 0.5 * dt_value * k2_dr_dt,
        particles.phi + 0.5 * dt_value * k2_dphi_dt,
        particles.u_r + 0.5 * dt_value * k2_du_r_dt,
    )
    if GR and recompute_metric_each_stage:
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
            metric_A_solver=metric_A_solver,
            EM=EM,
        )
    else:
        fields_k3 = fields_for_step

    k3_dr_dt, k3_dphi_dt, k3_du_r_dt = _compute_particle_derivatives(
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
        GR=GR,
        EM=EM,
    )
    state_k4 = particles.with_updated_orbital_state(
        particles.r + dt_value * k3_dr_dt,
        particles.phi + dt_value * k3_dphi_dt,
        particles.u_r + dt_value * k3_du_r_dt,
    )
    if GR and recompute_metric_each_stage:
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
            metric_A_solver=metric_A_solver,
            EM=EM,
        )
    else:
        fields_k4 = fields_for_step

    k4_dr_dt, k4_dphi_dt, k4_du_r_dt = _compute_particle_derivatives(
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
        GR=GR,
        EM=EM,
    )

    updated_r = particles.r + (dt_value / 6.0) * (
        k1_dr_dt + 2.0 * k2_dr_dt + 2.0 * k3_dr_dt + k4_dr_dt
    )
    updated_phi = particles.phi + (dt_value / 6.0) * (
        k1_dphi_dt + 2.0 * k2_dphi_dt + 2.0 * k3_dphi_dt + k4_dphi_dt
    )
    updated_u_r = particles.u_r + (dt_value / 6.0) * (
        k1_du_r_dt + 2.0 * k2_du_r_dt + 2.0 * k3_du_r_dt + k4_du_r_dt
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
    metric_A_solver="newton",
    GR=True,
    EM=True,
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
                    metric_A_solver=metric_A_solver,
                    source_particles=particle_list,
                    source_particle_index=species_index,
                    GR=GR,
                    EM=EM,
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
        metric_A_solver=metric_A_solver,
        GR=GR,
        EM=EM,
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
    metric_A_solver="newton",
    GR=True,
    EM=True,
):
    if GR and schwarzschild_mass is None:
        raise ValueError("advance_one_step requires an explicit schwarzschild_mass.")

    if schwarzschild_mass is None:
        current_mass = 0.0
    else:
        current_mass = float(schwarzschild_mass)

    if not GR:
        fields = _flat_metric_state(grid)
    elif fixed_fields is None:
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
            metric_A_solver=metric_A_solver,
            EM=EM,
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
        recompute_metric_each_stage=GR and fixed_fields is None,
        metric_A_solver=metric_A_solver,
        GR=GR,
        EM=EM,
    )

    if GR and fixed_fields is None:
        fields = compute_metric(
            updated_particles,
            grid,
            current_mass,
            initial_A_guess=fields.A,
            shape_mode=shape_mode,
            metric_A_solver=metric_A_solver,
            EM=EM,
        )

    return updated_particles, fields
