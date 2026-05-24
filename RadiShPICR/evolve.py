import jax.numpy as jnp

from RadiShPICR.EM import compute_radial_lorentz_force_terms, solve_radial_electric_field
from RadiShPICR.relativity.A import euler_step_A
from RadiShPICR.relativity.geodesic import compute_geodesic_terms
from RadiShPICR.relativity.metric import compute_metric


def _compute_particle_derivatives(
    particles,
    fields,
    grid,
    schwarzschild_mass,
    shape_mode="nearest",
):
    geodesic_terms = compute_geodesic_terms(
        particles,
        fields,
        grid,
        schwarzschild_mass,
        shape_mode=shape_mode,
    )
    electric_solve = solve_radial_electric_field(
        particles,
        fields.A,
        grid,
        shape_mode=shape_mode,
    )
    lorentz_terms = compute_radial_lorentz_force_terms(
        particles,
        fields,
        grid,
        electric_solve.electric_field,
        shape_mode=shape_mode,
    )

    return geodesic_terms._replace(
        du_r_dt=geodesic_terms.du_r_dt + lorentz_terms.du_r_dt,
    )


def rk4_step(particles, fields, grid, dt, schwarzschild_mass, shape_mode="nearest"):
    dt_value = jnp.asarray(dt, dtype=particles.r.dtype)

    k1 = _compute_particle_derivatives(
        particles,
        fields,
        grid,
        schwarzschild_mass,
        shape_mode=shape_mode,
    )
    state_k2 = particles.with_updated_orbital_state(
        particles.r + 0.5 * dt_value * k1.dr_dt,
        particles.phi + 0.5 * dt_value * k1.dphi_dt,
        particles.u_r + 0.5 * dt_value * k1.du_r_dt,
    )

    k2 = _compute_particle_derivatives(
        state_k2,
        fields,
        grid,
        schwarzschild_mass,
        shape_mode=shape_mode,
    )
    state_k3 = particles.with_updated_orbital_state(
        particles.r + 0.5 * dt_value * k2.dr_dt,
        particles.phi + 0.5 * dt_value * k2.dphi_dt,
        particles.u_r + 0.5 * dt_value * k2.du_r_dt,
    )

    k3 = _compute_particle_derivatives(
        state_k3,
        fields,
        grid,
        schwarzschild_mass,
        shape_mode=shape_mode,
    )
    state_k4 = particles.with_updated_orbital_state(
        particles.r + dt_value * k3.dr_dt,
        particles.phi + dt_value * k3.dphi_dt,
        particles.u_r + dt_value * k3.du_r_dt,
    )

    k4 = _compute_particle_derivatives(
        state_k4,
        fields,
        grid,
        schwarzschild_mass,
        shape_mode=shape_mode,
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
            schwarzschild_mass,
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
    )

    return updated_particles, fields
