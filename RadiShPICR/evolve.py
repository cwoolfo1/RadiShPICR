from RadiShPICR.forces.geodesic import compute_geodesic_terms
from RadiShPICR.forces.lorentz_force import compute_lorentz_terms
from RadiShPICR.forces.solve_metric import calculate_metric
from RadiShPICR.forces.utils import safe_radius


def step(particles, r_grid, dr, dt):
    U_state = calculate_metric(particles, r_grid, dr)

    dr_dt, dur_dt_GR = compute_geodesic_terms(particles, U_state)
    dur_dt_EM = compute_lorentz_terms(particles, U_state)
    dur_dt = dur_dt_GR + dur_dt_EM

    r, phi = particles.get_positions()
    ur, uphi = particles.get_velocities()

    particles.r = r + dr_dt * dt
    particles.ur = ur + dur_dt * dt
    particles.phi = phi + uphi * dt / safe_radius(r, 0.5 * dr)
    particles.uphi = uphi

    return particles


def _copy_particle_state(particles, r, phi, ur):
    return type(particles)(
        name=particles.name,
        charge=particles.charges,
        mass=particles.masses,
        weight=particles.weight,
        r=r,
        ur=ur,
        phi=phi,
        uphi=particles.uphi,
        shape_mode=particles.shape_mode,
    )


def _particle_derivatives(particles, r_grid, dr):
    U_state = calculate_metric(particles, r_grid, dr)
    dr_dt, dur_dt_GR = compute_geodesic_terms(particles, U_state)
    dur_dt_EM = compute_lorentz_terms(particles, U_state)

    r, phi = particles.get_positions()
    ur, uphi = particles.get_velocities()
    dphi_dt = uphi / safe_radius(r, 0.5 * dr)

    return dr_dt, dphi_dt, dur_dt_GR + dur_dt_EM


def step_rk4(particles, r_grid, dr, dt):
    r0, phi0 = particles.get_positions()
    ur0, uphi0 = particles.get_velocities()

    k1_r, k1_phi, k1_ur = _particle_derivatives(particles, r_grid, dr)

    stage2 = _copy_particle_state(
        particles,
        r0 + 0.5 * dt * k1_r,
        phi0 + 0.5 * dt * k1_phi,
        ur0 + 0.5 * dt * k1_ur,
    )
    k2_r, k2_phi, k2_ur = _particle_derivatives(stage2, r_grid, dr)

    stage3 = _copy_particle_state(
        particles,
        r0 + 0.5 * dt * k2_r,
        phi0 + 0.5 * dt * k2_phi,
        ur0 + 0.5 * dt * k2_ur,
    )
    k3_r, k3_phi, k3_ur = _particle_derivatives(stage3, r_grid, dr)

    stage4 = _copy_particle_state(
        particles,
        r0 + dt * k3_r,
        phi0 + dt * k3_phi,
        ur0 + dt * k3_ur,
    )
    k4_r, k4_phi, k4_ur = _particle_derivatives(stage4, r_grid, dr)

    particles.r = r0 + (dt / 6.0) * (k1_r + 2.0 * k2_r + 2.0 * k3_r + k4_r)
    particles.phi = phi0 + (dt / 6.0) * (
        k1_phi + 2.0 * k2_phi + 2.0 * k3_phi + k4_phi
    )
    particles.ur = ur0 + (dt / 6.0) * (k1_ur + 2.0 * k2_ur + 2.0 * k3_ur + k4_ur)
    particles.uphi = uphi0

    return particles
