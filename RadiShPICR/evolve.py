from RadiShPICR.relativity.solve_metric import calculate_metric
from RadiShPICR.relativity.geodesic import compute_geodesic_terms
from RadiShPICR.EM.lorentz_force import compute_lorentz_terms
import jax.numpy as jnp



def step(particles, r_grid, dr, dt):
    # calculate metric
    A_values, phi_values, alpha_values, beta_over_r_values, Krr_values, Er_values, source_terms_values = calculate_metric(particles, r_grid, dr)

    # update particle positions and momenta using the calculated metric
    # for each particle:
    metric_terms = (A_values, phi_values, alpha_values, beta_over_r_values, Krr_values, Er_values, source_terms_values)

    dr_dt, dur_dt_GR = compute_geodesic_terms(particles, metric_terms, r_grid, dr)
    # compute relativistic corrections to the Lorentz force
    dur_dt_EM = compute_lorentz_terms(particles, metric_terms, r_grid, dr)
    # compute the lorentz force
    # update momenta using the geodesic and lorentz terms
    # for each particle:
    dur_dt = dur_dt_GR + dur_dt_EM

    r, phi = particles.get_positions()
    ur, uphi = particles.get_velocities()

    r_new = r + dr_dt * dt
    ur_new = ur + dur_dt * dt
    phi_new = phi + uphi * dt / r
    uphi_new = uphi # uphi is treated as constant

    # update particle species with new positions and momenta
    particles.r = r_new
    particles.ur = ur_new
    particles.phi = phi_new
    particles.uphi = uphi_new

    return particles
    



