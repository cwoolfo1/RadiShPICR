import jax.numpy as jnp

from RadiShPICR.Z4C.constraint_terms import dGammadt, dZdt, dthetadt
from RadiShPICR.Z4C.extrinsic_curvature import dArrdt, dAtdt, dKhdt
from RadiShPICR.Z4C.shift_and_lapse import dalphadt, dbetadt
from RadiShPICR.Z4C.spatial_metric import dchidt, dgrrdt, dgtdt
from RadiShPICR.Z4C.z4c_metric import Z4C_Metric
from RadiShPICR.Z4C.energy_momentum_tensor import MatterTerms
from RadiShPICR.Z4C.energy_momentum_tensor import compute_radial_matter_terms
from RadiShPICR.Z4C.geodesic import compute_geodesic_terms


def metric_time_derivatives(metric: Z4C_Metric, matter_terms):
    zeros = jnp.zeros_like(metric.r)
    zero_dr = jnp.zeros_like(metric.dr)

    return Z4C_Metric(
        alpha=dalphadt(metric, matter_terms),
        beta=dbetadt(metric, matter_terms),
        conformal_grr=dgrrdt(metric, matter_terms),
        conformal_gt=dgtdt(metric, matter_terms),
        chi=dchidt(metric, matter_terms),
        Kh=dKhdt(metric, matter_terms),
        Arr=dArrdt(metric, matter_terms),
        At=dAtdt(metric, matter_terms),
        theta=dthetadt(metric, matter_terms),
        Zr=dZdt(metric, matter_terms),
        Gamma=dGammadt(metric, matter_terms),
        kappa=zeros,
        eta=zeros,
        r=zeros,
        dr=zero_dr,
    )


def _add_metric_derivative(metric, derivative, scale):
    return Z4C_Metric(
        alpha=metric.alpha + scale * derivative.alpha,
        beta=metric.beta + scale * derivative.beta,
        conformal_grr=metric.conformal_grr + scale * derivative.conformal_grr,
        conformal_gt=metric.conformal_gt + scale * derivative.conformal_gt,
        chi=metric.chi + scale * derivative.chi,
        Kh=metric.Kh + scale * derivative.Kh,
        Arr=metric.Arr + scale * derivative.Arr,
        At=metric.At + scale * derivative.At,
        theta=metric.theta + scale * derivative.theta,
        Zr=metric.Zr + scale * derivative.Zr,
        Gamma=metric.Gamma + scale * derivative.Gamma,
        kappa=metric.kappa + scale * derivative.kappa,
        eta=metric.eta + scale * derivative.eta,
        r=metric.r + scale * derivative.r,
        dr=metric.dr + scale * derivative.dr,
    )


def _combine_rk4_derivatives(k1, k2, k3, k4):
    return Z4C_Metric(
        alpha=k1.alpha + 2.0 * k2.alpha + 2.0 * k3.alpha + k4.alpha,
        beta=k1.beta + 2.0 * k2.beta + 2.0 * k3.beta + k4.beta,
        conformal_grr=(
            k1.conformal_grr
            + 2.0 * k2.conformal_grr
            + 2.0 * k3.conformal_grr
            + k4.conformal_grr
        ),
        conformal_gt=(
            k1.conformal_gt
            + 2.0 * k2.conformal_gt
            + 2.0 * k3.conformal_gt
            + k4.conformal_gt
        ),
        chi=k1.chi + 2.0 * k2.chi + 2.0 * k3.chi + k4.chi,
        Kh=k1.Kh + 2.0 * k2.Kh + 2.0 * k3.Kh + k4.Kh,
        Arr=k1.Arr + 2.0 * k2.Arr + 2.0 * k3.Arr + k4.Arr,
        At=k1.At + 2.0 * k2.At + 2.0 * k3.At + k4.At,
        theta=k1.theta + 2.0 * k2.theta + 2.0 * k3.theta + k4.theta,
        Zr=k1.Zr + 2.0 * k2.Zr + 2.0 * k3.Zr + k4.Zr,
        Gamma=k1.Gamma + 2.0 * k2.Gamma + 2.0 * k3.Gamma + k4.Gamma,
        kappa=k1.kappa + 2.0 * k2.kappa + 2.0 * k3.kappa + k4.kappa,
        eta=k1.eta + 2.0 * k2.eta + 2.0 * k3.eta + k4.eta,
        r=k1.r + 2.0 * k2.r + 2.0 * k3.r + k4.r,
        dr=k1.dr + 2.0 * k2.dr + 2.0 * k3.dr + k4.dr,
    )


def rk4_step(metric: Z4C_Metric, matter_terms, dt):
    k1 = metric_time_derivatives(metric, matter_terms)

    metric_k2 = _add_metric_derivative(metric, k1, 0.5 * dt)
    k2 = metric_time_derivatives(metric_k2, matter_terms)

    metric_k3 = _add_metric_derivative(metric, k2, 0.5 * dt)
    k3 = metric_time_derivatives(metric_k3, matter_terms)

    metric_k4 = _add_metric_derivative(metric, k3, dt)
    k4 = metric_time_derivatives(metric_k4, matter_terms)

    weighted_derivative = _combine_rk4_derivatives(k1, k2, k3, k4)

    return _add_metric_derivative(metric, weighted_derivative, dt / 6.0)


def particles_rk4_step(particles, metric: Z4C_Metric, dt):
    k1_dvr_dt, k1_duphi_dt, k1_drdt, k1_dphidt = compute_geodesic_terms(particles, metric)
    particles_k1 = particles.copy()
    particles_k1.set_positions(
        particles.get_positions()[0] + k1_drdt * (0.5 * dt),
        None,
    )
    particles_k1.set_velocities(
        particles.get_velocities()[0] + k1_dvr_dt * (0.5 * dt),
        particles.get_velocities()[1] + k1_duphi_dt * (0.5 * dt),
    )
    # update particle positions and velocities using k1
    # step 1: compute k1 using the current metric and geodesic terms
    matter_terms = compute_radial_matter_terms(particles, metric)
    k1 = metric_time_derivatives(metric, matter_terms)
    # step 1: compute k1 using the current metric and matter terms

    metric_k2 = _add_metric_derivative(metric, k1, 0.5 * dt)
    k2_dvr_dt, k2_duphi_dt, k2_drdt, k2_dphidt = compute_geodesic_terms(particles_k1, metric_k2)
    # step 2: compute k2 using the metric updated by k1 and the geodesic terms
    new_r_particle = particles.get_positions()[0] + k2_drdt * (0.5 * dt)
    new_ur = particles.get_velocities()[0] + k2_dvr_dt * (0.5 * dt)
    new_uphi = particles.get_velocities()[1] + k2_duphi_dt * (0.5 * dt)
    particles_k2 = particles.copy()
    particles_k2.set_positions(new_r_particle, None)
    particles_k2.set_velocities(new_ur, new_uphi)
    # update particle positions and velocities using k2

    matter_terms_k2 = compute_radial_matter_terms(particles_k1, metric_k2)
    k2 = metric_time_derivatives(metric_k2, matter_terms_k2)
    # step 2: compute k2 using the metric updated by k1 and the geodesic terms

    metric_k3 = _add_metric_derivative(metric, k2, 0.5 * dt)
    k3_dvr_dt, k3_duphi_dt, k3_drdt, k3_dphidt = compute_geodesic_terms(particles_k2, metric_k3)
    # step 3: compute k3 using the metric updated by k2 and the geodesic terms
    new_r_particle = particles.get_positions()[0] + k3_drdt * (0.5 * dt)
    new_ur = particles.get_velocities()[0] + k3_dvr_dt * (0.5 * dt)
    new_uphi = particles.get_velocities()[1] + k3_duphi_dt * (0.5 * dt)
    particles_k3 = particles.copy()
    particles_k3.set_positions(new_r_particle, None)
    particles_k3.set_velocities(new_ur, new_uphi)
    # update particle positions and velocities using k3

    matter_terms_k3 = compute_radial_matter_terms(particles_k2, metric_k3)
    k3 = metric_time_derivatives(metric_k3, matter_terms_k3)
    # step 3: compute k3 using the metric updated by k2 and the geodesic terms

    metric_k4 = _add_metric_derivative(metric, k3, dt)
    # step 4: compute k4 using the metric updated by k3
    k4_dvr_dt, k4_duphi_dt, k4_drdt, k4_dphidt = compute_geodesic_terms(particles_k3, metric_k4)
    new_r_particle = particles.get_positions()[0] + k4_drdt * dt
    new_ur = particles.get_velocities()[0] + k4_dvr_dt * dt
    new_uphi = particles.get_velocities()[1] + k4_duphi_dt * dt
    particles_k4 = particles.copy()
    particles_k4.set_positions(new_r_particle, None)
    particles_k4.set_velocities(new_ur, new_uphi)
    # update particle positions and velocities using k4


    matter_terms_k4 = compute_radial_matter_terms(particles_k4, metric_k4)
    k4 = metric_time_derivatives(metric_k4, matter_terms_k4)
    # step 4: compute k4 using the metric updated by k3 and the geodesic terms

    weighted_derivative = _combine_rk4_derivatives(k1, k2, k3, k4)

    final_metric = _add_metric_derivative(metric, weighted_derivative, dt / 6.0)
    # finish the RK4 step by combining the weighted derivatives and updating the metric

    new_r_particle = particles.get_positions()[0] + (dt / 6.0) * (k1_drdt + 2 * k2_drdt + 2 * k3_drdt + k4_drdt)
    new_ur = particles.get_velocities()[0] + (dt / 6.0) * (k1_dvr_dt + 2 * k2_dvr_dt + 2 * k3_dvr_dt + k4_dvr_dt)
    new_uphi = particles.get_velocities()[1] + (dt / 6.0) * (k1_duphi_dt + 2 * k2_duphi_dt + 2 * k3_duphi_dt + k4_duphi_dt)
    particles.set_positions(new_r_particle, None)
    particles.set_velocities(new_ur, new_uphi)
    # update particle positions and velocities using the final metric


    return particles, final_metric
