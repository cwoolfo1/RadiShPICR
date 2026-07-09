import jax.numpy as jnp
import jax

from RadiShPICR.particles.particle_shapes import interpolate_field_to_particles
from RadiShPICR.ConstraintBasedRelativity.grid import RadialGrid
from RadiShPICR.Z4C.z4c_metric import Z4C_Metric
from RadiShPICR.Z4C.derivatives import first_derivative, second_derivative


def _radial_grid_from_metric(metric: Z4C_Metric):
    return RadialGrid(
        r_full=metric.r,
        r_interior=metric.r,
        dr=metric.dr,
        epsilon=0.5 * metric.dr,
        r_max=metric.r[-1],
    )


def compute_geodesic_terms(particles, metric: Z4C_Metric):
    r_particle, _ = particles.get_positions()
    ur, uphi = particles.get_velocities()
    particle_shape = particles.get_shape()
    # unpack particle positions, velocities, and shape

    alpha = metric.alpha
    beta = metric.beta
    # get the lapse and shift from the metric
    chi = metric.chi
    conformal_grr = metric.conformal_grr
    conformal_gt = metric.conformal_gt
    Arr = metric.Arr
    At = metric.At
    KTh = metric.Kh
    r = metric.r
    dr = metric.dr
    # unpack metric components
    grid = _radial_grid_from_metric(metric)

    dalphadr = first_derivative(alpha, dr, parity=1)
    dbetadr = first_derivative(beta, dr, parity=-1)
    dchidr = first_derivative(chi, dr, parity=1)
    dgrrdr = first_derivative(conformal_grr, dr, parity=1)
    dgtdr = first_derivative(conformal_gt, dr, parity=1)
    # compute derivatives of the metric functions using finite difference methods

    grr_p = interpolate_field_to_particles(conformal_grr, r_particle, grid, shape_mode=particle_shape)
    gt_p = interpolate_field_to_particles(conformal_gt, r_particle, grid, shape_mode=particle_shape)
    alpha_p = interpolate_field_to_particles(alpha, r_particle, grid, shape_mode=particle_shape)
    beta_p = interpolate_field_to_particles(beta, r_particle, grid, shape_mode=particle_shape)
    chi_p = interpolate_field_to_particles(chi, r_particle, grid, shape_mode=particle_shape)
    Arr_p = interpolate_field_to_particles(Arr, r_particle, grid, shape_mode=particle_shape)
    At_p = interpolate_field_to_particles(At, r_particle, grid, shape_mode=particle_shape)
    KTh_p = interpolate_field_to_particles(KTh, r_particle, grid, shape_mode=particle_shape)
    dalphadr_p = interpolate_field_to_particles(dalphadr, r_particle, grid, shape_mode=particle_shape)
    dbetadr_p = interpolate_field_to_particles(dbetadr, r_particle, grid, shape_mode=particle_shape)
    dchidr_p = interpolate_field_to_particles(dchidr, r_particle, grid, shape_mode=particle_shape)
    dgrrdr_p = interpolate_field_to_particles(dgrrdr, r_particle, grid, shape_mode=particle_shape)
    dgtdr_p = interpolate_field_to_particles(dgtdr, r_particle, grid, shape_mode=particle_shape)
    # interpolate metric functions and their derivatives to particle

    #    1/(6 \[Chi][t, r])
    #  gT[t, r]^2 (6 \[Chi][t, 
    #      r] ((Ur^2/gT[t, r]^2 - \[Chi][t, r]) dalphadr - (Ur 
    # \!\(\*SuperscriptBox[\(b\), 
    # TagBox[
    # RowBox[{"(", 
    # RowBox[{"0", ",", "1"}], ")"}],
    # Derivative],
    # MultilineFunction->None]\)[t, r])/gT[t, r]^2) + 
    #    alpha[t, 
    #      r] (-((2 Ur^3 KTh[t, r])/gT[t, r]^4) + 
    #       3 \[Chi][t, 
    #         r] (Ur (4 Arr[t, r] \[Chi][t, r] - Ur dgrrdr) + 
    #          r Uphi^2 (2 gT[t, r] + rdgtdr)) - 
    #       3 r^2 Uphi^2 gT[t, r] dchidr + (
    #       Ur (-2 Uphi^2 gT[t, r] KTh[t, r] - 
    #          2 (3 Ur^2 Arr[t, r] + 3 Uphi^2 AT[t, r] - 
    #             2 KTh[t, r]) \[Chi][t, r] + 3 Ur dchidr))/gT[t, r]^2))
    # original expression from mathematica notebook

    dvr_dt = 1 / (6 * chi_p) * gt_p**2 * (
        6 * chi_p * ((ur**2 / gt_p**2 - chi_p) * dalphadr_p - (ur * dbetadr_p) / gt_p**2)
        + alpha_p * (
            -(2 * ur**3 * KTh_p) / gt_p**4
            + 3 * chi_p * (ur * (4 * Arr_p * chi_p - ur * dgrrdr_p) + r_particle * uphi**2 * (2 * gt_p + r_particle * dgtdr_p))
            - 3 * r_particle**2 * uphi**2 * gt_p * dchidr_p
            + (ur *(-2 * uphi**2 * gt_p * KTh_p - 2 * (3 * ur**2 * Arr_p + 3 * uphi**2 * At_p - 2 * KTh_p) * chi_p + 3 * ur * dchidr_p)) / gt_p**2
        )
    )
    # compute radial acceleration


    # -((2 Uphi Ur alpha[t, r])/r) - Uphi Ur^2 alpha[t, r] Arr[t, r] - 
    #  Uphi^3 alpha[t, r] AT[t, r] - (
    #  Uphi alpha[t, r] (Ur^2/gT[t, r]^2 + Uphi^2 gT[t, r]) KTh[t, r])/(
    #  3 \[Chi][t, r]) + (
    #  2/3 Uphi alpha[t, r] KTh[t, r] + (
    #   2 Uphi alpha[t, r] AT[t, r] \[Chi][t, r])/gT[t, r])/r^2 + 
    #  Uphi Ur dalphadr - (Uphi Ur alpha[t, r] dgtdr)/gT[t, r] + (
    #  Uphi Ur alpha[t, r] dchidr)/\[Chi][t, r]
    # original expression from mathematica notebook

    duphi_dt = -( (2 * uphi * ur * alpha_p)/r_particle )
    duphi_dt += -uphi * ur**2 * alpha_p * Arr_p
    duphi_dt += -uphi**3 * alpha_p * At_p
    duphi_dt += -uphi * alpha_p * (ur**2 / gt_p**2 + uphi**2 * gt_p) * KTh_p / (3 * chi_p)
    duphi_dt += (2/3 * uphi * alpha_p * KTh_p + (2 * uphi * alpha_p * At_p * chi_p) / gt_p) / r_particle**2
    duphi_dt += uphi * ur * dalphadr_p
    duphi_dt += -(uphi * ur * alpha_p * dgtdr_p) / gt_p
    duphi_dt += (uphi * ur * alpha_p * dchidr_p) / chi_p
    # compute angular acceleration


    drdt = alpha_p * ur - beta_p
    dphidt = alpha_p * uphi
    # compute the time derivatives of the particle's radial and angular positions using the lapse and shift

    return dvr_dt, duphi_dt, drdt, dphidt
