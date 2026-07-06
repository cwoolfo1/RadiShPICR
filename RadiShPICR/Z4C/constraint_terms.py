import jax.numpy as jnp
from RadiShPICR.Z4C.derivatives import first_derivative, second_derivative, sixth_derivative
from RadiShPICR.Z4C.z4c_metric import Z4C_Metric  



def dthetadt(metric: Z4C_Metric, matter_terms):
    
    Arr = metric.Arr
    At = metric.At
    alpha = metric.alpha
    beta = metric.beta
    chi = metric.chi
    grr = metric.conformal_grr
    gt = metric.conformal_gt
    Kh = metric.Kh
    chi = metric.chi
    kappa = metric.kappa
    theta = metric.theta
    Gamma = metric.Gamma
    nu    = metric.nu
    # unpack the metric and matter terms

    dchidr = first_derivative(chi, metric.dr)
    dgrrdr = first_derivative(grr, metric.dr)
    dgtdr = first_derivative(gt, metric.dr)
    dGammadr = first_derivative(Gamma, metric.dr)
    dthetadr = first_derivative(theta, metric.dr)
    d2chdr2 = second_derivative(chi, metric.dr)
    # compute derivatives of the metric functions using finite difference methods

    rho = matter_terms.rho
    # unpack the matter terms

    # dThetadt = ((At )^2 alpha )/(gt )^2 + (2 Arr * At * alpha )/(
    #    grr * gt ) - 2 \[Kappa]1 * alpha * theta  + 
    #    4/3 alpha * (theta)^2 + 4/3 alpha * theta* Kh + 1/3 alpha * Kh^2 - 
    #    8 \[Pi] alpha *rho *-((2 alpha * chi  )/(r^2 grr )) - (
    #    2 grr * alpha * chi  )/(r^2 (gt )^2) + (4 alpha * chi  )/(
    #    r^2 gt ) + (grr * alpha * Gamma * chi  )/(r gt ) + (
    #    alpha * chi *  dgrrdr )/(r (grr )^2) - (3 alpha * chi *  dgrrdr )/(
    #    2 r grr * gt ) + (alpha * Gamma * chi *  dgrrdr )/(2 grr ) + (
    #    2 alpha * chi *  dgtdr )/(r * (gt )^2) - (
    #    3 alpha * chi *  dgtdr )/(r * grr * gt ) + (
    #    alpha * chi *  dgrrdr * dgtdr )/(2 ((grr )^2) * gt ) - (
    #    3 alpha * chi *  (dgtdr)^2)/(4 grr * (gt)^2) + beta * dthetadr + 
    #    1/2 alpha * chi *  dGammadr + (2 alpha * dchidr)/(r grr) - (
    #    alpha * dgrrdr * dchidr )/(2 (grr )^2) + (alpha * dgtdr * dchidr)/(
    #    grr * gt ) - (5 alpha * (dchidr )^2)/(4 grr * chi  ) + (
    #    alpha * d2chidr2)/grr;
    #   original equation from mathmatica notebook

    dthetadt = (At ** 2 * alpha) / (gt ** 2 )
    dthetadt += (2 * Arr * At * alpha) / (grr * gt)
    dthetadt += -2 * kappa * alpha * theta
    dthetadt += (4 / 3) * alpha * (theta ** 2)
    dthetadt += (4 / 3) * alpha * theta * Kh
    dthetadt += (1 / 3) * alpha * (Kh ** 2)
    dthetadt += -8 * jnp.pi * alpha * rho
    dthetadt += -((2 * alpha * chi) / (metric.r ** 2 * grr))
    dthetadt += -(2 * grr * alpha * chi) / (metric.r ** 2 * (gt ** 2))
    dthetadt += (4 * alpha * chi) / (metric.r ** 2 * gt)
    dthetadt += (grr * alpha * Gamma * chi) / (metric.r * gt)
    dthetadt += (alpha * chi * dgrrdr) / (metric.r * (grr ** 2))
    dthetadt += -(3 * alpha * chi * dgrrdr) / (2 * metric.r * grr * gt)
    dthetadt += (alpha * Gamma * chi * dgrrdr) / (2 * grr)
    dthetadt += (2 * alpha * chi * dgtdr) / (metric.r * (gt ** 2))
    dthetadt += -(3 * alpha * chi * dgtdr) / (metric.r * grr * gt)
    dthetadt += (alpha * chi * dgrrdr * dgtdr) / (2 * (grr ** 2) * gt)
    dthetadt += -(3 * alpha * chi * (dgtdr ** 2)) / (4 * grr * (gt ** 2))
    dthetadt += beta * dthetadr
    dthetadt += (1 / 2) * alpha * chi * dGammadr
    dthetadt += (2 * alpha * dchidr) / (metric.r * grr)
    dthetadt += -(alpha * dgrrdr * dchidr) / (2 * (grr ** 2))
    dthetadt += (alpha * dgtdr * dchidr) / (grr * gt)
    dthetadt += -(5 * alpha * (dchidr ** 2)) / (4 * grr * chi)
    dthetadt += (alpha * d2chdr2) / grr
    # compute the time derivative of theta using the Z4C evolution equations

    dthetadt += nu / 64 * (sixth_derivative(theta, metric.dr)) * (metric.dr ** 5)
    # add the Kreiss-Oliger dissipation term to the time derivative of theta

    return dthetadt


def dGammadt(metric: Z4C_Metric, matter_terms):
    Arr = metric.Arr
    At = metric.At
    alpha = metric.alpha
    beta = metric.beta
    chi = metric.chi
    grr = metric.conformal_grr
    gt = metric.conformal_gt
    Kh = metric.Kh
    chi = metric.chi
    kappa = metric.kappa
    theta = metric.theta
    Gamma = metric.Gamma
    nu    = metric.nu
    # unpack the metric and matter terms

    dchidr = first_derivative(chi, metric.dr)
    dgrrdr = first_derivative(grr, metric.dr)
    dgtdr = first_derivative(gt, metric.dr)
    dGammadr = first_derivative(Gamma, metric.dr)
    dalphadr = first_derivative(alpha, metric.dr)
    dbetadr = first_derivative(beta, metric.dr)
    dthetadr = first_derivative(theta, metric.dr)
    dKhdr = first_derivative(Kh, metric.dr)
    d2betadr2 = second_derivative(beta, metric.dr)
    # compute derivatives of the metric functions using finite difference methods

    # dGammadt = -((2 \[Kappa]1 * alpha )/(r grr )) + (4 At * alpha )/(
    #    r (gt )^2) + (2 \[Kappa]1 alpha )/(r * gt ) - (4 At * alpha )/(
    #    r grr * gt ) - (10 beta )/(3 (r^2) grr ) + (2 beta )/(
    #    3 (r^2) gt ) - \[Kappa]1 alpha * Gamma - (16 \[Pi] Sr* alpha )/
    #    chi  + (Arr * alpha * dgrrdr )/(grr )^3 + (\[Kappa]1 alpha * 
    #     dgrrdr )/(grr )^2 + (4 beta * dgrrdr )/(3 r (grr )^2) - (
    #    2 At * alpha * dgtdr )/(grr * (gt )^2) - (
    #    2 Arr * dalphadr)/(grr )^2 - (2 alpha * dthetadr)/(3 grr ) - (
    #    4 alpha * dKhdr)/(3 grr ) + (4 dbetadr )/(3 r grr ) + (
    #    4 dbetadr )/(3 r gt ) - (dgrrdr * dbetadr )/(3 (grr )^2) + 
    #    beta * dGammadr - (3 Arr * alpha * dchidr )/((grr )^2 chi ) + (
    #    4 d2betadr2)/(3 grr );
    #  original equation from mathmatica notebook

    dGammadt = -((2 * kappa * alpha) / (metric.r * grr))
    dGammadt += (4 * At * alpha) / (metric.r * (gt ** 2))
    dGammadt += (2 * kappa * alpha) / (metric.r * gt)
    dGammadt += -(4 * At * alpha) / (metric.r * grr * gt)
    dGammadt += -(10 * beta) / (3 * (metric.r ** 2) * grr)
    dGammadt += (2 * beta) / (3 * (metric.r ** 2) * gt)
    dGammadt += -kappa * alpha * Gamma
    dGammadt += -(16 * jnp.pi * matter_terms.Sr * alpha) / chi
    dGammadt += (Arr * alpha * dgrrdr) / (grr ** 3)
    dGammadt += (kappa * alpha * dgrrdr) / (grr ** 2)
    dGammadt += (4 * beta * dgrrdr) / (3 * metric.r * (grr ** 2))
    dGammadt += -(2 * At * alpha * dgtdr) / (grr * (gt ** 2))
    dGammadt += -(2 * Arr * alpha * dalphadr) / (grr ** 2)
    dGammadt += -(2 * alpha * dthetadr) / (3 * grr)
    dGammadt += -(4 * alpha * dKhdr) / (3 * grr)
    dGammadt += (4 * dbetadr) / (3 * metric.r * grr)
    dGammadt += (4 * dbetadr) / (3 * metric.r * gt)
    dGammadt += -(dgrrdr * dbetadr) / (3 * (grr ** 2))
    dGammadt += beta * dGammadr
    dGammadt += -(3 * Arr * alpha * dchidr) / ((grr ** 2) * chi)
    dGammadt += (4 * d2betadr2) / (3 * grr)
    # compute the time derivative of Gamma using the Z4C evolution equations

    dGammadt += nu / 64 * (sixth_derivative(Gamma, metric.dr)) * (metric.dr ** 5)
    # add the Kreiss-Oliger dissipation term to the time derivative of Gamma

    return dGammadt


def dZdt(metric: Z4C_Metric, matter_terms):
    Arr = metric.Arr
    At = metric.At
    alpha = metric.alpha
    beta = metric.beta
    chi = metric.chi
    grr = metric.conformal_grr
    gt = metric.conformal_gt
    Kh = metric.Kh
    chi = metric.chi
    kappa = metric.kappa
    theta = metric.theta
    Zr = metric.Zr
    nu = metric.nu
    # unpack the metric and matter terms

    dchidr = first_derivative(chi, metric.dr)
    dgtdr = first_derivative(gt, metric.dr)
    dAtdr = first_derivative(At, metric.dr)
    dthetadr = first_derivative(theta, metric.dr)
    dKhdr = first_derivative(Kh, metric.dr)
    dZrdr = first_derivative(Zr, metric.dr)
    # compute derivatives of the metric functions using finite difference methods


    #     dZdt = (2 Arr * alpha )/(r * grr ) - (2 At * alpha )/(
    #    r * gt ) - \[Kappa]1 * alpha *Zr - (8 \[Pi] grr * Sr * alpha )/
    #    chi   - (2 alpha *dAtdr)/gt  + (At * alpha * dgtdr )/(gt )^2 + (
    #    Arr * alpha * dgtdr )/(grr * gt ) - 1/3 alpha * dthetadr - 
    #    2/3 alpha * dKhdr + beta * dZrdr - (Arr * alpha * dchidr )/(
    #    grr * chi ) + (At * alpha * dchidr )/(gt * chi );
    #   # original equation from mathmatica notebook

    dZdt = (2 * Arr * alpha) / (metric.r * grr)
    dZdt += -(2 * At * alpha) / (metric.r * gt)
    dZdt += -kappa * alpha * Zr
    dZdt += -(8 * jnp.pi * grr * matter_terms.Sr * alpha) / chi
    dZdt += -(2 * alpha * dAtdr) / gt
    dZdt += (At * alpha * dgtdr) / (gt ** 2)
    dZdt += (Arr * alpha * dgtdr) / (grr * gt)
    dZdt += -(1 / 3) * alpha * dthetadr
    dZdt += -(2 / 3) * alpha * dKhdr
    dZdt += beta * dZrdr
    dZdt += -(Arr * alpha * dchidr) / (grr * chi)
    dZdt += (At * alpha * dchidr) / (gt * chi)
    # compute the time derivative of Zr using the Z4C evolution equations

    dZdt += nu / 64 * (sixth_derivative(Zr, metric.dr)) * (metric.dr ** 5)
    # add the Kreiss-Oliger dissipation term to the time derivative of Zr

    return dZdt