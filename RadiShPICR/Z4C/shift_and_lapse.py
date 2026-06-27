import jax.numpy as jnp
from RadiShPICR.Z4C.z4c_metric import Z4C_Metric
from RadiShPICR.Z4C.derivatives import first_derivative, second_derivative

def dalphadt(metric: Z4C_Metric, matter_terms):
    alpha = metric.alpha
    beta = metric.beta
    Kh = metric.Kh

    dalphadr = first_derivative(alpha, metric.dr)
    # compute derivatives of the metric functions using finite difference methods
 
    # dalphadt = -2 * alpha * Kh + beta * dalphadr;
    # original equation from mathmatica notebook

    dalphadt = -2 * alpha * Kh
    dalphadt += beta * dalphadr

    return dalphadt


def dbetadt(metric: Z4C_Metric, matter_terms):
    beta = metric.beta
    Gamma = metric.Gamma
    # unpack the metric and matter terms

    dbetadr = first_derivative(beta, metric.dr)
    # compute derivatives of the metric functions using finite difference methods

    # dbetadt = -\[Eta] beta +  5/2 * Gamma + beta * dbetadr;
    # original equation from mathmatica notebook

    dbetadt = -metric.eta * beta
    dbetadt += (5 / 2) * Gamma
    dbetadt += beta * dbetadr

    return dbetadt