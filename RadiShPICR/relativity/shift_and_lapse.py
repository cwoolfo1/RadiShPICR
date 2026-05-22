import jax
import jax.numpy as jnp

from src.utils import safe_radius, compute_metric_radial_derivative, reverse_cumulative_hermite, reverse_cumulative_trapezoid
from src.schwarzschild import schwarzschild_A, schwarzschild_lapse

@jax.jit
def compute_polar_lapse_denominator(
    A,
    schwarzschild_mass,
    grid,
    exact_exterior_points = None,
):
    """Compute ``1 + r d_r ln A`` for the polar-slicing lapse denominator."""

    safe_r = safe_radius(grid.r_full, grid.epsilon)
    # ensure that the radius is never too small to avoid numerical issues with the logarithmic derivative

    dA_dr = compute_metric_radial_derivative(
        A,
        schwarzschild_mass,
        grid,
        exact_exterior_points=exact_exterior_points,
    )
    # compute the derivative of A

    dln_A = jnp.zeros_like(A)
    dln_A = dln_A.at[1:-1].set(dA_dr[1:-1] / A[1:-1])
    # define the logarithmic derivative of A, which is used in the lapse denominator

    denominator = jnp.ones_like(A)
    denominator = denominator.at[1:-1].set(1.0 + safe_r[1:-1] * dln_A[1:-1])
    # define the denominator of the polar lapse integrand
    return denominator


@jax.jit
def compute_lapse(A, S_rr, schwarzschild_mass, grid, exact_exterior_points = None):
    """Compute the polar-slicing lapse from the radial integral formula."""

    safe_r = safe_radius(grid.r_full, grid.epsilon)
    # ensure that the radius is never too small to avoid numerical issues with the lapse formula

    dA_dr = compute_metric_radial_derivative(
        A,
        schwarzschild_mass,
        grid,
        exact_exterior_points=exact_exterior_points,
    )
    # compute the derivative of A

    dln_A = jnp.zeros_like(A)
    dln_A = dln_A.at[1:-1].set(dA_dr[1:-1] / A[1:-1])
    # define the logarithmic derivative of A, which is used in the lapse formula

    denominator = jnp.ones_like(A)
    denominator = denominator.at[1:-1].set(1.0 + safe_r[1:-1] * dln_A[1:-1])
    # define the denominator of the integrand in the lapse formula directly inside the production lapse solve

    integrand = jnp.zeros_like(A)

    interior_integrand = (
        safe_r[1:-1] * dln_A[1:-1] ** 2 + 8.0 * jnp.pi * safe_r[1:-1] * S_rr[1:-1]
    ) / denominator[1:-1]
    integrand = integrand.at[1:-1].set(interior_integrand)
    # define the integrand in the lapse formula

    integral_to_outer = reverse_cumulative_hermite(integrand, grid.dr)
    # integrate the integrand from each grid point to the outer boundary using fourth-order Hermite quadrature

    r_max = grid.r_full[-1]
    outer_A = schwarzschild_A( r_max, schwarzschild_mass, grid.epsilon)
    # compute the outer A using the Vaccuum solution

    outer_lapse = schwarzschild_lapse( r_max, schwarzschild_mass, grid.epsilon)
    # compute the outer lapse using the Vaccuum solution

    outer_lapse_prefactor = outer_A * outer_lapse
    lapse = outer_lapse_prefactor * jnp.exp(-0.5 * integral_to_outer) / A
    # define the lapse throughout the domain

    if exact_exterior_points is not None:
        vaccuum_lapse = schwarzschild_lapse(grid.r_full, schwarzschild_mass, grid.epsilon)
        lapse = jnp.where(exact_exterior_points, vaccuum_lapse, lapse)
    # if the user specifies which points are in the exterior vacuum, set those points to the Schwarzschild lapse.

    lapse = lapse.at[-1].set(outer_lapse)
    # set the boundary lapse to the vaccuum solution
    lapse = lapse.at[0].set(lapse[1])
    # ensure regularity by setting the lapse at the center to be the same as the lapse at the first grid point

    return lapse


@jax.jit
def compute_shift( lapse, extrinsic_curvature, grid, exact_exterior_points = None):
    """Compute the shift by integrating the curvature from r to the outer edge."""

    safe_r = safe_radius(grid.r_full, grid.epsilon)
    # ensure the radius is never completely zero to avoid numerical issues

    integrand = jnp.zeros_like(lapse)
    integrand = integrand.at[1:-1].set(lapse[1:-1] * extrinsic_curvature[1:-1] / safe_r[1:-1])
    # using definition of curvature from BSSN book in polar slicing gauge

    integral_to_outer = reverse_cumulative_trapezoid(integrand, grid.dr)
    # integrate shift using standard trapezoid
    shift = -safe_r * integral_to_outer
    # define shift vector

    if exact_exterior_points is not None:
        shift = jnp.where(exact_exterior_points, 0.0, shift)
    # if the user has specified vaccuum points, then use them

    # The boundary cells are vacuum, so the shift is fixed to zero there.
    shift = shift.at[-1].set(0.0)
    shift = shift.at[0].set(0.0)

    return shift
