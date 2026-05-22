import jax
import jax.numpy as jnp

from RadiShPICR.relativity.schwarzschild import schwarzschild_A, schwarzschild_lapse
from RadiShPICR.relativity.utils import (
    compute_metric_radial_derivative,
    reverse_cumulative_hermite,
    safe_radius,
)


@jax.jit
def compute_polar_lapse_denominator(
    A,
    schwarzschild_mass,
    grid,
    exact_exterior_points=None,
):
    """Compute the polar-slicing denominator ``1 + r d_r ln(A)``."""

    safe_r = safe_radius(grid.r_full, grid.epsilon)
    dA_dr = compute_metric_radial_derivative(
        A,
        schwarzschild_mass,
        grid,
        exact_exterior_points=exact_exterior_points,
    )

    dln_A_dr = jnp.zeros_like(A)
    dln_A_dr = dln_A_dr.at[1:-1].set(dA_dr[1:-1] / A[1:-1])

    denominator = jnp.ones_like(A)
    denominator = denominator.at[1:-1].set(1.0 + safe_r[1:-1] * dln_A_dr[1:-1])
    return denominator


@jax.jit
def compute_lapse(A, S_rr, schwarzschild_mass, grid, exact_exterior_points=None):
    """Compute the polar-slicing lapse from the radial integral equation."""

    safe_r = safe_radius(grid.r_full, grid.epsilon)
    dA_dr = compute_metric_radial_derivative(
        A,
        schwarzschild_mass,
        grid,
        exact_exterior_points=exact_exterior_points,
    )

    dln_A_dr = jnp.zeros_like(A)
    dln_A_dr = dln_A_dr.at[1:-1].set(dA_dr[1:-1] / A[1:-1])

    denominator = jnp.ones_like(A)
    denominator = denominator.at[1:-1].set(1.0 + safe_r[1:-1] * dln_A_dr[1:-1])

    # The lapse integrand contains geometric curvature from A plus the radial
    # stress source term. Boundary cells are filled after the integral.
    integrand = jnp.zeros_like(A)
    interior_integrand = (
        safe_r[1:-1] * dln_A_dr[1:-1] ** 2
        + 8.0 * jnp.pi * safe_r[1:-1] * S_rr[1:-1]
    ) / denominator[1:-1]
    integrand = integrand.at[1:-1].set(interior_integrand)

    # Integrate inward from the outer boundary because the exterior lapse is
    # known analytically from the Schwarzschild vacuum solution.
    integral_to_outer = reverse_cumulative_hermite(integrand, grid.dr)
    outer_radius = grid.r_full[-1]
    outer_A = schwarzschild_A(outer_radius, schwarzschild_mass, grid.epsilon)
    outer_lapse = schwarzschild_lapse(outer_radius, schwarzschild_mass, grid.epsilon)

    lapse = outer_A * outer_lapse * jnp.exp(-0.5 * integral_to_outer) / A

    if exact_exterior_points is not None:
        vacuum_lapse = schwarzschild_lapse(grid.r_full, schwarzschild_mass, grid.epsilon)
        lapse = jnp.where(exact_exterior_points, vacuum_lapse, lapse)

    lapse = lapse.at[-1].set(outer_lapse)
    lapse = lapse.at[0].set(lapse[1])
    return lapse
