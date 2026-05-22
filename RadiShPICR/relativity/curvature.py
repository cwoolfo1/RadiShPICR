import jax
import jax.numpy as jnp

from src.utils import safe_radius, compute_metric_radial_derivative


@jax.jit
def compute_extrinsic_curvature(
    A, S_r, schwarzschild_mass, grid, exact_exterior_points = None):
    """Compute ``K^r_r`` from the polar-slicing momentum constraint.

    Paper IV equation (8) gives

    ``K^r_r = -4 pi r t_r / (1 + r A_r / A)``.

    The code's deposited source ``S_r`` is ``-t_r`` in the paper's notation,
    so the implemented sign is positive:

    ``K^r_r = 4 pi r S_r / (1 + r d_r ln A)``.
    """

    safe_r = safe_radius(grid.r_full, grid.epsilon)
    # ensure the radius is never completely zero to avoid numerical issues
    dA_dr = compute_metric_radial_derivative(A, schwarzschild_mass, grid, exact_exterior_points=exact_exterior_points)
    # compute the derivative of A, which is needed for the denominator of the curvature formula

    dln_A = jnp.zeros_like(A)
    dln_A = dln_A.at[1:-1].set(dA_dr[1:-1] / A[1:-1])
    # define the logarithmic derivative of A, which is used in the denominator of the curvature formula

    denominator = jnp.ones_like(A)
    denominator = denominator.at[1:-1].set(1.0 + safe_r[1:-1] * dln_A[1:-1])
    # define the denominator of the curvature formula

    extrinsic_curvature = jnp.zeros_like(A)
    interior_curvature = 4.0 * jnp.pi * safe_r[1:-1] * S_r[1:-1] / denominator[1:-1]
    extrinsic_curvature = extrinsic_curvature.at[1:-1].set(interior_curvature)
    # define the curvature in the interior of the grid using the formula from the paper

    if exact_exterior_points is not None:
        extrinsic_curvature = jnp.where(exact_exterior_points, 0.0, extrinsic_curvature)
        # if the user specifies which points are in the exterior vacuum, set those points to zero curvature because of Birkhoff's theorem.

    extrinsic_curvature = extrinsic_curvature.at[0].set(0.0)
    extrinsic_curvature = extrinsic_curvature.at[-1].set(0.0)
    # set the boundary curvatures to 0
    
    return extrinsic_curvature
