import jax
import jax.numpy as jnp

from RadiShPICR.relativity.utils import reverse_cumulative_trapezoid, safe_radius


@jax.jit
def compute_shift(lapse, extrinsic_curvature, grid, exact_exterior_points=None):
    """Compute the radial shift from the extrinsic-curvature integral."""

    safe_r = safe_radius(grid.r_full, grid.epsilon)

    # In polar slicing, the shift equation integrates alpha K^r_r / r from
    # each radius to the exterior vacuum where the shift is fixed to zero.
    integrand = jnp.zeros_like(lapse)
    integrand = integrand.at[1:-1].set(
        lapse[1:-1] * extrinsic_curvature[1:-1] / safe_r[1:-1]
    )

    integral_to_outer = reverse_cumulative_trapezoid(integrand, grid.dr)
    shift = -safe_r * integral_to_outer

    if exact_exterior_points is not None:
        shift = jnp.where(exact_exterior_points, 0.0, shift)

    shift = shift.at[-1].set(0.0)
    shift = shift.at[0].set(0.0)
    return shift
