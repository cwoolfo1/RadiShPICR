from typing import NamedTuple

import jax.numpy as jnp


class RadialGrid(NamedTuple):
    """Uniform physical radial grid for the spherical relativity solve."""

    r_full: jnp.ndarray
    r_interior: jnp.ndarray
    dr: float
    epsilon: float
    r_max: float


def build_radial_grid(epsilon: float, r_max: float, num_interior_points: int) -> RadialGrid:
    """Build the radial grid used by the field and particle equations."""

    if num_interior_points < 2:
        raise ValueError("num_interior_points must be at least 2")

    epsilon_value = float(epsilon)
    r_max_value = float(r_max)

    if not (epsilon_value >= 0.0):
        raise ValueError(
            "epsilon must be nonnegative: it is used as the positive radius "
            "floor in divisions at the regular center."
        )
    if not (r_max_value > 0.0):
        raise ValueError("r_max must be strictly positive")
    if not (r_max_value > epsilon_value):
        raise ValueError("r_max must be strictly greater than epsilon")

    dr_value = r_max_value / float(num_interior_points - 1)
    radial_coordinates = jnp.linspace(0.0, r_max_value, int(num_interior_points))

    if epsilon_value == 0.0:
        epsilon_value = 0.5 * dr_value
        # The grid includes r = 0, but equations with 1/r need a positive floor.

    return RadialGrid(
        r_full=radial_coordinates,
        r_interior=radial_coordinates,
        dr=dr_value,
        epsilon=epsilon_value,
        r_max=r_max_value,
    )
