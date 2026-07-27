from typing import NamedTuple

import jax.numpy as jnp


class RadialGrid(NamedTuple):
    """Uniform physical radial grid for the spherical relativity solve."""

    r_full: jnp.ndarray
    r_interior: jnp.ndarray
    dr: float
    r_max: float


def build_radial_grid(r_max: float, num_interior_points: int) -> RadialGrid:
    """Build the radial grid used by the field and particle equations."""

    if num_interior_points < 2:
        raise ValueError("num_interior_points must be at least 2")

    r_max_value = float(r_max)

    if not (r_max_value > 0.0):
        raise ValueError("r_max must be strictly positive")

    dr_value = r_max_value / float(num_interior_points - 1)
    radial_coordinates = jnp.linspace(0.0, r_max_value, int(num_interior_points))

    return RadialGrid(
        r_full=radial_coordinates,
        r_interior=radial_coordinates,
        dr=dr_value,
        r_max=r_max_value,
    )
