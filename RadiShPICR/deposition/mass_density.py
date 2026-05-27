from functools import partial

import jax

from RadiShPICR.deposition.number_density import (
    compute_number_density
)


@partial(jax.jit, static_argnames=("shape_mode",))
def compute_mass_density(particles, A, grid, shape_mode="nearest"):
    """Deposited mass-energy density on the polar-slicing radial grid."""

    number_density = compute_number_density(
        particles,
        A,
        grid,
        shape_mode=shape_mode,
    )
    return particles.get_mass() * number_density
