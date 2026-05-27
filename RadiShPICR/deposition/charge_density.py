from functools import partial

import jax

from RadiShPICR.deposition.number_density import compute_number_density


@partial(jax.jit, static_argnames=("shape_mode",))
def compute_charge_density(particles, A, grid, shape_mode="nearest"):
    """Charge density from the shared relativistic number-density deposit."""

    number_density = compute_number_density(
        particles,
        A,
        grid,
        shape_mode=shape_mode,
    )
    return particles.get_charge() * number_density
