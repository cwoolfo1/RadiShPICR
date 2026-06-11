import jax
import jax.numpy as jnp


@jax.jit
def safe_radius(radius, epsilon):
    """Keep radial denominators away from the origin."""

    epsilon_value = jnp.asarray(epsilon, dtype=radius.dtype)
    # convert to array to avoid issues with broadcasting when radius is an array
    return jnp.maximum(radius, epsilon_value)



def nearest_interior_index(radial_positions, grid):
    """Map particles to the nearest interior grid point.

    The two edge cells are reserved as vacuum boundary cells, so matter is only
    deposited on indices `1` through `N-2`.
    """

    floating_index = (radial_positions - grid.r_full[0]) / grid.dr
    nearest = jnp.rint(floating_index).astype(jnp.int32)
    return jnp.clip(nearest, 1, grid.r_full.shape[0] - 2)

