import jax.numpy as jnp
from typing import NamedTuple

class RadialGrid(NamedTuple):
    """Uniform radial grid on physical cells only.

    The `r_full` name is kept so the existing notebooks can still refer to the
    main physical grid without any ghost-cell padding.
    """

    r_full: jnp.ndarray
    r_interior: jnp.ndarray
    dr: float
    epsilon: float
    r_max: float


class FieldState(NamedTuple):
    """Grid-based fields needed by the particle evolution."""

    rho: jnp.ndarray
    A: jnp.ndarray
    lapse: jnp.ndarray
    shift: jnp.ndarray
    extrinsic_curvature: jnp.ndarray
    S_r: jnp.ndarray
    S_rr: jnp.ndarray
