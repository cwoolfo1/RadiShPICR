from typing import NamedTuple
import jax.numpy as jnp

class Z4C_Metric(NamedTuple):
    alpha: jnp.ndarray
    beta: jnp.ndarray
    # lapse and shift
    conformal_grr: jnp.ndarray
    conformal_gt: jnp.ndarray
    # conformal metric
    chi: jnp.ndarray
    # conformal factor
    Kh: jnp.ndarray
    Arr: jnp.ndarray
    At : jnp.ndarray
    # trace and traceless part of extrinsic curvature
    theta: jnp.ndarray
    Zr: jnp.ndarray
    Gamma: jnp.ndarray
    # constraint terms
    kappa: jnp.ndarray
    eta: jnp.ndarray
    nu: jnp.ndarray
    # damping parameters
    r: jnp.ndarray
    dr: jnp.ndarray
    # radial coordinate