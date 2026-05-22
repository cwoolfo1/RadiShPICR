import jax.numpy as jnp

from src.utils import safe_radius
from src.states import FieldState


def schwarzschild_u(radius, schwarzschild_mass, epsilon):
    """Isotropic Schwarzschild U = sqrt(A) used for the A boundary values."""

    safe_r = safe_radius(radius, epsilon)
    mass_value = jnp.asarray(schwarzschild_mass, dtype=safe_r.dtype)
    return 1.0 + mass_value / (2.0 * safe_r)


def schwarzschild_A(radius, schwarzschild_mass, epsilon):
    """Isotropic Schwarzschild A = U^2 used as a readable helper."""

    U = schwarzschild_u(radius, schwarzschild_mass, epsilon)
    return U**2


def schwarzschild_dA_dr(radius, schwarzschild_mass, epsilon):
    """Vaccuum Schwarzschild derivative for A.

    This uses the boundary formula requested by the user:
    dA/dr = 2 * (1 + M / (2 r)) * (-M / (2 r^2)).
    """

    safe_r = safe_radius(radius, epsilon)
    mass_value = jnp.asarray(schwarzschild_mass, dtype=safe_r.dtype)
    return 2.0 * (1.0 + mass_value / (2.0 * safe_r)) * (-mass_value / (2.0 * safe_r**2))

def schwarzschild_lapse(radius, schwarzschild_mass, epsilon):
    """Schwarzschild lapse used on the outer boundary cell."""

    safe_r = safe_radius(radius, epsilon)
    mass_value = jnp.asarray(schwarzschild_mass, dtype=safe_r.dtype)
    numerator = 1.0 - mass_value / (2.0 * safe_r)
    denominator = 1.0 + mass_value / (2.0 * safe_r)
    return numerator / denominator


def schwarzschild_d_lapse_dr(
    radius, schwarzschild_mass, epsilon):
    """Analytic isotropic-radial derivative of the Schwarzschild lapse."""

    safe_r = safe_radius(radius, epsilon)
    # ensure the radial positions are safe for division
    mass_value = jnp.asarray(schwarzschild_mass, dtype=safe_r.dtype)
    denominator = safe_r**2 * (1.0 + mass_value / (2.0 * safe_r)) ** 2
    return mass_value / denominator


def isotropic_to_areal_radius(
    isotropic_radius: jnp.ndarray,
    schwarzschild_mass: float,
    epsilon: float,
) -> jnp.ndarray:
    """Convert isotropic radius to the Schwarzschild areal radius."""

    safe_r = safe_radius(isotropic_radius, epsilon)
    # keep the isotropic radius away from zero before applying the exterior formula
    return safe_r * schwarzschild_A(safe_r, schwarzschild_mass, epsilon)
    # use the exact isotropic Schwarzschild relation r_s = r * A


def areal_to_isotropic_radius(
    areal_radius: jnp.ndarray,
    schwarzschild_mass: float,
) -> jnp.ndarray:
    """Convert areal radius to the exterior isotropic Schwarzschild branch."""

    mass_value = jnp.asarray(schwarzschild_mass, dtype=areal_radius.dtype)
    # convert the Schwarzschild mass to the same dtype as the areal radius samples
    discriminant = areal_radius * (areal_radius - 2.0 * mass_value)
    # build the exterior quadratic discriminant for the isotropic-radius inversion
    return 0.5 * (areal_radius - mass_value + jnp.sqrt(discriminant))
    # return the exterior isotropic root that matches the paper's Schwarzschild coordinates


def build_static_schwarzschild_fields(
    grid,
    schwarzschild_mass: float,
) -> FieldState:
    """Build the exact static Schwarzschild fields on the grid."""

    A = schwarzschild_A(grid.r_full, schwarzschild_mass, grid.epsilon)
    # evaluate the exact isotropic Schwarzschild conformal factor on the whole grid
    lapse = schwarzschild_lapse(grid.r_full, schwarzschild_mass, grid.epsilon)
    # evaluate the exact polar-slicing lapse on the whole grid
    zeros = jnp.zeros_like(grid.r_full)
    # matter sources, shift, and curvature all vanish in the static exterior spacetime
    return FieldState(
        rho=zeros,
        A=A,
        lapse=lapse,
        shift=zeros,
        extrinsic_curvature=zeros,
        S_r=zeros,
        S_rr=zeros,
    )
    # return the full field state used by the fixed-background validation notebooks


def schwarzschild_mass_from_metric(
    A: jnp.ndarray,
    dA_dr: jnp.ndarray,
    grid,
) -> jnp.ndarray:
    """Recover the exterior Schwarzschild mass from the polar metric data."""

    safe_r = safe_radius(grid.r_full, grid.epsilon)
    # keep the isotropic radius positive before evaluating the mass formula
    areal_radius = A * safe_r
    # compute the Schwarzschild areal radius r_s = A r on the isotropic grid
    return 0.5 * areal_radius * (1.0 - (1.0 + safe_r * dA_dr / A) ** 2)
    # apply the exact polar-gauge vacuum mass relation used in the old diagnostics
