import jax.numpy as jnp

from RadiShPICR.relativity.utils import safe_radius


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


def build_static_schwarzschild_metric(
    grid,
    schwarzschild_mass: float,
):
    """Build the exact static Schwarzschild metric on the whole grid."""

    from RadiShPICR.relativity.metric import MetricState

    A = schwarzschild_A(grid.r_full, schwarzschild_mass, grid.epsilon)
    # evaluate the exact isotropic Schwarzschild conformal factor on the whole grid
    lapse = schwarzschild_lapse(grid.r_full, schwarzschild_mass, grid.epsilon)
    # evaluate the exact polar-slicing lapse on the whole grid
    zeros = jnp.zeros_like(grid.r_full)
    # matter sources, shift, and curvature all vanish in the static exterior spacetime
    exact_exterior_points = jnp.ones_like(grid.r_full, dtype=bool)
    # every grid point belongs to the analytic vacuum solution in this helper.
    return MetricState(
        rho=zeros,
        A=A,
        lapse=lapse,
        shift=zeros,
        extrinsic_curvature=zeros,
        S_r=zeros,
        S_rr=zeros,
        exact_exterior_points=exact_exterior_points,
    )
    # return the full metric state used by fixed-background geodesic tests.
