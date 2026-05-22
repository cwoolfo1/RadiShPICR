import jax
import jax.numpy as jnp


@jax.jit
def safe_radius(radius, epsilon):
    """Keep radial denominators away from the origin."""

    epsilon_value = jnp.asarray(epsilon, dtype=radius.dtype)
    # convert to array to avoid issues with broadcasting when radius is an array
    return jnp.maximum(radius, epsilon_value)

@jax.jit
def safe_metric_A(A):
    """Keep trial metric values positive during the density reconstruction."""

    metric_floor = jnp.asarray(1e-12, dtype=A.dtype)
    return jnp.maximum(A, metric_floor)

@jax.jit
def centered_first_derivative(field, dr):
    """Second-order first derivative on the physical grid.

    Interior points use centered differences. The two boundaries use the usual
    second-order one-sided stencil because there are no ghost cells anymore.
    """

    derivative = jnp.zeros_like(field)
    interior_derivative = (field[2:] - field[:-2]) / (2.0 * dr)
    derivative = derivative.at[1:-1].set(interior_derivative)
    derivative = derivative.at[0].set((-3.0 * field[0] + 4.0 * field[1] - field[2]) / (2.0 * dr))
    derivative = derivative.at[-1].set((3.0 * field[-1] - 4.0 * field[-2] + field[-3]) / (2.0 * dr))
    return derivative

@jax.jit
def centered_second_derivative(field, dr):
    """Second-order second derivative on the physical grid."""

    derivative = jnp.zeros_like(field)
    interior_derivative = (field[2:] - 2.0 * field[1:-1] + field[:-2]) / (dr**2)
    derivative = derivative.at[1:-1].set(interior_derivative)

    if field.shape[0] >= 4:
        derivative = derivative.at[0].set(
            (2.0 * field[0] - 5.0 * field[1] + 4.0 * field[2] - field[3]) / (dr**2)
        )
        derivative = derivative.at[-1].set(
            (2.0 * field[-1] - 5.0 * field[-2] + 4.0 * field[-3] - field[-4]) / (dr**2)
        )

    return derivative


@jax.jit
def compute_metric_radial_derivative(
    A,
    schwarzschild_mass,
    grid,
    exact_exterior_points = None,
):
    """Compute ``dA/dr`` with the regular-center and exterior Schwarzschild BCs."""

    from RadiShPICR.relativity.schwarzschild import schwarzschild_dA_dr

    dA_dr = centered_first_derivative(A, grid.dr)
    # start with a centered finite difference

    exact_dA_dr = schwarzschild_dA_dr(grid.r_full, schwarzschild_mass, grid.epsilon)
    # compute the dA/dr for the Schwarzschild metric, because it should be Schwarzschild at the 
    # outer boundary because of Birkoff's theorem.

    dA_dr = dA_dr.at[0].set(0.0)
    # set the center to zero, which is the regularity condition for A at the origin

    if exact_exterior_points is None:
        dA_dr = dA_dr.at[-1].set(exact_dA_dr[-1])
        return dA_dr
    # if the user does not specify which points in the grid are vaccuum, default to just setting
    # the outer boundary point to the Schwarzschild value/

    dA_dr = jnp.where(exact_exterior_points, exact_dA_dr, dA_dr)
    # if the user specifies which points are in the exterior vacuum, set those points to the Schwarzschild value. 

    return dA_dr


@jax.jit
def last_matter_support_index(radial_positions, grid):
    """Return the outermost grid point that carries deposited matter."""

    deposition_index = nearest_interior_index(radial_positions, grid)
    # find the nearest deposition point for every particle on the physical grid
    return jnp.max(deposition_index)
    # return the largest occupied deposition index, which defines the discrete matter surface


@jax.jit
def exact_exterior_points_from_last_matter_index(last_support_index, grid):
    """Mark the exact Schwarzschild vacuum exterior points used by the field solve."""

    grid_index = jnp.arange(grid.r_full.shape[0], dtype=last_support_index.dtype)
    # build the grid-point index array with the same integer type as the support index
    exact_exterior_points = grid_index > last_support_index
    # mark every grid point outside the outermost occupied matter point as exact vacuum
    exact_exterior_points = exact_exterior_points.at[-1].set(True)
    # always pin the outer boundary point to the exact Schwarzschild exterior
    return exact_exterior_points
    # return the boolean exterior-point array used by the notebook diagnostics and the field routines


def reverse_cumulative_trapezoid(values: jnp.ndarray, dr: float) -> jnp.ndarray:
    """Integrate from each grid point to the outer edge on a uniform grid."""

    trapezoid_segments = 0.5 * dr * (values[:-1] + values[1:])
    reverse_integral = jnp.cumsum(trapezoid_segments[::-1])[::-1]
    integral = jnp.zeros_like(values)
    return integral.at[:-1].set(reverse_integral)


def reverse_cumulative_hermite(values, dr):
    """Integrate from each grid point to the outer edge using cubic Hermite quadrature.

    Fourth-order accurate replacement for reverse_cumulative_trapezoid.
    Uses the closed-form integral of a cubic Hermite interpolant on each
    sub-interval:  h*(y0+y1)/2 + h^2*(m0-m1)/12, where slopes are
    estimated via second-order finite differences.
    """
    slopes = jnp.zeros_like(values)
    slopes = slopes.at[1:-1].set((values[2:] - values[:-2]) / (2.0 * dr))
    slopes = slopes.at[0].set(
        (-3.0 * values[0] + 4.0 * values[1] - values[2]) / (2.0 * dr)
    )
    slopes = slopes.at[-1].set(
        (3.0 * values[-1] - 4.0 * values[-2] + values[-3]) / (2.0 * dr)
    )

    hermite_segments = (
        dr * (values[:-1] + values[1:]) / 2.0
        + dr**2 * (slopes[:-1] - slopes[1:]) / 12.0
    )

    reverse_integral = jnp.cumsum(hermite_segments[::-1])[::-1]
    integral = jnp.zeros_like(values)
    return integral.at[:-1].set(reverse_integral)

def nearest_interior_index(radial_positions, grid):
    """Map particles to the nearest interior grid point.

    The two edge cells are reserved as vacuum boundary cells, so matter is only
    deposited on indices `1` through `N-2`.
    """

    floating_index = (radial_positions - grid.r_full[0]) / grid.dr
    nearest = jnp.rint(floating_index).astype(jnp.int32)
    return jnp.clip(nearest, 1, grid.r_full.shape[0] - 2)

