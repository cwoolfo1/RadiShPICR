from __future__ import annotations

import jax.numpy as jnp

from RadiShPICR.deposition import compute_charge_density


def build_radial_gauss_law_operator(A, grid):
    """Build the dense finite-difference operator for radial Gauss law.

    The continuum equation solved here is

        dE_r/dr + (2/r + d ln(A)/dr) E_r = rho / epsilon_0.

    The unknown vector is ``E_r[1:]``.  The center value is fixed to
    ``E_r[0] = 0`` by spherical symmetry, the interior rows use centered
    differences, and the outer row uses a backward derivative so the outer
    boundary value is solved rather than pinned.
    """

    A = jnp.asarray(A)
    radial_grid = jnp.asarray(grid.r_full)
    num_unknowns = A.shape[0] - 1
    dr = grid.dr
    operator = jnp.zeros((num_unknowns, num_unknowns), dtype=A.dtype)

    # Interior rows: centered derivative for dE_r/dr and centered d ln(A)/dr.
    grid_indices = jnp.arange(1, A.shape[0] - 1)
    rows = grid_indices - 1
    dlnA_dr = (A[grid_indices + 1] - A[grid_indices - 1]) / (
        2.0 * dr * A[grid_indices]
    )
    operator = operator.at[rows, rows].set(2.0 / radial_grid[grid_indices] + dlnA_dr)
    operator = operator.at[rows[1:], rows[1:] - 1].set(-1.0 / (2.0 * dr))
    operator = operator.at[rows, rows + 1].set(1.0 / (2.0 * dr))

    # Outer row: one-sided derivative closes the solved field at r_max.
    outer_row = num_unknowns - 1
    outer_r = radial_grid[-1]
    outer_dlnA_dr = (A[-1] - A[-2]) / (dr * A[-1])
    operator = operator.at[outer_row, outer_row - 1].set(-1.0 / dr)
    operator = operator.at[outer_row, outer_row].set(
        1.0 / dr + 2.0 / outer_r + outer_dlnA_dr
    )

    return operator


def compute_radial_electric_field(charge_density, A, grid, epsilon_0=1.0):
    """Return the radial electric field implied by Gauss law.

    Parameters
    ----------
    charge_density : array_like
        Charge density ``rho`` deposited on ``grid.r_full``.
    A : array_like
        Radial metric factor on the same grid.
    grid : object
        Radial grid with ``r_full`` and uniform spacing ``dr``.
    epsilon_0 : float, optional
        Permittivity normalization in ``rho / epsilon_0``.

    Returns
    -------
    jax.numpy.ndarray
        Radial electric field ``E_r`` on ``grid.r_full``.
    """

    charge_density = jnp.asarray(charge_density)
    operator = build_radial_gauss_law_operator(A, grid)
    rhs = charge_density[1:] / float(epsilon_0)

    interior_electric_field = jnp.linalg.solve(operator, rhs)

    electric_field = jnp.zeros_like(grid.r_full, dtype=interior_electric_field.dtype)
    electric_field = electric_field.at[1:].set(interior_electric_field)
    return electric_field


def compute_charge_density_and_radial_electric_field(
    particle_list,
    A,
    grid,
    epsilon_0=1.0,
    shape_mode="nearest",
):
    """Deposit charge from particle species and solve radial Gauss law.

    Each particle object is deposited with the existing metric-aware charge
    density routine.  The species contributions are summed on the radial grid,
    then the total charge density is passed to ``compute_radial_electric_field``.

    Returns
    -------
    charge_density, electric_field : tuple[jax.numpy.ndarray, jax.numpy.ndarray]
        The deposited total charge density and the corresponding radial field.
    """

    total_charge_density = jnp.zeros_like(grid.r_full)

    # Sum species explicitly so multi-species electrostatic sources remain
    # visible at the call site and use the same deposition path as relativity.
    for particles in particle_list:
        species_charge_density = compute_charge_density(
            particles,
            A,
            grid,
            shape_mode=shape_mode,
        )
        total_charge_density = total_charge_density + species_charge_density

    electric_field = compute_radial_electric_field(
        total_charge_density,
        A,
        grid,
        epsilon_0=epsilon_0,
    )
    return total_charge_density, electric_field


def compute_dEdr(Er, rho, dlnA_dr, grid, epsilon_0=1.0):
    """Compute the radial derivative of the electric field from Gauss law.

    This is a direct evaluation of the continuum equation

        dE_r/dr = rho / epsilon_0 - (2/r + d ln(A)/dr) E_r.

    """

    # dE_r_dr + 2/r E_r + d ln(A)/dr E_r = rho / epsilon_0

    dEdr = rho / float(epsilon_0) - (2.0 / grid.r_full + dlnA_dr) * Er
    return dEdr
