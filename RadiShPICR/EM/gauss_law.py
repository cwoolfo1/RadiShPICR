from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

from RadiShPICR.deposition import compute_charge_density


def build_radial_gauss_law_operator(A, grid):
    """Build the sparse finite-difference operator for radial Gauss law.

    The continuum equation solved here is

        dE_r/dr + (2/r + d ln(A)/dr) E_r = rho / epsilon_0.

    The unknown vector is ``E_r[1:]``.  The center value is fixed to
    ``E_r[0] = 0`` by spherical symmetry, the interior rows use centered
    differences, and the outer row uses a backward derivative so the outer
    boundary value is solved rather than pinned.
    """

    A_array = np.asarray(A, dtype=float)
    radial_grid = np.asarray(grid.r_full, dtype=float)
    if A_array.shape != radial_grid.shape:
        raise ValueError(
            "A and grid.r_full must have the same shape for the radial Gauss-law stencil."
        )

    num_unknowns = A_array.size - 1
    dr = float(grid.dr)
    row_indices = []
    column_indices = []
    values = []

    # Interior rows: centered derivative for dE_r/dr and centered d ln(A)/dr.
    for row, grid_index in enumerate(range(1, A_array.size - 1)):
        r_i = radial_grid[grid_index]
        dlnA_dr = (A_array[grid_index + 1] - A_array[grid_index - 1]) / (
            2.0 * dr * A_array[grid_index]
        )

        row_indices.append(row)
        column_indices.append(row)
        values.append(2.0 / r_i + dlnA_dr)

        if grid_index - 1 >= 1:
            row_indices.append(row)
            column_indices.append(row - 1)
            values.append(-1.0 / (2.0 * dr))

        if grid_index + 1 <= A_array.size - 1:
            row_indices.append(row)
            column_indices.append(row + 1)
            values.append(1.0 / (2.0 * dr))

    # Outer row: one-sided derivative closes the solved field at r_max.
    outer_row = num_unknowns - 1
    outer_r = radial_grid[-1]
    outer_dlnA_dr = (A_array[-1] - A_array[-2]) / (dr * A_array[-1])
    row_indices.extend([outer_row, outer_row])
    column_indices.extend([outer_row - 1, outer_row])
    values.extend([
        -1.0 / dr,
        1.0 / dr + 2.0 / outer_r + outer_dlnA_dr,
    ])

    return sparse.csr_matrix(
        (values, (row_indices, column_indices)),
        shape=(num_unknowns, num_unknowns),
    )


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

    charge_density_array = np.asarray(charge_density, dtype=float)
    radial_grid = np.asarray(grid.r_full, dtype=float)
    if charge_density_array.shape != radial_grid.shape:
        raise ValueError(
            "charge_density and grid.r_full must have the same shape for Gauss law."
        )

    operator = build_radial_gauss_law_operator(A, grid)
    rhs = charge_density_array[1:] / float(epsilon_0)

    # Direct solves of the homogeneous system can return NaNs for the singular
    # zero-source case, while the physical Gauss-law field is exactly zero.
    if np.allclose(rhs, 0.0):
        interior_electric_field = np.zeros_like(rhs)
    else:
        interior_electric_field = sparse_linalg.spsolve(operator, rhs)
        interior_electric_field = np.asarray(interior_electric_field, dtype=float)

    electric_field = np.zeros_like(radial_grid)
    electric_field[1:] = interior_electric_field
    return jnp.asarray(electric_field)


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
