from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from RadiShPICR.deposition import compute_charge_density


class RadialElectricFieldSolveResult(NamedTuple):
    """Result from the radial relativistic Poisson solve."""

    electric_field: jnp.ndarray
    charge_density: jnp.ndarray
    operator: object
    residual_norm: float
    info: int
    iterations: int


def _require_scipy():
    try:
        from scipy import sparse
        from scipy.sparse import linalg as sparse_linalg
    except ImportError as exc:
        raise ImportError(
            "Radial electric-field sparse direct solve requires scipy. "
            "Install scipy to use RadiShPICR.EM.solve_radial_electric_field."
        ) from exc

    return sparse, sparse_linalg


def build_radial_poisson_operator(A, grid):
    """Build the interior sparse operator for the radial electric field.

    The unknown vector contains only ``E_r[1:-1]``. The first and last
    electric-field entries are fixed to zero, so centered derivative references
    to the boundary do not add right-hand-side terms.
    """

    sparse, _ = _require_scipy()

    A_array = np.asarray(A, dtype=float)
    if A_array.shape != tuple(grid.r_full.shape):
        raise ValueError(
            "A must have the same shape as grid.r_full: "
            f"expected {tuple(grid.r_full.shape)}, got {A_array.shape}."
        )
    if A_array.size < 3:
        raise ValueError("At least three radial grid points are required.")
    if not np.all(np.isfinite(A_array)):
        raise ValueError("A must contain only finite values.")
    if np.any(A_array == 0.0):
        raise ValueError("A must not contain zero values.")

    num_interior = A_array.size - 2
    dr = float(grid.dr)
    row_indices = []
    column_indices = []
    values = []

    for row, grid_index in enumerate(range(1, A_array.size - 1)):
        diagonal = (A_array[grid_index + 1] - A_array[grid_index - 1]) / (
            2.0 * dr * A_array[grid_index]
        )
        row_indices.append(row)
        column_indices.append(row)
        values.append(diagonal)

        left_grid_index = grid_index - 1
        if left_grid_index >= 1:
            row_indices.append(row)
            column_indices.append(row - 1)
            values.append(-1.0 / (2.0 * dr))

        right_grid_index = grid_index + 1
        if right_grid_index <= A_array.size - 2:
            row_indices.append(row)
            column_indices.append(row + 1)
            values.append(1.0 / (2.0 * dr))

    return sparse.csr_matrix(
        (values, (row_indices, column_indices)),
        shape=(num_interior, num_interior),
    )


def _spsolve(operator, rhs):
    _, sparse_linalg = _require_scipy()

    solution = sparse_linalg.spsolve(operator, rhs)
    return np.asarray(solution, dtype=float), 0, 1


def solve_radial_electric_field_from_charge_density(
    charge_density,
    A,
    grid,
    epsilon_0=1.0,
    tolerance=1.0e-10,
):
    """Solve the radial relativistic Poisson equation from charge density."""

    if epsilon_0 == 0.0:
        raise ValueError("epsilon_0 must be nonzero.")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive.")

    charge_density_array = np.asarray(charge_density, dtype=float)
    if charge_density_array.shape != tuple(grid.r_full.shape):
        raise ValueError(
            "charge_density must have the same shape as grid.r_full: "
            f"expected {tuple(grid.r_full.shape)}, got {charge_density_array.shape}."
        )
    if not np.all(np.isfinite(charge_density_array)):
        raise ValueError("charge_density must contain only finite values.")

    operator = build_radial_poisson_operator(A, grid)
    rhs = charge_density_array[1:-1] / float(epsilon_0)

    if np.allclose(rhs, 0.0):
        interior_solution = np.zeros_like(rhs)
        info = 0
        iterations = 0
    else:
        interior_solution, info, iterations = _spsolve(operator, rhs)

    residual = operator @ interior_solution - rhs
    residual_norm = float(np.linalg.norm(residual, ord=np.inf))

    electric_field = np.zeros_like(np.asarray(grid.r_full, dtype=float))
    electric_field[1:-1] = interior_solution

    return RadialElectricFieldSolveResult(
        electric_field=jnp.asarray(electric_field),
        charge_density=jnp.asarray(charge_density),
        operator=operator,
        residual_norm=residual_norm,
        info=info,
        iterations=iterations,
    )


def solve_radial_electric_field(
    particles,
    A,
    grid,
    epsilon_0=1.0,
    shape_mode="nearest",
    tolerance=1.0e-10,
    maxiter=None,
):
    """Solve the radial relativistic Poisson equation for ``E_r`` with spsolve."""

    charge_density = compute_charge_density(
        particles,
        A,
        grid,
        shape_mode=shape_mode,
    )
    return solve_radial_electric_field_from_charge_density(
        charge_density,
        A,
        grid,
        epsilon_0=epsilon_0,
        tolerance=tolerance,
    )
