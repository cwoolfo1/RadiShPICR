import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("scipy")

from RadiShPICR.EM.radial_poisson import (
    build_radial_poisson_operator,
    solve_radial_electric_field,
)
from RadiShPICR.deposition import compute_charge_density
from RadiShPICR.particles.particle_species import particle_species
from RadiShPICR.relativity.grid import build_radial_grid


def make_species(charge=0.0, weight=1.0):
    return particle_species(
        name="charged-dust",
        number_of_particles=3,
        charge=charge,
        mass=1.0,
        temperature=0.0,
        r=jnp.array([0.25, 0.50, 0.75]),
        phi=jnp.zeros(3),
        u_r=jnp.zeros(3),
        u_phi=jnp.zeros(3),
        weight=weight,
        r_min=0.0,
        r_max=1.0,
        dr=0.25,
    )


def test_radial_poisson_operator_uses_centered_interior_stencil():
    grid = build_radial_grid(epsilon=0.05, r_max=1.0, num_interior_points=5)
    A = jnp.array([1.00, 1.10, 1.40, 1.90, 2.60])

    operator = build_radial_poisson_operator(A, grid)
    dense_operator = operator.toarray()

    assert dense_operator.shape == (3, 3)

    dr = float(grid.dr)
    expected = np.zeros((3, 3))
    full_A = np.asarray(A)
    for row, i in enumerate(range(1, 4)):
        expected[row, row] = (full_A[i + 1] - full_A[i - 1]) / (
            2.0 * dr * full_A[i]
        )
        if i - 1 >= 1:
            expected[row, row - 1] = -1.0 / (2.0 * dr)
        if i + 1 <= 3:
            expected[row, row + 1] = 1.0 / (2.0 * dr)

    assert np.allclose(dense_operator, expected)


def test_zero_charge_density_solves_zero_electric_field():
    grid = build_radial_grid(epsilon=0.05, r_max=1.0, num_interior_points=5)
    species = make_species(charge=0.0)
    A = jnp.ones_like(grid.r_full)

    result = solve_radial_electric_field(species, A, grid)

    assert result.info == 0
    assert result.iterations >= 0
    assert result.electric_field.shape == grid.r_full.shape
    assert np.allclose(np.asarray(result.electric_field), 0.0)
    assert result.residual_norm <= 1.0e-10


def test_nonzero_charge_density_returns_finite_field_with_small_residual():
    grid = build_radial_grid(epsilon=0.05, r_max=1.0, num_interior_points=5)
    species = make_species(charge=0.2)
    A = jnp.array([1.0, 1.05, 1.10, 1.15, 1.20])

    result = solve_radial_electric_field(species, A, grid, tolerance=1.0e-11)

    assert result.info == 0
    assert result.electric_field.shape == grid.r_full.shape
    assert np.isclose(float(result.electric_field[0]), 0.0)
    assert np.isclose(float(result.electric_field[-1]), 0.0)
    assert np.all(np.isfinite(np.asarray(result.electric_field)))
    assert result.residual_norm <= 1.0e-9


def test_solver_uses_metric_charge_density_with_species_weight():
    grid = build_radial_grid(epsilon=0.05, r_max=1.0, num_interior_points=5)
    A = jnp.array([1.0, 1.05, 1.10, 1.15, 1.20])
    unit_weight_species = make_species(charge=0.2, weight=1.0)
    weighted_species = make_species(charge=0.2, weight=3.0)

    unit_result = solve_radial_electric_field(unit_weight_species, A, grid)
    weighted_result = solve_radial_electric_field(weighted_species, A, grid)
    expected_weighted_charge_density = compute_charge_density(
        weighted_species,
        A,
        grid,
    )

    assert np.allclose(
        np.asarray(weighted_result.charge_density),
        np.asarray(expected_weighted_charge_density),
    )
    assert np.allclose(
        np.asarray(weighted_result.electric_field[1:-1]),
        3.0 * np.asarray(unit_result.electric_field[1:-1]),
    )
