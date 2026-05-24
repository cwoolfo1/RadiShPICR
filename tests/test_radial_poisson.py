import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("scipy")

from RadiShPICR.EM.radial_poisson import (
    build_radial_poisson_operator,
    solve_radial_electric_field,
    solve_radial_electric_field_from_charge_density,
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


def test_radial_poisson_operator_uses_centered_interior_stencil_with_backward_outer_row():
    grid = build_radial_grid(epsilon=0.05, r_max=1.0, num_interior_points=5)
    A = jnp.array([1.00, 1.10, 1.40, 1.90, 2.60])

    operator = build_radial_poisson_operator(A, grid)
    dense_operator = operator.toarray()

    assert dense_operator.shape == (4, 4)

    dr = float(grid.dr)
    expected = np.zeros((4, 4))
    full_A = np.asarray(A)
    radial_coordinates = np.asarray(grid.r_full)
    for row, i in enumerate(range(1, 4)):
        expected[row, row] = 2.0 / radial_coordinates[i] + (
            full_A[i + 1] - full_A[i - 1]
        ) / (
            2.0 * dr * full_A[i]
        )
        if i - 1 >= 1:
            expected[row, row - 1] = -1.0 / (2.0 * dr)
        if i + 1 <= 4:
            expected[row, row + 1] = 1.0 / (2.0 * dr)
    expected[-1, -2] = -1.0 / dr
    expected[-1, -1] = 1.0 / dr + 2.0 / radial_coordinates[-1] + (
        full_A[-1] - full_A[-2]
    ) / (
        dr * full_A[-1]
    )

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
    assert np.all(np.isfinite(np.asarray(result.electric_field)))
    assert result.residual_norm <= 1.0e-9


def test_radial_poisson_uses_sparse_direct_solve(monkeypatch):
    import scipy.sparse.linalg as sparse_linalg

    grid = build_radial_grid(epsilon=0.05, r_max=1.0, num_interior_points=5)
    species = make_species(charge=0.2)
    A = jnp.array([1.0, 1.05, 1.10, 1.15, 1.20])

    calls = {"spsolve": 0, "gmres": 0}
    original_spsolve = sparse_linalg.spsolve

    def tracked_spsolve(operator, rhs):
        calls["spsolve"] += 1
        return original_spsolve(operator, rhs)

    def tracked_gmres(*args, **kwargs):
        calls["gmres"] += 1
        raise AssertionError("radial Poisson solve should not call GMRES")

    monkeypatch.setattr(sparse_linalg, "spsolve", tracked_spsolve)
    monkeypatch.setattr(sparse_linalg, "gmres", tracked_gmres)

    result = solve_radial_electric_field(species, A, grid, tolerance=1.0e-11)

    assert calls == {"spsolve": 1, "gmres": 0}
    assert result.info == 0
    assert result.iterations == 1
    assert result.residual_norm <= 1.0e-9


def test_explicit_charge_density_solver_uses_sparse_direct_solve(monkeypatch):
    import scipy.sparse.linalg as sparse_linalg

    grid = build_radial_grid(epsilon=0.05, r_max=1.0, num_interior_points=5)
    A = jnp.array([1.0, 1.05, 1.10, 1.15, 1.20])
    charge_density = jnp.array([0.0, 0.5, -0.2, 0.3, 0.0])

    calls = {"spsolve": 0, "gmres": 0}
    original_spsolve = sparse_linalg.spsolve

    def tracked_spsolve(operator, rhs):
        calls["spsolve"] += 1
        return original_spsolve(operator, rhs)

    def tracked_gmres(*args, **kwargs):
        calls["gmres"] += 1
        raise AssertionError("explicit charge-density solve should not call GMRES")

    monkeypatch.setattr(sparse_linalg, "spsolve", tracked_spsolve)
    monkeypatch.setattr(sparse_linalg, "gmres", tracked_gmres)

    result = solve_radial_electric_field_from_charge_density(
        charge_density,
        A,
        grid,
        epsilon_0=2.0,
        tolerance=1.0e-11,
    )

    assert calls == {"spsolve": 1, "gmres": 0}
    assert result.info == 0
    assert result.iterations == 1
    assert np.allclose(np.asarray(result.charge_density), np.asarray(charge_density))
    assert np.isclose(float(result.electric_field[0]), 0.0)
    assert result.residual_norm <= 1.0e-9

    electric_field = np.asarray(result.electric_field)
    full_A = np.asarray(A)
    dr = float(grid.dr)
    outer_dlnA_dr = (full_A[-1] - full_A[-2]) / (dr * full_A[-1])
    outer_residual = (
        (electric_field[-1] - electric_field[-2]) / dr
        + 2.0 / float(grid.r_full[-1]) * electric_field[-1]
        + outer_dlnA_dr * electric_field[-1]
        - np.asarray(charge_density)[-1] / 2.0
    )
    assert abs(outer_residual) <= 1.0e-7


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


def charged_sphere_exact_field(r, sphere_radius, charge_density, epsilon_0):
    r = np.asarray(r, dtype=float)
    field = np.zeros_like(r)
    nonzero = r > 0.0
    inside = nonzero & (r <= sphere_radius)
    outside = nonzero & (r > sphere_radius)
    field[inside] = charge_density * r[inside] / (3.0 * epsilon_0)
    field[outside] = (
        charge_density
        * sphere_radius**3
        / (3.0 * epsilon_0 * r[outside] ** 2)
    )
    return field


def solve_uniform_charged_sphere(num_grid_points):
    grid = build_radial_grid(epsilon=0.0, r_max=1.0, num_interior_points=num_grid_points)
    r = np.asarray(grid.r_full)
    sphere_radius = 0.25
    charge_density_value = 1.0
    epsilon_0 = 1.0
    charge_density = np.where(r <= sphere_radius, charge_density_value, 0.0)
    result = solve_radial_electric_field_from_charge_density(
        charge_density,
        np.ones_like(r),
        grid,
        epsilon_0=epsilon_0,
    )
    exact = charged_sphere_exact_field(
        r,
        sphere_radius=sphere_radius,
        charge_density=charge_density_value,
        epsilon_0=epsilon_0,
    )
    numerical = np.asarray(result.electric_field)
    mask = r > 0.0
    l2_error = float(np.sqrt(np.mean((numerical[mask] - exact[mask]) ** 2)))
    linf_error = float(np.max(np.abs(numerical[mask] - exact[mask])))
    return grid, result, exact, l2_error, linf_error


def test_flat_space_charged_sphere_matches_exact_solution():
    grid, result, exact, l2_error, linf_error = solve_uniform_charged_sphere(257)

    assert result.info == 0
    assert result.residual_norm <= 1.0e-10
    assert np.isclose(float(result.electric_field[0]), 0.0)
    assert l2_error <= 1.0e-3
    assert linf_error <= 3.0e-3
    assert np.isclose(float(result.electric_field[-1]), exact[-1], rtol=0.03)


def test_flat_space_charged_sphere_converges_at_first_order():
    grid_sizes = [65, 129, 257, 513]
    errors = []
    spacings = []

    for grid_size in grid_sizes:
        grid, _, _, l2_error, _ = solve_uniform_charged_sphere(grid_size)
        spacings.append(float(grid.dr))
        errors.append(l2_error)

    orders = [
        np.log(errors[index - 1] / errors[index]) / np.log(spacings[index - 1] / spacings[index])
        for index in range(1, len(errors))
    ]

    assert all(fine < coarse for coarse, fine in zip(errors, errors[1:]))
    assert min(orders[-2:]) >= 0.85
