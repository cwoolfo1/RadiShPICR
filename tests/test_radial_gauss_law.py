import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from RadiShPICR.EM.gauss_law import (
    build_radial_gauss_law_operator,
    compute_charge_density_and_radial_electric_field,
    compute_radial_electric_field,
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


def radial_gauss_law_residual(electric_field, charge_density, A, grid, epsilon_0=1.0):
    operator = build_radial_gauss_law_operator(A, grid)
    rhs = np.asarray(charge_density, dtype=float)[1:] / float(epsilon_0)
    return np.asarray(operator) @ np.asarray(electric_field, dtype=float)[1:] - rhs


def test_radial_gauss_law_operator_uses_centered_interior_stencil_with_backward_outer_row():
    grid = build_radial_grid(epsilon=0.05, r_max=1.0, num_interior_points=5)
    A = jnp.array([1.00, 1.10, 1.40, 1.90, 2.60])

    operator = build_radial_gauss_law_operator(A, grid)
    dense_operator = np.asarray(operator)

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

    charge_density, electric_field = compute_charge_density_and_radial_electric_field(
        [species],
        A,
        grid,
    )

    assert charge_density.shape == grid.r_full.shape
    assert electric_field.shape == grid.r_full.shape
    assert np.allclose(np.asarray(electric_field), 0.0)


def test_nonzero_charge_density_returns_finite_field_with_small_residual():
    grid = build_radial_grid(epsilon=0.05, r_max=1.0, num_interior_points=5)
    species = make_species(charge=0.2)
    A = jnp.array([1.0, 1.05, 1.10, 1.15, 1.20])

    charge_density, electric_field = compute_charge_density_and_radial_electric_field(
        [species],
        A,
        grid,
    )
    residual = radial_gauss_law_residual(electric_field, charge_density, A, grid)

    assert electric_field.shape == grid.r_full.shape
    assert np.isclose(float(electric_field[0]), 0.0)
    assert np.all(np.isfinite(np.asarray(electric_field)))
    assert np.linalg.norm(residual, ord=np.inf) <= 1.0e-6


def test_radial_gauss_law_particle_helper_is_jittable():
    grid = build_radial_grid(epsilon=0.05, r_max=1.0, num_interior_points=5)
    species = make_species(charge=0.2)
    A = jnp.array([1.0, 1.05, 1.10, 1.15, 1.20])

    solve = jax.jit(
        lambda metric_A: compute_charge_density_and_radial_electric_field(
            [species],
            metric_A,
            grid,
        )
    )
    charge_density, electric_field = solve(A)
    residual = radial_gauss_law_residual(electric_field, charge_density, A, grid)

    assert isinstance(build_radial_gauss_law_operator(A, grid), jax.Array)
    assert np.linalg.norm(residual, ord=np.inf) <= 1.0e-6


def test_explicit_charge_density_solver_is_jittable():
    grid = build_radial_grid(epsilon=0.05, r_max=1.0, num_interior_points=5)
    A = jnp.array([1.0, 1.05, 1.10, 1.15, 1.20])
    charge_density = jnp.array([0.0, 0.5, -0.2, 0.3, 0.0])

    solve = jax.jit(
        lambda density, metric_A: compute_radial_electric_field(
            density,
            metric_A,
            grid,
            epsilon_0=2.0,
        )
    )
    electric_field = solve(charge_density, A)
    residual = radial_gauss_law_residual(
        electric_field,
        charge_density,
        A,
        grid,
        epsilon_0=2.0,
    )

    assert np.isclose(float(electric_field[0]), 0.0)
    assert np.linalg.norm(residual, ord=np.inf) <= 1.0e-6

    electric_field = np.asarray(electric_field)
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

    _, unit_electric_field = compute_charge_density_and_radial_electric_field(
        [unit_weight_species],
        A,
        grid,
    )
    weighted_charge_density, weighted_electric_field = (
        compute_charge_density_and_radial_electric_field([weighted_species], A, grid)
    )
    expected_weighted_charge_density = compute_charge_density(
        weighted_species,
        A,
        grid,
    )

    assert np.allclose(
        np.asarray(weighted_charge_density),
        np.asarray(expected_weighted_charge_density),
    )
    assert np.allclose(
        np.asarray(weighted_electric_field[1:-1]),
        3.0 * np.asarray(unit_electric_field[1:-1]),
    )


def test_particle_list_helper_sums_charge_density_from_multiple_species():
    grid = build_radial_grid(epsilon=0.05, r_max=1.0, num_interior_points=5)
    A = jnp.array([1.0, 1.05, 1.10, 1.15, 1.20])
    positive_species = make_species(charge=0.2, weight=2.0)
    negative_species = make_species(charge=-0.1, weight=3.0)

    charge_density, electric_field = compute_charge_density_and_radial_electric_field(
        [positive_species, negative_species],
        A,
        grid,
    )
    expected_charge_density = compute_charge_density(positive_species, A, grid)
    expected_charge_density = expected_charge_density + compute_charge_density(
        negative_species,
        A,
        grid,
    )

    assert np.allclose(np.asarray(charge_density), np.asarray(expected_charge_density))
    assert np.all(np.isfinite(np.asarray(electric_field)))


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
    electric_field = compute_radial_electric_field(
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
    numerical = np.asarray(electric_field)
    mask = r > 0.0
    l2_error = float(np.sqrt(np.mean((numerical[mask] - exact[mask]) ** 2)))
    linf_error = float(np.max(np.abs(numerical[mask] - exact[mask])))
    return grid, charge_density, electric_field, exact, l2_error, linf_error


def test_flat_space_charged_sphere_matches_exact_solution():
    grid, charge_density, electric_field, exact, l2_error, linf_error = (
        solve_uniform_charged_sphere(257)
    )
    residual = radial_gauss_law_residual(
        electric_field,
        charge_density,
        np.ones_like(grid.r_full),
        grid,
    )

    assert np.linalg.norm(residual, ord=np.inf) <= 2.0e-6
    assert np.isclose(float(electric_field[0]), 0.0)
    assert l2_error <= 1.0e-3
    assert linf_error <= 3.0e-3
    assert np.isclose(float(electric_field[-1]), exact[-1], rtol=0.03)


def test_flat_space_charged_sphere_converges_at_first_order():
    grid_sizes = [65, 129, 257, 513]
    errors = []
    spacings = []

    for grid_size in grid_sizes:
        grid, _, _, _, l2_error, _ = solve_uniform_charged_sphere(grid_size)
        spacings.append(float(grid.dr))
        errors.append(l2_error)

    orders = [
        np.log(errors[index - 1] / errors[index]) / np.log(spacings[index - 1] / spacings[index])
        for index in range(1, len(errors))
    ]

    assert all(fine < coarse for coarse, fine in zip(errors, errors[1:]))
    assert min(orders[-2:]) >= 0.85
