import jax.numpy as jnp

from RadiShPICR.particles.particle_species import particle_species
from RadiShPICR.relativity.A import (
    build_dense_operator,
    nonlinear_residual_u,
    solve_metric_A,
)
from RadiShPICR.relativity.grid import build_radial_grid
from RadiShPICR.relativity.metric import compute_metric
from RadiShPICR.relativity.solvers import (
    finite_difference_residual_jacobian,
    solve_metric_A_broyden,
)


def make_zero_source_species(grid):
    return particle_species(
        name="zero-source",
        number_of_particles=1,
        charge=0.0,
        mass=0.0,
        temperature=0.0,
        r=jnp.array([0.50]),
        phi=jnp.zeros(1),
        u_r=jnp.zeros(1),
        u_phi=jnp.zeros(1),
        weight=1.0,
        r_min=0.0,
        r_max=grid.r_max,
        dr=grid.dr,
    )


def make_static_charged_sphere(grid, num_particles=16):
    sphere_radius = 0.65
    particle_fraction = (jnp.arange(num_particles) + 0.5) / float(num_particles)
    radial_positions = sphere_radius * particle_fraction ** (1.0 / 3.0)

    return particle_species(
        name="static-charged-sphere",
        number_of_particles=num_particles,
        charge=2.0e-3,
        mass=1.0e-4,
        temperature=0.0,
        r=radial_positions,
        phi=jnp.zeros(num_particles),
        u_r=jnp.zeros(num_particles),
        u_phi=jnp.zeros(num_particles),
        weight=1.0,
        r_min=0.0,
        r_max=grid.r_max,
        dr=grid.dr,
    )


def test_broyden_metric_A_solver_converges_for_zero_source():
    grid = build_radial_grid(epsilon=0.05, r_max=1.0, num_interior_points=5)
    species = make_zero_source_species(grid)
    initial_A = jnp.ones_like(grid.r_full)

    solved_A, converged, residual = solve_metric_A_broyden(
        species,
        grid,
        schwarzschild_mass=0.0,
        initial_A_guess=initial_A,
        tolerance=1.0e-8,
        max_broyden_steps=4,
    )

    assert bool(converged)
    assert float(residual) <= 1.0e-8
    assert jnp.allclose(solved_A, initial_A)


def test_finite_difference_jacobian_matches_zero_source_operator():
    grid = build_radial_grid(epsilon=0.05, r_max=1.0, num_interior_points=5)
    U = jnp.ones_like(grid.r_full)
    source_term = jnp.zeros_like(grid.r_full)
    exact_exterior_points = jnp.array([False, False, False, False, True])
    boundary_u = jnp.ones_like(grid.r_full)

    def residual_from_U(trial_U):
        return nonlinear_residual_u(
            trial_U,
            source_term,
            grid,
            boundary_u,
            exact_exterior_points,
        )

    jacobian = finite_difference_residual_jacobian(
        residual_from_U,
        U,
        step_size=1.0e-4,
    )
    expected = build_dense_operator(
        grid.r_full,
        grid.dr,
        jnp.zeros_like(grid.r_full),
        exact_exterior_points,
    )

    assert jnp.allclose(jacobian, expected, rtol=2.0e-3, atol=2.0e-5)


def test_broyden_metric_A_solver_matches_newton_for_static_charged_sphere():
    grid = build_radial_grid(epsilon=0.05, r_max=1.2, num_interior_points=13)
    species = make_static_charged_sphere(grid)
    initial_A = jnp.ones_like(grid.r_full)

    newton_A, newton_converged, newton_residual = solve_metric_A(
        species,
        grid,
        schwarzschild_mass=0.0,
        initial_A_guess=initial_A,
        tolerance=1.0e-8,
        max_newton_steps=80,
        max_line_search_steps=40,
        EM=True,
    )
    broyden_A, broyden_converged, broyden_residual = solve_metric_A_broyden(
        species,
        grid,
        schwarzschild_mass=0.0,
        initial_A_guess=initial_A,
        tolerance=1.0e-7,
        max_broyden_steps=80,
        max_line_search_steps=40,
        EM=True,
    )

    assert bool(newton_converged)
    assert bool(broyden_converged)
    assert float(newton_residual) <= 1.0e-8
    assert float(broyden_residual) <= 1.0e-7
    assert jnp.allclose(broyden_A, newton_A, rtol=2.0e-4, atol=2.0e-6)


def test_compute_metric_uses_broyden_metric_A_solver_for_static_charged_sphere():
    grid = build_radial_grid(epsilon=0.05, r_max=1.2, num_interior_points=13)
    species = make_static_charged_sphere(grid)
    initial_A = jnp.ones_like(grid.r_full)

    newton_metric = compute_metric(
        species,
        grid,
        schwarzschild_mass=0.0,
        initial_A_guess=initial_A,
        metric_A_solver="newton",
        EM=True,
    )
    broyden_metric = compute_metric(
        species,
        grid,
        schwarzschild_mass=0.0,
        initial_A_guess=initial_A,
        metric_A_solver="broyden",
        EM=True,
    )

    assert jnp.allclose(broyden_metric.A, newton_metric.A, rtol=2.0e-4, atol=2.0e-6)
