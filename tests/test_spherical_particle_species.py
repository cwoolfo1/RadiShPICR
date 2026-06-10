import inspect

import jax
import jax.numpy as jnp

from RadiShPICR.particles.particle_species import particle_species

from RadiShPICR.deposition.charge_density import compute_charge_density

from RadiShPICR.deposition.mass_density import compute_mass_density

from RadiShPICR.deposition.number_density import (
    compute_number_density,
    compute_number_density_metric_derivative,
    compute_number_density_metric_jacobian,
)



from RadiShPICR.relativity.energy_momentum_tensor import (
    compute_Sr,
    compute_Srr,
)
from RadiShPICR.evolve import advance_one_step, rk4_step
from RadiShPICR.relativity.geodesic import compute_geodesic_terms
from RadiShPICR.relativity.grid import build_radial_grid
from RadiShPICR.relativity.metric import MetricState, compute_metric


def make_species():
    return particle_species(
        name="ions",
        number_of_particles=3,
        charge=2.0,
        mass=4.0,
        temperature=0.5,
        r=jnp.array([0.25, 0.50, 0.75]),
        phi=jnp.array([0.0, 0.1, 0.2]),
        u_r=jnp.array([0.01, -0.02, 0.03]),
        u_phi=jnp.array([0.4, 0.5, 0.6]),
        weight=3.0,
        r_min=0.0,
        r_max=1.0,
        dr=0.25,
        shape=2,
        dt=0.01,
    )


def make_second_species():
    return particle_species(
        name="electrons",
        number_of_particles=3,
        charge=-1.0,
        mass=1.0,
        temperature=0.0,
        r=jnp.array([0.20, 0.45, 0.70]),
        phi=jnp.array([0.0, 0.2, 0.4]),
        u_r=jnp.array([-0.04, 0.01, 0.02]),
        u_phi=jnp.array([0.1, 0.2, 0.3]),
        weight=2.0,
        r_min=0.0,
        r_max=1.0,
        dr=0.25,
        shape=2,
        dt=0.01,
    )


def test_species_stores_spherical_state_and_scalar_metadata():
    species = make_species()

    assert species.count() == 3
    assert species.get_number_of_particles() == 3
    assert species.get_name() == "ions"
    assert species.mass == 4.0
    assert species.charge == 2.0
    assert species.weight == 3.0
    assert species.get_mass() == 12.0
    assert species.get_charge() == 6.0
    assert species.get_temperature() == 0.5
    r, phi = species.get_position()
    u_r, u_phi = species.get_velocity()
    assert jnp.allclose(r, species.r)
    assert jnp.allclose(phi, species.phi)
    assert jnp.allclose(u_r, species.u_r)
    assert jnp.allclose(u_phi, species.u_phi)
    assert not hasattr(species, "u" + "_theta")


def test_orbital_update_only_replaces_r_phi_and_u_r():
    species = make_species()

    updated = species.with_updated_orbital_state(
        jnp.array([0.3, 0.6, 0.9]),
        jnp.array([1.0, 1.1, 1.2]),
        jnp.array([0.2, 0.3, 0.4]),
    )

    assert updated is not species
    assert jnp.allclose(updated.r, jnp.array([0.3, 0.6, 0.9]))
    assert jnp.allclose(updated.phi, jnp.array([1.0, 1.1, 1.2]))
    assert jnp.allclose(updated.u_r, jnp.array([0.2, 0.3, 0.4]))
    assert jnp.allclose(updated.u_phi, species.u_phi)


def test_boundary_conditions_are_intentionally_inactive():
    species = make_species()

    unchanged = species.boundary_conditions()

    assert unchanged is species
    assert jnp.allclose(species.r, jnp.array([0.25, 0.50, 0.75]))
    assert jnp.allclose(species.u_r, jnp.array([0.01, -0.02, 0.03]))


def test_radial_index_uses_interior_deposition_cells():
    species = particle_species(
        name="edge-test",
        number_of_particles=4,
        charge=1.0,
        mass=1.0,
        temperature=0.0,
        r=jnp.array([-0.1, 0.0, 1.0, 1.2]),
        phi=jnp.zeros(4),
        u_r=jnp.zeros(4),
        u_phi=jnp.zeros(4),
        r_min=0.0,
        r_max=1.0,
        dr=0.25,
    )

    assert jnp.all(species.get_index() == jnp.array([1, 1, 3, 3]))


def test_species_is_jax_pytree():
    species = make_species()

    leaves, treedef = jax.tree_util.tree_flatten(species)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)

    assert rebuilt.name == species.name
    assert rebuilt.mass == species.mass
    assert rebuilt.charge == species.charge
    assert jnp.allclose(rebuilt.r, species.r)
    assert jnp.allclose(rebuilt.phi, species.phi)
    assert jnp.allclose(rebuilt.u_r, species.u_r)
    assert jnp.allclose(rebuilt.u_phi, species.u_phi)


def test_number_density_metric_derivatives_live_with_number_density_deposit():
    assert (
        compute_number_density_metric_derivative.__module__
        == "RadiShPICR.deposition.number_density"
    )
    assert (
        compute_number_density_metric_jacobian.__module__
        == "RadiShPICR.deposition.number_density"
    )


def test_charge_density_scales_new_number_density_by_scalar_charge():
    grid = build_radial_grid(epsilon=0.05, r_max=1.0, num_interior_points=5)
    species = particle_species(
        name="charged-dust",
        number_of_particles=1,
        charge=2.0,
        mass=4.0,
        temperature=0.0,
        r=jnp.array([0.50]),
        phi=jnp.zeros(1),
        u_r=jnp.array([0.30]),
        u_phi=jnp.array([0.40]),
        weight=3.0,
        r_min=0.0,
        r_max=1.0,
        dr=0.25,
    )
    A = jnp.ones_like(grid.r_full)

    number_density = compute_number_density(species, A, grid)
    charge_density = compute_charge_density(species, A, grid)

    assert jnp.allclose(charge_density, species.get_charge() * number_density)


def test_charge_density_uses_same_relativistic_factor_as_number_density():
    grid = build_radial_grid(epsilon=0.05, r_max=1.0, num_interior_points=5)
    resting_species = make_species()
    moving_species = resting_species.with_updated_radial_state(
        resting_species.r,
        jnp.array([0.4, -0.5, 0.6]),
    )
    A = jnp.ones_like(grid.r_full)

    resting_charge_density = compute_charge_density(resting_species, A, grid)
    moving_charge_density = compute_charge_density(moving_species, A, grid)
    resting_number_density = compute_number_density(resting_species, A, grid)
    moving_number_density = compute_number_density(moving_species, A, grid)

    assert jnp.allclose(
        resting_charge_density,
        resting_species.get_charge() * resting_number_density,
    )
    assert jnp.allclose(
        moving_charge_density,
        moving_species.get_charge() * moving_number_density,
    )
    assert not jnp.allclose(moving_charge_density, resting_charge_density)


def test_scalar_mass_broadcasts_in_source_terms():
    grid = build_radial_grid(epsilon=0.05, r_max=1.0, num_interior_points=5)
    species = make_species()
    A = jnp.ones_like(grid.r_full)

    for shape_mode in ("nearest", "quadratic"):
        Sr_from_species = compute_Sr(species, A, grid, shape_mode=shape_mode)
        Srr_from_species = compute_Srr(species, A, grid, shape_mode=shape_mode)

        assert Sr_from_species.shape == grid.r_full.shape
        assert Srr_from_species.shape == grid.r_full.shape


def test_A_solver_source_terms_include_electromagnetic_energy_density(monkeypatch):
    import RadiShPICR.relativity.A as A_solver

    grid = build_radial_grid(epsilon=0.05, r_max=1.0, num_interior_points=5)
    species = make_species()
    A = jnp.ones_like(grid.r_full)
    electric_field = jnp.array([0.0, 0.2, 0.4, 0.6, 0.0])

    def fake_charge_density_and_field(
        particle_list,
        metric_A,
        solve_grid,
        shape_mode="nearest",
    ):
        assert particle_list == [species]
        assert jnp.allclose(metric_A, A)
        assert solve_grid is grid
        return jnp.zeros_like(grid.r_full), electric_field

    monkeypatch.setattr(
        A_solver,
        "compute_charge_density_and_radial_electric_field",
        fake_charge_density_and_field,
    )

    _, rho, _, source_term, _, _ = A_solver.metric_source_terms_from_U(
        species,
        jnp.sqrt(A),
        grid,
    )

    expected_particle_rho = compute_mass_density(species, A, grid)
    expected_total_rho = expected_particle_rho + 0.5 * electric_field**2

    assert jnp.allclose(rho, expected_total_rho)
    assert jnp.allclose(source_term, -2.0 * jnp.pi * expected_total_rho)


def test_A_solver_source_terms_skip_electromagnetic_energy_density_when_EM_is_false(monkeypatch):
    import RadiShPICR.relativity.A as A_solver

    grid = build_radial_grid(epsilon=0.05, r_max=1.0, num_interior_points=5)
    species = make_species()
    A = jnp.ones_like(grid.r_full)

    def fail_charge_density_and_field(
        particle_list,
        metric_A,
        solve_grid,
        shape_mode="nearest",
    ):
        raise AssertionError("EM=False should not solve Gauss law for the A source")

    monkeypatch.setattr(
        A_solver,
        "compute_charge_density_and_radial_electric_field",
        fail_charge_density_and_field,
    )

    _, rho, _, source_term, _, _ = A_solver.metric_source_terms_from_U(
        species,
        jnp.sqrt(A),
        grid,
        EM=False,
    )

    expected_particle_rho = compute_mass_density(species, A, grid)

    assert jnp.allclose(rho, expected_particle_rho)
    assert jnp.allclose(source_term, -2.0 * jnp.pi * expected_particle_rho)


def test_A_solver_source_jacobian_matches_jax_autodiff_with_EM():
    import RadiShPICR.relativity.A as A_solver

    grid = build_radial_grid(epsilon=0.05, r_max=1.0, num_interior_points=5)
    species = make_species()
    U = jnp.sqrt(jnp.array([1.0, 1.05, 1.10, 1.15, 1.20]))

    _, _, _, _, _, source_term_U_jacobian = A_solver.metric_source_terms_from_U(
        species,
        U,
        grid,
    )

    def source_term_from_U(trial_U):
        _, _, _, source_term, _, _ = A_solver.metric_source_terms_from_U(
            species,
            trial_U,
            grid,
        )
        return source_term

    autodiff_source_term_U_jacobian = jax.jacfwd(source_term_from_U)(U)

    assert jnp.allclose(
        source_term_U_jacobian,
        autodiff_source_term_U_jacobian,
        rtol=2.0e-5,
        atol=2.0e-7,
    )


def test_solve_metric_A_is_jitted_for_zero_source():
    import RadiShPICR.relativity.A as A_solver

    grid = build_radial_grid(epsilon=0.05, r_max=1.0, num_interior_points=5)
    species = particle_species(
        name="neutral-test-particles",
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
        r_max=1.0,
        dr=0.25,
    )
    initial_A = jnp.ones_like(grid.r_full)

    solved_A, converged, residual = A_solver.solve_metric_A(
        species,
        grid,
        schwarzschild_mass=0.0,
        initial_A_guess=initial_A,
        max_newton_steps=3,
    )

    assert bool(converged)
    assert float(residual) <= 1.0e-6
    assert jnp.allclose(solved_A, initial_A)


def test_solve_metric_A_skips_gauss_law_when_EM_is_false(monkeypatch):
    import RadiShPICR.relativity.A as A_solver

    grid = build_radial_grid(epsilon=0.05, r_max=1.0, num_interior_points=5)
    species = particle_species(
        name="charged-massless-test-particles",
        number_of_particles=1,
        charge=1.0,
        mass=0.0,
        temperature=0.0,
        r=jnp.array([0.50]),
        phi=jnp.zeros(1),
        u_r=jnp.zeros(1),
        u_phi=jnp.zeros(1),
        weight=1.0,
        r_min=0.0,
        r_max=1.0,
        dr=0.25,
    )
    initial_A = jnp.ones_like(grid.r_full)

    def fail_charge_density_and_field(
        particle_list,
        metric_A,
        solve_grid,
        shape_mode="nearest",
    ):
        raise AssertionError("EM=False should not solve Gauss law in solve_metric_A")

    monkeypatch.setattr(
        A_solver,
        "compute_charge_density_and_radial_electric_field",
        fail_charge_density_and_field,
    )

    solved_A, converged, residual = A_solver.solve_metric_A(
        species,
        grid,
        schwarzschild_mass=0.0,
        initial_A_guess=initial_A,
        max_newton_steps=3,
        EM=False,
    )

    assert bool(converged)
    assert float(residual) <= 1.0e-6
    assert jnp.allclose(solved_A, initial_A)


def test_A_solver_linear_correction_uses_jax_direct_solve():
    import RadiShPICR.relativity.A as A_solver

    jacobian_matrix = jnp.array(
        [
            [4.0, 1.0, 0.0],
            [1.0, 3.0, 1.0],
            [0.0, 1.0, 2.0],
        ]
    )
    residual = jnp.array([1.0, -2.0, 0.5])

    delta_U = A_solver.solve_newton_linear_correction(
        jacobian_matrix,
        residual,
    )

    expected_delta_U = jnp.linalg.solve(jacobian_matrix, -residual)

    assert jnp.allclose(delta_U, expected_delta_U)


def test_compute_metric_reports_electromagnetic_energy_density_in_rho(monkeypatch):
    import RadiShPICR.relativity.metric as metric_module

    grid = build_radial_grid(epsilon=0.05, r_max=1.0, num_interior_points=5)
    species = make_species()
    solved_A = jnp.ones_like(grid.r_full)
    electric_field = jnp.array([0.0, 0.3, 0.5, 0.7, 0.0])

    def fake_solve_metric_A(
        particles,
        metric_grid,
        schwarzschild_mass,
        initial_A_guess,
        shape_mode="nearest",
        EM=True,
    ):
        assert EM is True
        return solved_A, True, 0.0

    def fake_charge_density_and_field(
        particle_list,
        metric_A,
        solve_grid,
        shape_mode="nearest",
    ):
        assert particle_list == [species]
        assert jnp.allclose(metric_A, solved_A)
        assert solve_grid is grid
        return jnp.zeros_like(grid.r_full), electric_field

    monkeypatch.setattr(metric_module, "solve_metric_A", fake_solve_metric_A)
    monkeypatch.setattr(
        metric_module,
        "compute_charge_density_and_radial_electric_field",
        fake_charge_density_and_field,
    )

    metric = compute_metric(
        species,
        grid,
        schwarzschild_mass=0.0,
        initial_A_guess=solved_A,
    )

    expected_rho = compute_mass_density(
        species,
        solved_A,
        grid,
    ) + 0.5 * electric_field**2
    expected_rho = jnp.where(metric.exact_exterior_points, 0.0, expected_rho)

    assert jnp.allclose(metric.rho, expected_rho)


def test_compute_metric_sums_particle_and_em_sources_for_species_list(monkeypatch):
    import RadiShPICR.relativity.metric as metric_module

    grid = build_radial_grid(epsilon=0.05, r_max=1.0, num_interior_points=5)
    ions = make_species()
    electrons = make_second_species()
    particles = [ions, electrons]
    solved_A = jnp.ones_like(grid.r_full)
    electric_field = jnp.array([0.0, 0.2, 0.4, 0.6, 0.0])

    def fake_solve_metric_A(
        particle_list,
        metric_grid,
        schwarzschild_mass,
        initial_A_guess,
        shape_mode="nearest",
        EM=True,
    ):
        assert particle_list == particles
        assert EM is True
        return solved_A, True, 0.0

    def fake_charge_density_and_field(
        particle_list,
        metric_A,
        solve_grid,
        shape_mode="nearest",
    ):
        assert particle_list == particles
        assert jnp.allclose(metric_A, solved_A)
        assert solve_grid is grid
        return jnp.zeros_like(grid.r_full), electric_field

    monkeypatch.setattr(metric_module, "solve_metric_A", fake_solve_metric_A)
    monkeypatch.setattr(
        metric_module,
        "compute_charge_density_and_radial_electric_field",
        fake_charge_density_and_field,
    )

    metric = compute_metric(
        particles,
        grid,
        schwarzschild_mass=0.0,
        initial_A_guess=solved_A,
    )

    expected_rho = (
        compute_mass_density(ions, solved_A, grid)
        + compute_mass_density(electrons, solved_A, grid)
        + 0.5 * electric_field**2
    )
    expected_Sr = compute_Sr(ions, solved_A, grid) + compute_Sr(electrons, solved_A, grid)
    expected_Srr = (
        compute_Srr(ions, solved_A, grid)
        + compute_Srr(electrons, solved_A, grid)
        + 0.5 * solved_A**2 * electric_field**2
    )
    expected_rho = jnp.where(metric.exact_exterior_points, 0.0, expected_rho)
    expected_Sr = jnp.where(metric.exact_exterior_points, 0.0, expected_Sr)
    expected_Srr = jnp.where(metric.exact_exterior_points, 0.0, expected_Srr)

    assert jnp.allclose(metric.rho, expected_rho)
    assert jnp.allclose(metric.S_r, expected_Sr)
    assert jnp.allclose(metric.S_rr, expected_Srr)


def test_compute_metric_reports_electromagnetic_stress_in_Srr(monkeypatch):
    import RadiShPICR.relativity.metric as metric_module

    grid = build_radial_grid(epsilon=0.05, r_max=1.0, num_interior_points=5)
    species = make_species()
    solved_A = jnp.array([1.0, 1.05, 1.10, 1.15, 1.20])
    electric_field = jnp.array([0.0, 0.3, 0.5, 0.7, 0.0])

    def fake_solve_metric_A(
        particles,
        metric_grid,
        schwarzschild_mass,
        initial_A_guess,
        shape_mode="nearest",
        EM=True,
    ):
        assert EM is True
        return solved_A, True, 0.0

    def fake_charge_density_and_field(
        particle_list,
        metric_A,
        solve_grid,
        shape_mode="nearest",
    ):
        assert particle_list == [species]
        assert jnp.allclose(metric_A, solved_A)
        assert solve_grid is grid
        return jnp.zeros_like(grid.r_full), electric_field

    monkeypatch.setattr(metric_module, "solve_metric_A", fake_solve_metric_A)
    monkeypatch.setattr(
        metric_module,
        "compute_charge_density_and_radial_electric_field",
        fake_charge_density_and_field,
    )

    metric = compute_metric(
        species,
        grid,
        schwarzschild_mass=0.0,
        initial_A_guess=solved_A,
    )

    expected_Srr = compute_Srr(species, solved_A, grid) + 0.5 * solved_A**2 * electric_field**2
    expected_Srr = jnp.where(metric.exact_exterior_points, 0.0, expected_Srr)

    assert jnp.allclose(metric.S_rr, expected_Srr)


def test_compute_metric_keeps_neutral_Srr_particle_only(monkeypatch):
    import RadiShPICR.relativity.metric as metric_module

    grid = build_radial_grid(epsilon=0.05, r_max=1.0, num_interior_points=5)
    species = make_species()
    solved_A = jnp.array([1.0, 1.05, 1.10, 1.15, 1.20])
    electric_field = jnp.zeros_like(grid.r_full)

    def fake_solve_metric_A(
        particles,
        metric_grid,
        schwarzschild_mass,
        initial_A_guess,
        shape_mode="nearest",
        EM=True,
    ):
        assert EM is True
        return solved_A, True, 0.0

    def fake_charge_density_and_field(
        particle_list,
        metric_A,
        solve_grid,
        shape_mode="nearest",
    ):
        return jnp.zeros_like(grid.r_full), electric_field

    monkeypatch.setattr(metric_module, "solve_metric_A", fake_solve_metric_A)
    monkeypatch.setattr(
        metric_module,
        "compute_charge_density_and_radial_electric_field",
        fake_charge_density_and_field,
    )

    metric = compute_metric(
        species,
        grid,
        schwarzschild_mass=0.0,
        initial_A_guess=solved_A,
    )

    expected_Srr = compute_Srr(species, solved_A, grid)
    expected_Srr = jnp.where(metric.exact_exterior_points, 0.0, expected_Srr)

    assert jnp.allclose(metric.S_rr, expected_Srr)


def test_compute_metric_skips_electromagnetic_sources_when_EM_is_false(monkeypatch):
    import RadiShPICR.relativity.metric as metric_module

    grid = build_radial_grid(epsilon=0.05, r_max=1.0, num_interior_points=5)
    species = make_species()
    solved_A = jnp.array([1.0, 1.05, 1.10, 1.15, 1.20])

    def fake_solve_metric_A(
        particles,
        metric_grid,
        schwarzschild_mass,
        initial_A_guess,
        shape_mode="nearest",
        EM=True,
    ):
        assert EM is False
        return solved_A, True, 0.0

    def fail_charge_density_and_field(
        particle_list,
        metric_A,
        solve_grid,
        shape_mode="nearest",
    ):
        raise AssertionError("EM=False should not solve Gauss law in compute_metric")

    monkeypatch.setattr(metric_module, "solve_metric_A", fake_solve_metric_A)
    monkeypatch.setattr(
        metric_module,
        "compute_charge_density_and_radial_electric_field",
        fail_charge_density_and_field,
    )

    metric = compute_metric(
        species,
        grid,
        schwarzschild_mass=0.0,
        initial_A_guess=solved_A,
        EM=False,
    )

    expected_rho = compute_mass_density(species, solved_A, grid)
    expected_Srr = compute_Srr(species, solved_A, grid)
    expected_rho = jnp.where(metric.exact_exterior_points, 0.0, expected_rho)
    expected_Srr = jnp.where(metric.exact_exterior_points, 0.0, expected_Srr)

    assert jnp.allclose(metric.rho, expected_rho)
    assert jnp.allclose(metric.S_rr, expected_Srr)


def test_geodesic_terms_return_evolved_orbit_derivatives():
    grid = build_radial_grid(epsilon=0.05, r_max=1.0, num_interior_points=5)
    species = make_species()
    fields = MetricState(
        rho=jnp.zeros_like(grid.r_full),
        A=jnp.ones_like(grid.r_full),
        lapse=jnp.ones_like(grid.r_full),
        shift=jnp.zeros_like(grid.r_full),
        extrinsic_curvature=jnp.zeros_like(grid.r_full),
        S_r=jnp.zeros_like(grid.r_full),
        S_rr=jnp.zeros_like(grid.r_full),
        exact_exterior_points=jnp.ones_like(grid.r_full, dtype=bool),
    )

    dr_dt, dphi_dt, du_r_dt = compute_geodesic_terms(
        species,
        fields,
        grid,
        schwarzschild_mass=0.0,
    )

    assert dr_dt.shape == species.r.shape
    assert dphi_dt.shape == species.r.shape
    assert du_r_dt.shape == species.r.shape


def test_rk4_step_preserves_constrained_momenta_with_new_species():
    grid = build_radial_grid(epsilon=0.05, r_max=1.0, num_interior_points=5)
    species = make_species()
    fields = MetricState(
        rho=jnp.zeros_like(grid.r_full),
        A=jnp.ones_like(grid.r_full),
        lapse=jnp.ones_like(grid.r_full),
        shift=jnp.zeros_like(grid.r_full),
        extrinsic_curvature=jnp.zeros_like(grid.r_full),
        S_r=jnp.zeros_like(grid.r_full),
        S_rr=jnp.zeros_like(grid.r_full),
        exact_exterior_points=jnp.ones_like(grid.r_full, dtype=bool),
    )

    rk4_particles = rk4_step(species, fields, grid, dt=0.01, schwarzschild_mass=0.0)

    assert jnp.allclose(rk4_particles.u_phi, species.u_phi)


def test_rk4_step_has_no_separate_species_list_stepper():
    import RadiShPICR.evolve as evolve

    assert not hasattr(evolve, "_rk4_step_particle_list")


def test_rk4_step_advances_species_list_with_whole_list_charge_source(monkeypatch):
    import RadiShPICR.evolve as evolve

    grid = build_radial_grid(epsilon=0.05, r_max=1.0, num_interior_points=5)
    ions = make_species()
    electrons = make_second_species()
    particles = [ions, electrons]
    fields = MetricState(
        rho=jnp.zeros_like(grid.r_full),
        A=jnp.ones_like(grid.r_full),
        lapse=jnp.ones_like(grid.r_full),
        shift=jnp.zeros_like(grid.r_full),
        extrinsic_curvature=jnp.zeros_like(grid.r_full),
        S_r=jnp.zeros_like(grid.r_full),
        S_rr=jnp.zeros_like(grid.r_full),
        exact_exterior_points=jnp.ones_like(grid.r_full, dtype=bool),
    )
    electric_field_calls = []

    def fake_compute_geodesic_terms(
        stage_species,
        metric,
        metric_grid,
        schwarzschild_mass,
        shape_mode="nearest",
    ):
        return (
            stage_species.r,
            jnp.zeros_like(stage_species.r),
            jnp.zeros_like(stage_species.r),
        )

    def fake_compute_radial_lorentz_force_terms(
        stage_species,
        metric,
        metric_grid,
        electric_field,
        shape_mode="nearest",
    ):
        return jnp.zeros_like(stage_species.r)

    def fake_solve(particle_list, A, solve_grid, shape_mode="nearest"):
        electric_field_calls.append(tuple(species.get_name() for species in particle_list))
        return jnp.zeros_like(solve_grid.r_full), jnp.zeros_like(solve_grid.r_full)

    monkeypatch.setattr(evolve, "compute_geodesic_terms", fake_compute_geodesic_terms)
    monkeypatch.setattr(
        evolve,
        "compute_radial_lorentz_force_terms",
        fake_compute_radial_lorentz_force_terms,
    )
    monkeypatch.setattr(
        evolve,
        "compute_charge_density_and_radial_electric_field",
        fake_solve,
    )

    updated_particles = rk4_step(
        particles,
        fields,
        grid,
        dt=0.2,
        schwarzschild_mass=0.0,
    )

    assert isinstance(updated_particles, list)
    assert [species.get_name() for species in updated_particles] == ["ions", "electrons"]
    assert len(electric_field_calls) == 8
    assert all(call == ("ions", "electrons") for call in electric_field_calls)


def test_rk4_step_recomputes_electric_field_for_each_stage(monkeypatch):
    import RadiShPICR.evolve as evolve

    grid = build_radial_grid(epsilon=0.05, r_max=1.0, num_interior_points=5)
    species = particle_species(
        name="ions",
        number_of_particles=3,
        charge=2.0,
        mass=4.0,
        temperature=0.0,
        r=jnp.array([0.25, 0.50, 0.75]),
        phi=jnp.zeros(3),
        u_r=jnp.zeros(3),
        u_phi=jnp.zeros(3),
        weight=5.0,
        r_min=0.0,
        r_max=1.0,
        dr=0.25,
    )
    fields = MetricState(
        rho=jnp.zeros_like(grid.r_full),
        A=jnp.ones_like(grid.r_full),
        lapse=jnp.ones_like(grid.r_full),
        shift=jnp.zeros_like(grid.r_full),
        extrinsic_curvature=jnp.zeros_like(grid.r_full),
        S_r=jnp.zeros_like(grid.r_full),
        S_rr=jnp.zeros_like(grid.r_full),
        exact_exterior_points=jnp.ones_like(grid.r_full, dtype=bool),
    )
    calls = []

    def fake_solve(particle_list, A, solve_grid, shape_mode="nearest"):
        stage_particles = particle_list[0]
        calls.append(jnp.asarray(stage_particles.u_r))
        return jnp.zeros_like(solve_grid.r_full), jnp.ones_like(solve_grid.r_full)

    monkeypatch.setattr(
        evolve,
        "compute_charge_density_and_radial_electric_field",
        fake_solve,
    )

    updated = rk4_step(
        species,
        fields,
        grid,
        dt=0.2,
        schwarzschild_mass=0.0,
    )

    assert len(calls) == 4
    assert jnp.allclose(updated.u_r, jnp.full_like(species.u_r, 0.1))


def test_rk4_step_uses_flat_metric_derivatives_when_GR_is_false(monkeypatch):
    import RadiShPICR.evolve as evolve

    grid = build_radial_grid(epsilon=0.05, r_max=1.0, num_interior_points=5)
    species = make_species()
    fields = MetricState(
        rho=jnp.zeros_like(grid.r_full),
        A=jnp.ones_like(grid.r_full),
        lapse=jnp.ones_like(grid.r_full),
        shift=jnp.zeros_like(grid.r_full),
        extrinsic_curvature=jnp.zeros_like(grid.r_full),
        S_r=jnp.zeros_like(grid.r_full),
        S_rr=jnp.zeros_like(grid.r_full),
        exact_exterior_points=jnp.ones_like(grid.r_full, dtype=bool),
    )

    geodesic_metric_samples = []

    def fake_compute_geodesic_terms(
        stage_species,
        metric,
        metric_grid,
        schwarzschild_mass,
        shape_mode="nearest",
    ):
        geodesic_metric_samples.append((metric.A, metric.lapse, metric.shift))
        return (
            jnp.ones_like(stage_species.r),
            jnp.zeros_like(stage_species.r),
            jnp.zeros_like(stage_species.r),
        )

    def fail_lorentz_force(*args, **kwargs):
        raise AssertionError("EM=False should not compute Lorentz force")

    def fail_gauss_law(*args, **kwargs):
        raise AssertionError("EM=False should not solve Gauss law")

    monkeypatch.setattr(evolve, "compute_geodesic_terms", fake_compute_geodesic_terms)
    monkeypatch.setattr(evolve, "compute_radial_lorentz_force_terms", fail_lorentz_force)
    monkeypatch.setattr(evolve, "compute_charge_density_and_radial_electric_field", fail_gauss_law)

    updated = rk4_step(
        species,
        fields,
        grid,
        dt=0.2,
        schwarzschild_mass=0.0,
        GR=False,
        EM=False,
    )

    assert len(geodesic_metric_samples) == 4
    for metric_A, lapse, shift in geodesic_metric_samples:
        assert jnp.allclose(metric_A, 1.0)
        assert jnp.allclose(lapse, 1.0)
        assert jnp.allclose(shift, 0.0)

    assert jnp.allclose(updated.r, species.r + 0.2)
    assert jnp.allclose(updated.phi, species.phi)
    assert jnp.allclose(updated.u_r, species.u_r)


def test_rk4_step_keeps_geodesics_but_skips_em_when_EM_is_false(monkeypatch):
    import RadiShPICR.evolve as evolve

    grid = build_radial_grid(epsilon=0.05, r_max=1.0, num_interior_points=5)
    species = make_species()
    fields = MetricState(
        rho=jnp.zeros_like(grid.r_full),
        A=jnp.ones_like(grid.r_full),
        lapse=jnp.ones_like(grid.r_full),
        shift=jnp.zeros_like(grid.r_full),
        extrinsic_curvature=jnp.zeros_like(grid.r_full),
        S_r=jnp.zeros_like(grid.r_full),
        S_rr=jnp.zeros_like(grid.r_full),
        exact_exterior_points=jnp.ones_like(grid.r_full, dtype=bool),
    )
    geodesic_calls = []

    def fake_compute_geodesic_terms(
        stage_species,
        metric,
        metric_grid,
        schwarzschild_mass,
        shape_mode="nearest",
    ):
        geodesic_calls.append(jnp.asarray(stage_species.r))
        return (
            jnp.ones_like(stage_species.r),
            jnp.zeros_like(stage_species.r),
            jnp.zeros_like(stage_species.r),
        )

    def fail_lorentz_force(*args, **kwargs):
        raise AssertionError("EM=False should not compute Lorentz force")

    def fail_gauss_law(*args, **kwargs):
        raise AssertionError("EM=False should not solve Gauss law")

    monkeypatch.setattr(evolve, "compute_geodesic_terms", fake_compute_geodesic_terms)
    monkeypatch.setattr(evolve, "compute_radial_lorentz_force_terms", fail_lorentz_force)
    monkeypatch.setattr(evolve, "compute_charge_density_and_radial_electric_field", fail_gauss_law)

    updated = rk4_step(
        species,
        fields,
        grid,
        dt=0.2,
        schwarzschild_mass=0.0,
        EM=False,
    )

    assert len(geodesic_calls) == 4
    assert jnp.allclose(updated.r, species.r + 0.2)
    assert jnp.allclose(updated.u_r, species.u_r)


def test_advance_one_step_recomputes_metric_and_electric_field_for_dynamic_rk4(monkeypatch):
    import RadiShPICR.evolve as evolve

    grid = build_radial_grid(epsilon=0.05, r_max=1.0, num_interior_points=5)
    species = particle_species(
        name="ions",
        number_of_particles=3,
        charge=2.0,
        mass=4.0,
        temperature=0.0,
        r=jnp.array([0.25, 0.50, 0.75]),
        phi=jnp.zeros(3),
        u_r=jnp.zeros(3),
        u_phi=jnp.zeros(3),
        weight=5.0,
        r_min=0.0,
        r_max=1.0,
        dr=0.25,
    )
    metric_calls = []
    metric_solver_calls = []
    electric_field_calls = []

    def fake_compute_metric(
        stage_particles,
        metric_grid,
        schwarzschild_mass,
        initial_A_guess=None,
        shape_mode="nearest",
        metric_A_solver="newton",
        EM=True,
    ):
        assert EM is True
        if isinstance(stage_particles, (list, tuple)):
            stage_particles = stage_particles[0]
        stage_radius_mean = jnp.mean(stage_particles.r)
        metric_calls.append(stage_radius_mean)
        metric_solver_calls.append(metric_A_solver)
        stage_A = jnp.full_like(metric_grid.r_full, stage_radius_mean)

        return MetricState(
            rho=jnp.zeros_like(metric_grid.r_full),
            A=stage_A,
            lapse=jnp.ones_like(metric_grid.r_full),
            shift=jnp.zeros_like(metric_grid.r_full),
            extrinsic_curvature=jnp.zeros_like(metric_grid.r_full),
            S_r=jnp.zeros_like(metric_grid.r_full),
            S_rr=jnp.zeros_like(metric_grid.r_full),
            exact_exterior_points=jnp.ones_like(metric_grid.r_full, dtype=bool),
        )

    def fake_compute_geodesic_terms(
        stage_particles,
        metric,
        metric_grid,
        schwarzschild_mass,
        shape_mode="nearest",
    ):
        return (
            stage_particles.r,
            jnp.zeros_like(stage_particles.r),
            jnp.zeros_like(stage_particles.r),
        )

    def fake_compute_radial_lorentz_force_terms(
        stage_particles,
        metric,
        metric_grid,
        electric_field,
        shape_mode="nearest",
    ):
        return jnp.zeros_like(stage_particles.r)

    def fake_solve(particle_list, A, solve_grid, shape_mode="nearest"):
        stage_particles = particle_list[0]
        electric_field_calls.append((jnp.mean(stage_particles.r), A[0]))
        return jnp.zeros_like(solve_grid.r_full), jnp.zeros_like(solve_grid.r_full)

    monkeypatch.setattr(evolve, "compute_metric", fake_compute_metric)
    monkeypatch.setattr(evolve, "compute_geodesic_terms", fake_compute_geodesic_terms)
    monkeypatch.setattr(
        evolve,
        "compute_radial_lorentz_force_terms",
        fake_compute_radial_lorentz_force_terms,
    )
    monkeypatch.setattr(
        evolve,
        "compute_charge_density_and_radial_electric_field",
        fake_solve,
    )

    updated_particles, returned_fields = advance_one_step(
        species,
        grid,
        dt=0.2,
        schwarzschild_mass=0.0,
        metric_A_solver="broyden",
    )

    expected_stage_radius_means = jnp.array([0.5, 0.55, 0.555, 0.611])
    expected_final_radius_mean = jnp.mean(updated_particles.r)

    assert len(metric_calls) == 5
    assert metric_solver_calls == ["broyden"] * 5
    assert jnp.allclose(jnp.asarray(metric_calls[:4]), expected_stage_radius_means)
    assert jnp.allclose(metric_calls[-1], expected_final_radius_mean)
    assert len(electric_field_calls) == 4
    assert jnp.allclose(
        jnp.asarray([call[0] for call in electric_field_calls]),
        expected_stage_radius_means,
    )
    assert jnp.allclose(
        jnp.asarray([call[1] for call in electric_field_calls]),
        expected_stage_radius_means,
    )
    assert jnp.allclose(returned_fields.A, expected_final_radius_mean)




def test_advance_one_step_advances_species_list_with_shared_stage_fields(monkeypatch):
    import RadiShPICR.evolve as evolve

    grid = build_radial_grid(epsilon=0.05, r_max=1.0, num_interior_points=5)
    ions = make_species()
    electrons = make_second_species()
    particles = [ions, electrons]
    metric_calls = []
    electric_field_calls = []

    def collection_radius_mean(stage_particles):
        return jnp.mean(jnp.concatenate([species.r for species in stage_particles]))

    def fake_compute_metric(
        stage_particles,
        metric_grid,
        schwarzschild_mass,
        initial_A_guess=None,
        shape_mode="nearest",
        metric_A_solver="newton",
        EM=True,
    ):
        assert EM is True
        stage_radius_mean = collection_radius_mean(stage_particles)
        metric_calls.append(stage_radius_mean)
        stage_A = jnp.full_like(metric_grid.r_full, stage_radius_mean)

        return MetricState(
            rho=jnp.zeros_like(metric_grid.r_full),
            A=stage_A,
            lapse=jnp.ones_like(metric_grid.r_full),
            shift=jnp.zeros_like(metric_grid.r_full),
            extrinsic_curvature=jnp.zeros_like(metric_grid.r_full),
            S_r=jnp.zeros_like(metric_grid.r_full),
            S_rr=jnp.zeros_like(metric_grid.r_full),
            exact_exterior_points=jnp.ones_like(metric_grid.r_full, dtype=bool),
        )

    def fake_compute_geodesic_terms(
        stage_species,
        metric,
        metric_grid,
        schwarzschild_mass,
        shape_mode="nearest",
    ):
        return (
            stage_species.r,
            jnp.zeros_like(stage_species.r),
            jnp.zeros_like(stage_species.r),
        )

    def fake_compute_radial_lorentz_force_terms(
        stage_species,
        metric,
        metric_grid,
        electric_field,
        shape_mode="nearest",
    ):
        return jnp.zeros_like(stage_species.r)

    def fake_solve(particle_list, A, solve_grid, shape_mode="nearest"):
        electric_field_calls.append(
            (
                tuple(species.get_name() for species in particle_list),
                collection_radius_mean(particle_list),
                A[0],
            )
        )
        return jnp.zeros_like(solve_grid.r_full), jnp.zeros_like(solve_grid.r_full)

    monkeypatch.setattr(evolve, "compute_metric", fake_compute_metric)
    monkeypatch.setattr(evolve, "compute_geodesic_terms", fake_compute_geodesic_terms)
    monkeypatch.setattr(
        evolve,
        "compute_radial_lorentz_force_terms",
        fake_compute_radial_lorentz_force_terms,
    )
    monkeypatch.setattr(
        evolve,
        "compute_charge_density_and_radial_electric_field",
        fake_solve,
    )

    updated_particles, returned_fields = advance_one_step(
        particles,
        grid,
        dt=0.2,
        schwarzschild_mass=0.0,
    )

    assert isinstance(updated_particles, list)
    assert [species.get_name() for species in updated_particles] == ["ions", "electrons"]
    assert len(metric_calls) == 8
    assert len(electric_field_calls) == 8
    assert all(call[0] == ("ions", "electrons") for call in electric_field_calls)
    assert jnp.allclose(returned_fields.A, collection_radius_mean(updated_particles))


def test_advance_one_step_uses_flat_fields_when_GR_is_false(monkeypatch):
    import RadiShPICR.evolve as evolve

    grid = build_radial_grid(epsilon=0.05, r_max=1.0, num_interior_points=5)
    species = make_species()

    def fail_compute_metric(*args, **kwargs):
        raise AssertionError("GR=False should not solve the metric")

    def fake_compute_geodesic_terms(
        stage_species,
        metric,
        metric_grid,
        schwarzschild_mass,
        shape_mode="nearest",
    ):
        assert jnp.allclose(metric.A, 1.0)
        assert jnp.allclose(metric.lapse, 1.0)
        assert jnp.allclose(metric.shift, 0.0)
        return (
            jnp.zeros_like(stage_species.r),
            jnp.zeros_like(stage_species.r),
            jnp.zeros_like(stage_species.r),
        )

    monkeypatch.setattr(evolve, "compute_metric", fail_compute_metric)
    monkeypatch.setattr(evolve, "compute_geodesic_terms", fake_compute_geodesic_terms)

    updated_particles, returned_fields = advance_one_step(
        species,
        grid,
        dt=0.2,
        schwarzschild_mass=0.0,
        GR=False,
        EM=False,
    )

    assert jnp.allclose(updated_particles.r, species.r)
    assert jnp.allclose(updated_particles.phi, species.phi)
    assert jnp.allclose(updated_particles.u_r, species.u_r)
    assert jnp.allclose(returned_fields.rho, 0.0)
    assert jnp.allclose(returned_fields.A, 1.0)
    assert jnp.allclose(returned_fields.lapse, 1.0)
    assert jnp.allclose(returned_fields.shift, 0.0)
    assert jnp.allclose(returned_fields.extrinsic_curvature, 0.0)
    assert jnp.allclose(returned_fields.S_r, 0.0)
    assert jnp.allclose(returned_fields.S_rr, 0.0)
    assert jnp.all(returned_fields.exact_exterior_points)


def test_advance_one_step_is_rk4_only_api():
    assert "integrator" not in inspect.signature(advance_one_step).parameters
