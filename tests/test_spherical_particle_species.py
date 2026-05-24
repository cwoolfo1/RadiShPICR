import inspect

import jax
import jax.numpy as jnp

from RadiShPICR.particles.particle_species import particle_species
from RadiShPICR.deposition import (
    compute_charge_density,
    compute_charge_density_metric_derivative,
    compute_charge_density_metric_jacobian,
    compute_mass_density,
    compute_mass_density_metric_derivative,
    compute_mass_density_metric_jacobian,
    compute_number_density,
    compute_number_density_metric_derivative,
    compute_number_density_metric_jacobian,
)
from RadiShPICR.relativity.energy_momentum import (
    compute_Sr,
    compute_Srr,
)
from RadiShPICR.evolve import advance_one_step, rk4_step
from RadiShPICR.relativity.geodesic import compute_geodesic_terms
from RadiShPICR.relativity.grid import build_radial_grid
from RadiShPICR.relativity.metric import MetricState


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


def test_density_helpers_scale_number_density_by_scalar_metadata():
    grid = build_radial_grid(epsilon=0.05, r_max=1.0, num_interior_points=5)
    species = make_species()
    A = jnp.ones_like(grid.r_full)

    for shape_mode in ("nearest", "quadratic"):
        number_density = compute_number_density(species, A, grid, shape_mode=shape_mode)
        mass_density = compute_mass_density(species, A, grid, shape_mode=shape_mode)
        charge_density = compute_charge_density(species, A, grid, shape_mode=shape_mode)
        dn_dA = compute_number_density_metric_derivative(species, A, grid, shape_mode=shape_mode)
        drho_dA = compute_mass_density_metric_derivative(species, A, grid, shape_mode=shape_mode)
        dq_dA = compute_charge_density_metric_derivative(species, A, grid, shape_mode=shape_mode)
        number_jacobian = compute_number_density_metric_jacobian(species, A, grid, shape_mode=shape_mode)
        mass_jacobian = compute_mass_density_metric_jacobian(species, A, grid, shape_mode=shape_mode)
        charge_jacobian = compute_charge_density_metric_jacobian(species, A, grid, shape_mode=shape_mode)

        assert number_density.shape == grid.r_full.shape
        assert jnp.allclose(mass_density, species.get_mass() * number_density)
        assert jnp.allclose(charge_density, species.get_charge() * number_density)
        assert jnp.allclose(drho_dA, species.get_mass() * dn_dA)
        assert jnp.allclose(dq_dA, species.get_charge() * dn_dA)
        assert jnp.allclose(mass_jacobian, species.get_mass() * number_jacobian)
        assert jnp.allclose(charge_jacobian, species.get_charge() * number_jacobian)


def test_scalar_mass_broadcasts_in_source_terms():
    grid = build_radial_grid(epsilon=0.05, r_max=1.0, num_interior_points=5)
    species = make_species()
    A = jnp.ones_like(grid.r_full)

    for shape_mode in ("nearest", "quadratic"):
        Sr_from_species = compute_Sr(species, A, grid, shape_mode=shape_mode)
        Srr_from_species = compute_Srr(species, A, grid, shape_mode=shape_mode)

        assert Sr_from_species.shape == grid.r_full.shape
        assert Srr_from_species.shape == grid.r_full.shape


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

    derivatives = compute_geodesic_terms(
        species,
        fields,
        grid,
        schwarzschild_mass=0.0,
    )

    assert derivatives.dr_dt.shape == species.r.shape
    assert derivatives.dphi_dt.shape == species.r.shape
    assert derivatives.du_r_dt.shape == species.r.shape


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


def test_advance_one_step_is_rk4_only_api():
    assert "integrator" not in inspect.signature(advance_one_step).parameters
