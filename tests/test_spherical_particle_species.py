import jax
import jax.numpy as jnp

from RadiShPICR.particles.particle_species import particle_species
from RadiShPICR.relativity.matter_source_terms import (
    compute_Sr,
    compute_Srr,
    compute_density_and_metric_derivative,
)
from RadiShPICR.relativity.evolve import euler_step, rk4_step
from RadiShPICR.relativity.states import FieldState
from RadiShPICR.relativity.utils import build_radial_grid


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
    u_r, u_theta, u_phi = species.get_velocity()
    assert jnp.allclose(r, species.r)
    assert jnp.allclose(phi, species.phi)
    assert jnp.allclose(u_r, species.u_r)
    assert jnp.allclose(u_theta, species.u_theta)
    assert jnp.allclose(u_phi, species.u_phi)
    assert jnp.all(species.u_theta == 0.0)


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
    assert jnp.allclose(updated.u_theta, species.u_theta)
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
    assert jnp.allclose(rebuilt.u_theta, species.u_theta)
    assert jnp.allclose(rebuilt.u_phi, species.u_phi)


def test_scalar_mass_broadcasts_in_source_terms():
    grid = build_radial_grid(epsilon=0.05, r_max=1.0, num_interior_points=5)
    species = make_species()
    A = jnp.ones_like(grid.r_full)

    rho, _ = compute_density_and_metric_derivative(species, A, grid)
    Sr_from_species = compute_Sr(species, A, grid)
    Srr_from_species = compute_Srr(species, A, grid)

    expected_masses = jnp.full(species.r.shape, species.get_mass())
    expected_Sr = compute_Sr(expected_masses, species.u_r, species.r, A, grid)
    expected_Srr = compute_Srr(expected_masses, species.u_r, species.u_phi, species.r, A, grid)

    assert rho.shape == grid.r_full.shape
    assert jnp.allclose(Sr_from_species, expected_Sr)
    assert jnp.allclose(Srr_from_species, expected_Srr)


def test_relativity_steps_preserve_constrained_momenta_with_new_species():
    grid = build_radial_grid(epsilon=0.05, r_max=1.0, num_interior_points=5)
    species = make_species()
    fields = FieldState(
        rho=jnp.zeros_like(grid.r_full),
        A=jnp.ones_like(grid.r_full),
        lapse=jnp.ones_like(grid.r_full),
        shift=jnp.zeros_like(grid.r_full),
        extrinsic_curvature=jnp.zeros_like(grid.r_full),
        S_r=jnp.zeros_like(grid.r_full),
        S_rr=jnp.zeros_like(grid.r_full),
    )

    euler_particles = euler_step(species, fields, grid, dt=0.01, schwarzschild_mass=0.0)
    rk4_particles = rk4_step(species, fields, grid, dt=0.01, schwarzschild_mass=0.0)

    assert jnp.allclose(euler_particles.u_theta, species.u_theta)
    assert jnp.allclose(euler_particles.u_phi, species.u_phi)
    assert jnp.allclose(rk4_particles.u_theta, species.u_theta)
    assert jnp.allclose(rk4_particles.u_phi, species.u_phi)
