import jax.numpy as jnp

from RadiShPICR.EM import compute_radial_lorentz_force_terms
from RadiShPICR.particles.particle_species import particle_species
from RadiShPICR.relativity.grid import build_radial_grid
from RadiShPICR.relativity.metric import MetricState


def make_metric(grid, lapse):
    return MetricState(
        rho=jnp.zeros_like(grid.r_full),
        A=jnp.ones_like(grid.r_full),
        lapse=lapse,
        shift=jnp.zeros_like(grid.r_full),
        extrinsic_curvature=jnp.zeros_like(grid.r_full),
        S_r=jnp.zeros_like(grid.r_full),
        S_rr=jnp.zeros_like(grid.r_full),
        exact_exterior_points=jnp.ones_like(grid.r_full, dtype=bool),
    )


def make_species(charge=2.0, mass=4.0, weight=3.0):
    return particle_species(
        name="charged-dust",
        number_of_particles=2,
        charge=charge,
        mass=mass,
        temperature=0.0,
        r=jnp.array([0.25, 0.75]),
        phi=jnp.zeros(2),
        u_r=jnp.zeros(2),
        u_phi=jnp.zeros(2),
        weight=weight,
        r_min=0.0,
        r_max=1.0,
        dr=0.25,
    )


def test_lorentz_force_interpolates_field_and_scales_by_lapse_charge_to_mass():
    grid = build_radial_grid(epsilon=0.05, r_max=1.0, num_interior_points=5)
    species = make_species(charge=2.0, mass=4.0, weight=9.0)
    metric = make_metric(grid, lapse=jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    electric_field = jnp.array([0.0, 10.0, 20.0, 30.0, 40.0])

    du_r_dt = compute_radial_lorentz_force_terms(
        species,
        metric,
        grid,
        electric_field,
    )

    expected = jnp.array([2.0, 4.0]) * (2.0 / 4.0) * jnp.array([10.0, 30.0])
    assert jnp.allclose(du_r_dt, expected)


def test_lorentz_force_uses_physical_charge_mass_not_weighted_metadata():
    grid = build_radial_grid(epsilon=0.05, r_max=1.0, num_interior_points=5)
    unit_weight_species = make_species(charge=2.0, mass=4.0, weight=1.0)
    weighted_species = make_species(charge=2.0, mass=4.0, weight=7.0)
    metric = make_metric(grid, lapse=jnp.ones_like(grid.r_full))
    electric_field = jnp.array([0.0, 10.0, 20.0, 30.0, 40.0])

    unit_du_r_dt = compute_radial_lorentz_force_terms(
        unit_weight_species,
        metric,
        grid,
        electric_field,
    )
    weighted_du_r_dt = compute_radial_lorentz_force_terms(
        weighted_species,
        metric,
        grid,
        electric_field,
    )

    assert jnp.allclose(weighted_du_r_dt, unit_du_r_dt)


def test_neutral_particles_have_zero_lorentz_force():
    grid = build_radial_grid(epsilon=0.05, r_max=1.0, num_interior_points=5)
    species = make_species(charge=0.0, mass=4.0)
    metric = make_metric(grid, lapse=jnp.ones_like(grid.r_full))
    electric_field = jnp.array([0.0, 10.0, 20.0, 30.0, 40.0])

    du_r_dt = compute_radial_lorentz_force_terms(
        species,
        metric,
        grid,
        electric_field,
    )

    assert jnp.allclose(du_r_dt, 0.0)
