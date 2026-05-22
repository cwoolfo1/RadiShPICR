import jax.numpy as jnp

from RadiShPICR.evolve import rk4_step
from RadiShPICR.particles.particle_species import particle_species
from RadiShPICR.relativity.geodesic import compute_geodesic_terms
from RadiShPICR.relativity.grid import build_radial_grid
from RadiShPICR.relativity.metric import MetricState
from RadiShPICR.relativity.schwarzschild import (
    build_static_schwarzschild_metric,
    schwarzschild_A,
    schwarzschild_dA_dr,
    schwarzschild_d_lapse_dr,
    schwarzschild_lapse,
)
from RadiShPICR.relativity.utils import centered_first_derivative


def make_single_particle(radius, radial_momentum=0.0, azimuthal_momentum=0.0):
    return particle_species(
        name="test-particle",
        number_of_particles=1,
        charge=0.0,
        mass=1.0,
        temperature=0.0,
        r=jnp.asarray([radius]),
        phi=jnp.asarray([0.0]),
        u_r=jnp.asarray([radial_momentum]),
        u_phi=jnp.asarray([azimuthal_momentum]),
        weight=1.0,
        r_min=0.0,
        r_max=20.0,
        dr=0.1,
    )


def test_schwarzschild_metric_helpers_match_finite_difference_derivatives():
    grid = build_radial_grid(epsilon=0.05, r_max=20.0, num_interior_points=401)
    mass = 1.0

    metric_A = schwarzschild_A(grid.r_full, mass, grid.epsilon)
    lapse = schwarzschild_lapse(grid.r_full, mass, grid.epsilon)

    finite_difference_dA_dr = centered_first_derivative(metric_A, grid.dr)
    finite_difference_d_lapse_dr = centered_first_derivative(lapse, grid.dr)

    interior = slice(20, -20)
    assert jnp.allclose(
        schwarzschild_dA_dr(grid.r_full, mass, grid.epsilon)[interior],
        finite_difference_dA_dr[interior],
        rtol=2.0e-3,
        atol=2.0e-4,
    )
    assert jnp.allclose(
        schwarzschild_d_lapse_dr(grid.r_full, mass, grid.epsilon)[interior],
        finite_difference_d_lapse_dr[interior],
        rtol=2.0e-3,
        atol=2.0e-4,
    )


def test_static_schwarzschild_metric_has_vacuum_sources_and_zero_shift():
    grid = build_radial_grid(epsilon=0.05, r_max=20.0, num_interior_points=101)
    metric = build_static_schwarzschild_metric(grid, schwarzschild_mass=1.0)

    assert isinstance(metric, MetricState)
    assert jnp.allclose(metric.rho, 0.0)
    assert jnp.allclose(metric.S_r, 0.0)
    assert jnp.allclose(metric.S_rr, 0.0)
    assert jnp.allclose(metric.shift, 0.0)
    assert jnp.allclose(metric.extrinsic_curvature, 0.0)
    assert jnp.all(metric.exact_exterior_points)


def test_static_schwarzschild_circular_geodesic_has_stationary_radius():
    grid = build_radial_grid(epsilon=0.05, r_max=40.0, num_interior_points=801)
    mass = 1.0
    metric = build_static_schwarzschild_metric(grid, schwarzschild_mass=mass)
    radius = 10.0

    metric_A = schwarzschild_A(jnp.asarray(radius), mass, grid.epsilon)
    lapse = schwarzschild_lapse(jnp.asarray(radius), mass, grid.epsilon)
    dA_dr = schwarzschild_dA_dr(jnp.asarray(radius), mass, grid.epsilon)
    d_lapse_dr = schwarzschild_d_lapse_dr(jnp.asarray(radius), mass, grid.epsilon)

    angular_force_factor = (
        1.0 / (radius**3 * metric_A**2)
        + dA_dr / (radius**2 * metric_A**3)
    )
    angular_momentum_squared = d_lapse_dr / (
        lapse * angular_force_factor - d_lapse_dr / (radius**2 * metric_A**2)
    )
    particles = make_single_particle(
        radius=radius,
        radial_momentum=0.0,
        azimuthal_momentum=jnp.sqrt(angular_momentum_squared),
    )

    derivatives = compute_geodesic_terms(particles, metric, grid, mass)

    assert jnp.allclose(derivatives.dr_dt, 0.0, atol=1.0e-10)
    assert jnp.allclose(derivatives.du_r_dt, 0.0, atol=1.0e-5)


def test_static_metric_particle_step_preserves_azimuthal_momentum():
    grid = build_radial_grid(epsilon=0.05, r_max=20.0, num_interior_points=401)
    mass = 1.0
    metric = build_static_schwarzschild_metric(grid, schwarzschild_mass=mass)
    particles = make_single_particle(
        radius=8.0,
        radial_momentum=-0.01,
        azimuthal_momentum=2.5,
    )

    updated = rk4_step(particles, metric, grid, dt=0.01, schwarzschild_mass=mass)

    assert jnp.allclose(updated.u_phi, particles.u_phi)
    assert updated.r.shape == particles.r.shape
    assert updated.phi.shape == particles.phi.shape
    assert updated.u_r.shape == particles.u_r.shape
