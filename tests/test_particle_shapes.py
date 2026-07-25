import jax
import jax.numpy as jnp

from RadiShPICR.ConstraintBasedRelativity.charge_density import charge_density_at_point
from RadiShPICR.ConstraintBasedRelativity.grid import RadialGrid
from RadiShPICR.ConstraintBasedRelativity.mass_density import mass_density_at_point
from RadiShPICR.particles import particle_species
from RadiShPICR.particles.particle_shapes import (
    radial_shape_stencil,
    shape_weights_at_point,
)


def make_grid(r_max=8.0, dr=1.0):
    r_grid = jnp.arange(0.0, r_max + dr, dr)
    return RadialGrid(
        r_full=r_grid,
        r_interior=r_grid,
        dr=dr,
        epsilon=0.5 * dr,
        r_max=r_max,
    )


def test_quadratic_shape_has_expected_tsc_weights():
    radial_grid_points = jnp.asarray([3.0, 4.0, 5.0])
    particle_positions = (
        jnp.asarray([4.0]),
        jnp.asarray([4.25]),
        jnp.asarray([4.5]),
    )
    expected_weights = (
        jnp.asarray([0.125, 0.75, 0.125]),
        jnp.asarray([0.03125, 0.6875, 0.28125]),
        jnp.asarray([0.0, 0.5, 0.5]),
    )

    for radial_position, expected in zip(particle_positions, expected_weights):
        weights = shape_weights_at_point(
            radial_position,
            radial_grid_points[:, jnp.newaxis],
            dr=1.0,
            shape_mode="quadratic",
        )

        assert jnp.allclose(weights[:, 0], expected)
        assert jnp.all(weights >= 0.0)
        assert jnp.allclose(jnp.sum(weights), 1.0)


def test_quadratic_pointwise_weights_match_indexed_stencil():
    grid = make_grid()
    radial_positions = jnp.asarray([3.0, 3.25, 3.5, 5.75])

    indices, stencil_weights = radial_shape_stencil(
        radial_positions,
        grid,
        shape_mode="quadratic",
    )
    particle_columns = jnp.broadcast_to(
        jnp.arange(radial_positions.shape[0])[jnp.newaxis, :],
        indices.shape,
    )
    indexed_weights = jnp.zeros(
        (grid.r_full.shape[0], radial_positions.shape[0])
    )
    indexed_weights = indexed_weights.at[indices, particle_columns].add(
        stencil_weights
    )

    pointwise_weights = shape_weights_at_point(
        radial_positions[jnp.newaxis, :],
        grid.r_full[:, jnp.newaxis],
        grid.dr,
        shape_mode="quadratic",
    )

    assert jnp.allclose(pointwise_weights, indexed_weights)
    assert jnp.allclose(jnp.sum(pointwise_weights, axis=0), 1.0)


def test_quadratic_source_deposition_conserves_particle_mass_and_charge():
    grid = make_grid()
    particles = particle_species(
        name="test",
        charge=3.0,
        mass=2.0,
        weight=0.25,
        r=jnp.asarray([4.25]),
        ur=jnp.asarray([0.0]),
        phi=jnp.asarray([0.0]),
        uphi=jnp.asarray([0.0]),
        shape_mode="quadratic",
    )

    mass_density = jax.vmap(
        lambda r: mass_density_at_point(particles, jnp.asarray(1.0), r, grid.dr)
    )(grid.r_full)
    charge_density = jax.vmap(
        lambda r: charge_density_at_point(particles, jnp.asarray(1.0), r, grid.dr)
    )(grid.r_full)

    safe_radius = jnp.maximum(grid.r_full, 0.5 * grid.dr)
    cell_volume = 4.0 * jnp.pi * safe_radius**2 * grid.dr
    deposited_mass = jnp.sum(mass_density * cell_volume)
    deposited_charge = jnp.sum(charge_density * cell_volume)

    assert jnp.allclose(deposited_mass, particles.get_mass())
    assert jnp.allclose(deposited_charge, particles.get_charge())


def test_quadratic_shape_jit_matches_eager_and_preserves_boundary_stencil():
    grid = make_grid()
    radial_positions = jnp.asarray([0.25, 4.25, 7.75])
    radial_coordinate = jnp.asarray(4.0)

    eager_weights = shape_weights_at_point(
        radial_positions,
        radial_coordinate,
        grid.dr,
        shape_mode="quadratic",
    )
    jitted_shape_weights = jax.jit(
        shape_weights_at_point,
        static_argnames=("shape_mode",),
    )
    jitted_weights = jitted_shape_weights(
        radial_positions,
        radial_coordinate,
        grid.dr,
        shape_mode="quadratic",
    )
    indices, stencil_weights = radial_shape_stencil(
        radial_positions,
        grid,
        shape_mode="quadratic",
    )

    assert jnp.allclose(jitted_weights, eager_weights)
    assert jnp.all(indices >= 1)
    assert jnp.all(indices <= grid.r_full.shape[0] - 2)
    assert jnp.allclose(jnp.sum(stencil_weights, axis=0), 1.0)
