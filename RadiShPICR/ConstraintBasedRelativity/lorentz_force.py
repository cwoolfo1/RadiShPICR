from RadiShPICR.deposition.particle_shapes import interpolate_field_to_particles
from RadiShPICR.ConstraintBasedRelativity.grid import RadialGrid


def _field_interpolation_grid(r_grid):
    dr_grid = r_grid[1] - r_grid[0]
    return RadialGrid(
        r_full=r_grid,
        r_interior=r_grid,
        dr=dr_grid,
        epsilon=0.5 * dr_grid,
        r_max=r_grid[-1],
    )


def compute_lorentz_terms(particles, U_state):
    A_values, phi_values, alpha_values, Krr_values, beta_over_r_values, Er_values, source_terms, r_grid = U_state
    r, _ = particles.get_positions()
    shape_mode = particles.get_shape()
    interpolation_grid = _field_interpolation_grid(r_grid)

    lapse_at_particle = interpolate_field_to_particles(
        alpha_values,
        r,
        interpolation_grid,
        shape_mode=shape_mode,
    )
    electric_field_at_particle = interpolate_field_to_particles(
        Er_values,
        r,
        interpolation_grid,
        shape_mode=shape_mode,
    )

    return lapse_at_particle * particles.get_charge() * electric_field_at_particle / particles.get_mass()
