import numpy as np
import jax.numpy as jnp

from RadiShPICR.diagnostics import write_A_solver_residual, write_phase_space
from RadiShPICR.particles.particle_species import particle_species
from RadiShPICR.relativity.A import nonlinear_residual_u
from RadiShPICR.relativity.grid import build_radial_grid
from RadiShPICR.relativity.metric import MetricState
from RadiShPICR.relativity.schwarzschild import schwarzschild_u


def test_write_phase_space_writes_radial_snapshot(tmp_path):
    particles = particle_species(
        name="ions",
        number_of_particles=3,
        charge=1.0,
        mass=2.0,
        temperature=0.0,
        r=jnp.array([0.2, 0.4, 0.6]),
        phi=jnp.zeros(3),
        u_r=jnp.array([0.01, -0.02, 0.03]),
        u_phi=jnp.zeros(3),
    )

    output_path = write_phase_space(
        particles,
        tmp_path,
        step=12,
        time=0.6,
    )

    assert output_path.name == "phase_space_ions_step_000012.npz"
    assert output_path.parent == tmp_path

    with np.load(output_path) as snapshot:
        assert np.allclose(snapshot["r"], np.asarray(particles.r))
        assert np.allclose(snapshot["vr"], np.asarray(particles.u_r))
        assert int(snapshot["step"]) == 12
        assert np.isclose(float(snapshot["time"]), 0.6)
        assert str(snapshot["species_name"]) == "ions"


def test_write_A_solver_residual_writes_hamiltonian_constraint_error(tmp_path):
    grid = build_radial_grid(epsilon=0.0, r_max=1.0, num_interior_points=5)
    schwarzschild_mass = 0.1
    A = jnp.array([1.00, 1.04, 1.09, 1.13, 1.20])
    rho = jnp.array([0.0, 0.03, 0.02, 0.0, 0.0])
    exact_exterior_points = jnp.array([False, False, False, False, True])
    metric = MetricState(
        rho=rho,
        A=A,
        lapse=jnp.ones_like(A),
        shift=jnp.zeros_like(A),
        extrinsic_curvature=jnp.zeros_like(A),
        S_r=jnp.zeros_like(A),
        S_rr=jnp.zeros_like(A),
        exact_exterior_points=exact_exterior_points,
    )

    output_path = write_A_solver_residual(
        metric,
        grid,
        schwarzschild_mass,
        tmp_path,
        step=7,
        time=0.35,
    )

    expected_residual = nonlinear_residual_u(
        jnp.sqrt(A),
        -2.0 * jnp.pi * rho,
        grid,
        schwarzschild_u(grid.r_full, schwarzschild_mass, grid.epsilon),
        exact_exterior_points,
    )

    assert output_path.name == "A_solver_residual_step_000007.npz"
    with np.load(output_path) as snapshot:
        assert np.allclose(snapshot["residual"], np.asarray(expected_residual))
        assert np.isclose(
            float(snapshot["residual_norm_inf"]),
            np.linalg.norm(np.asarray(expected_residual), ord=np.inf),
        )
        assert int(snapshot["step"]) == 7
        assert np.isclose(float(snapshot["time"]), 0.35)
