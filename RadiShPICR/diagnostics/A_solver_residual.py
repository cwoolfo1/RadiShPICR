from pathlib import Path

import jax.numpy as jnp
import numpy as np

from RadiShPICR.relativity.A import nonlinear_residual_u
from RadiShPICR.relativity.schwarzschild import schwarzschild_u


def compute_A_solver_residual(metric, grid, schwarzschild_mass):
    """Compute the A-solver residual for the current metric state.

    The A solver advances ``U = sqrt(A)`` through the polar-gauge Hamiltonian
    constraint.  This diagnostic reconstructs the same discrete residual after
    the solve so the output measures the constraint error on the simulation grid.
    """

    U = jnp.sqrt(metric.A)
    source_term = -2.0 * jnp.pi * metric.rho
    boundary_u = schwarzschild_u(grid.r_full, schwarzschild_mass, grid.epsilon)

    residual = nonlinear_residual_u(
        U,
        source_term,
        grid,
        boundary_u,
        metric.exact_exterior_points,
    )

    return residual


def write_A_solver_residual(
    metric,
    grid,
    schwarzschild_mass,
    output_folder,
    step,
    time=None,
):
    """Write one A-solver Hamiltonian-constraint residual snapshot."""

    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    filename = f"A_solver_residual_step_{int(step):06d}.npz"
    snapshot_path = output_path / filename

    residual = compute_A_solver_residual(metric, grid, schwarzschild_mass)
    residual_array = np.asarray(residual)
    output_time = np.nan if time is None else float(time)

    np.savez_compressed(
        snapshot_path,
        residual=residual_array,
        residual_norm_inf=np.linalg.norm(residual_array, ord=np.inf),
        r=np.asarray(grid.r_full),
        A=np.asarray(metric.A),
        rho=np.asarray(metric.rho),
        exact_exterior_points=np.asarray(metric.exact_exterior_points),
        step=int(step),
        time=output_time,
    )

    return snapshot_path
