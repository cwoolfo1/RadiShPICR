from functools import partial

import jax
import jax.numpy as jnp

from RadiShPICR.EM.EM_energy_momentum_tensor import compute_EM_energy_density
from RadiShPICR.EM.gauss_law import compute_charge_density_and_radial_electric_field
from RadiShPICR.deposition import (
    last_shape_support_index,
)
from RadiShPICR.deposition.number_density import (
    compute_number_density,
)
from RadiShPICR.relativity.schwarzschild import schwarzschild_A, schwarzschild_u
from RadiShPICR.relativity.utils import (
    centered_first_derivative,
    centered_second_derivative,
    compute_metric_radial_derivative,
    safe_metric_A,
    safe_radius,
)


def nonlinear_residual_u( U, source_term, grid, boundary_u, exact_exterior_points ):
    """Residual for the nonlinear polar-gauge ``U = sqrt(A)`` equation.

    The regular center uses the same second-order Neumann stencil as the
    Newton matrix. The residual and Jacobian must enforce the same inner
    boundary condition or the line search no longer minimizes the system that
    is actually being solved.
    """

    residual = jnp.zeros_like(U)
    residual = residual.at[0].set(
        (-3.0 * U[0] + 4.0 * U[1] - U[2]) / (2.0 * grid.dr)
    )

    safe_r = safe_radius(grid.r_full, grid.epsilon)
    # ensure that r is never 0 for numerical stability
    dU_dr = centered_first_derivative(U, grid.dr)
    # compute the first radial derivative for U
    d2U_dr2 = centered_second_derivative(U, grid.dr)
    # compute the second radial derivative for U
    interior_residual = d2U_dr2[1:-1] + 2.0 * dU_dr[1:-1] / safe_r[1:-1]
    interior_residual = interior_residual - source_term[1:-1] * U[1:-1] ** 5
    residual = residual.at[1:-1].set(interior_residual)
    # define the residual on the inside
    residual = jnp.where(exact_exterior_points, U - boundary_u, residual)
    # for vacuum points outside the matter, enforce the exact Schwarzschild solution for U as a boundary condition
    return residual


def build_dense_operator( radial_grid, dr, jacobian_diagonal, exact_exterior_points):
    """Build the finite-difference operator as a dense matrix for the elliptic U equation.

    The operator discretizes ``d^2U/dr^2 + (2/r) dU/dr + jacobian_diagonal * U``
    on the physical grid with:

    - Inner boundary (row 0): Neumann ``dU/dr = 0`` via second-order one-sided stencil
    - Interior rows: second-order centered FD stencil plus ``jacobian_diagonal``
    - Vacuum exterior rows: Dirichlet ``U = 1 + M / (2 r)``

    For the linear elliptic pre-solve pass ``jacobian_diagonal = zeros``.
    For the Newton Jacobian pass this function receives only the local
    ``-5 * source_term * U^4`` diagonal. Any off-diagonal matter-source
    derivative from shaped deposition is added explicitly by ``solve_metric_A``.
    """

    num_points = radial_grid.shape[0]
    matrix = jnp.zeros((num_points, num_points))

    # Inner boundary: Neumann dU/dr = 0 via second-order one-sided stencil
    # (-3 U[0] + 4 U[1] - U[2]) / (2 dr) = 0
    matrix = matrix.at[0, 0].set(-3.0 / (2.0 * dr))
    matrix = matrix.at[0, 1].set(4.0 / (2.0 * dr))
    matrix = matrix.at[0, 2].set(-1.0 / (2.0 * dr))

    # Interior rows: tridiagonal FD operator d^2/dr^2 + (2/r) d/dr + jacobian
    interior_indices = jnp.arange(1, num_points - 1)
    r_interior = radial_grid[1:-1]
    lower_diagonal = 1.0 / dr**2 - 1.0 / (r_interior * dr)
    main_diagonal = -2.0 / dr**2 + jacobian_diagonal[1:-1]
    upper_diagonal = 1.0 / dr**2 + 1.0 / (r_interior * dr)

    matrix = matrix.at[interior_indices, interior_indices - 1].set(lower_diagonal)
    matrix = matrix.at[interior_indices, interior_indices].set(main_diagonal)
    matrix = matrix.at[interior_indices, interior_indices + 1].set(upper_diagonal)

    # Overwrite the exact vacuum exterior by Dirichlet rows.
    matrix = jnp.where(exact_exterior_points[:, None], 0.0, matrix)
    matrix = matrix + jnp.diag(exact_exterior_points.astype(matrix.dtype))
    # For points in vacuuum, we know what the solution is

    return matrix


def metric_mass_energy_density_from_U(particles, U, grid, shape_mode="nearest"):
    """Compute the particle plus EM mass-energy density for the A equation."""
    A = U**2
    number_density = compute_number_density(particles, A, grid, shape_mode=shape_mode)
    particle_mass = particles.get_mass()
    particle_rho = particle_mass * number_density

    _, electric_field = compute_charge_density_and_radial_electric_field(
        [particles],
        A,
        grid,
        shape_mode=shape_mode,
    )
    rho_EM = compute_EM_energy_density(electric_field)
    rho = particle_rho + rho_EM
    return A, rho


def metric_source_term_from_U(particles, U, grid, shape_mode="nearest"):
    """Hamiltonian-constraint source term as a function of ``U = sqrt(A)``."""
    _, rho = metric_mass_energy_density_from_U(
        particles,
        U,
        grid,
        shape_mode=shape_mode,
    )
    return -2.0 * jnp.pi * rho


def metric_source_terms_from_U(particles, U, grid, shape_mode="nearest"):
    """Build the mass-energy source and its JAX-autodiff Jacobian."""

    A, rho = metric_mass_energy_density_from_U(
        particles,
        U,
        grid,
        shape_mode=shape_mode,
    )
    source_term = -2.0 * jnp.pi * rho
    source_term_U_jacobian = jax.jacfwd(
        lambda trial_U: metric_source_term_from_U(
            particles,
            trial_U,
            grid,
            shape_mode=shape_mode,
        )
    )(U)
    source_term_U_derivative = jnp.diag(source_term_U_jacobian)
    drho_dA = -source_term_U_derivative / (4.0 * jnp.pi * U)
    return (
        A,
        rho,
        drho_dA,
        source_term,
        source_term_U_derivative,
        source_term_U_jacobian,
    )


@partial(
    jax.jit,
    static_argnames=("max_newton_steps", "max_line_search_steps", "shape_mode"),
)
def solve_metric_A(
    particles,
    grid,
    schwarzschild_mass,
    initial_A_guess,
    tolerance=1.0e-4,
    max_newton_steps=10000,
    max_line_search_steps=100,
    armijo_c=1.0e-2,
    shape_mode="nearest",
):

    last_index = last_shape_support_index(particles.r, grid, shape_mode=shape_mode)
    # get the outermost grid index touched by the particle deposition support.

    grid_index = jnp.arange(grid.r_full.shape[0], dtype=last_index.dtype)
    # define the grid index array for masking the exact exterior region
    exact_exterior_points = grid_index > last_index
    # define the mask of points in the grid that are actually vaccuum
    exact_exterior_points = exact_exterior_points.at[-1].set(True)
    # set the last point to be vaccuuum.

    boundary_u = schwarzschild_u(grid.r_full, schwarzschild_mass, grid.epsilon)
    # compute the exact Schwarzschild U for the boundary condition in the vacuum region

    enforce_vaccuum_U = lambda U: jnp.where(exact_exterior_points, boundary_u, U)
    # helper function to enforce the exact Schwarzschild solution for U in the vacuum region

    initial_A_guess = jnp.where(exact_exterior_points, boundary_u**2, initial_A_guess)
    # apply the exact Schwarzschild solution for A in the vacuum region to the initial guess

    U_initial = jnp.sqrt(initial_A_guess)
    # convert the initial guess for A to an initial guess for U, which is what we
    # actually solve for.

    _, _, _, source_term_initial, _, _ = metric_source_terms_from_U(
        particles,
        U_initial,
        grid,
        shape_mode=shape_mode,
    )
    residual = nonlinear_residual_u(
        U_initial,
        source_term_initial,
        grid,
        boundary_u,
        exact_exterior_points,
    )
    residual_norm_sq = jnp.dot(residual, residual)
    residual_norm_inf = jnp.max(jnp.abs(residual))
    # define the initial residual and its norms for the convergence checks and line search

    initial_state = (
        U_initial,
        residual,
        residual_norm_sq,
        residual_norm_inf,
        jnp.int32(0),
        residual_norm_inf <= tolerance,
    )
    # build initial state for the JAX Newton iteration

    def newton_cond(state):
        _, _, _, _, step, converged = state
        return jnp.logical_and(~converged, step < max_newton_steps)

    def newton_body(state):
        U_current, _, _, _, step, _ = state

        _, _, _, source_term_current, _, source_term_U_jacobian = (
            metric_source_terms_from_U(
                particles,
                U_current,
                grid,
                shape_mode=shape_mode,
            )
        )
        # compute the current source term and its U derivative for the Jacobian

        current_residual = nonlinear_residual_u(
            U_current, source_term_current, grid, boundary_u,
            exact_exterior_points,
        )
        # compute the current residual

        residual_norm_sq_current = jnp.dot(current_residual, current_residual)
        # compute the current residual norm squared for the line search

        # The Jacobian keeps the finite-difference operator explicit. The
        # shaped source derivative is dense because rho_i depends on the
        # metric values used in each particle's interpolation stencil.
        jacobian_diagonal = -5.0 * source_term_current * U_current**4
        jacobian_matrix = build_dense_operator(
            grid.r_full, grid.dr, jacobian_diagonal,
            exact_exterior_points)
        jacobian_matrix = jacobian_matrix - (
            source_term_U_jacobian * U_current[:, jnp.newaxis] ** 5
        )
        # build the Jacobian matrix for the current state
        delta_U = jnp.linalg.solve(jacobian_matrix, -current_residual)
        # solve for the Newton update in the full system

        # Armijo backtracking on the merit phi = 0.5 ||R||^2.
        # For the Newton direction, d/d_alpha phi(U + alpha dU)|_0 = -||R||^2,
        # so the sufficient-decrease condition is:
        #   ||R_trial||^2 <= (1 - 2 c alpha) ||R||^2

        line_search_init = (
            jnp.asarray(1.0, dtype=U_current.dtype),
            U_current,
            current_residual,
            residual_norm_sq_current,
            jnp.bool_(False),
        )

        def line_search_body(_, ls_state):
            damping, U_best, res_best, norm_sq_best, accepted = ls_state
            trial_U = U_current + damping * delta_U
            # compute the trial solution for this line search step

            trial_U = enforce_vaccuum_U(trial_U)
            # enforce the exact Schwarzschild solution for U in the vacuum region on the trial solution

            def evaluate_trial_residual(candidate_U):
                _, _, _, trial_source_term, _, _ = metric_source_terms_from_U(
                    particles,
                    candidate_U,
                    grid,
                    shape_mode=shape_mode,
                )
                # compute the source term for the trial solution to evaluate the residual at that point

                trial_residual_local = nonlinear_residual_u(
                    candidate_U, trial_source_term, grid,
                    boundary_u, exact_exterior_points,
                )
                # compute the trial residual at the trial solution

                trial_norm_sq_local = jnp.dot(trial_residual_local, trial_residual_local)
                # compute the trial residual norm squared for the Armijo condition

                return trial_residual_local, trial_norm_sq_local

            def reject_trial_residual(_):
                invalid_residual = jnp.full_like(current_residual, jnp.inf)
                invalid_norm_sq = jnp.asarray(jnp.inf, dtype=current_residual.dtype)
                return invalid_residual, invalid_norm_sq

            trial_is_physical = jnp.logical_and(
                jnp.all(jnp.isfinite(trial_U)),
                jnp.all(trial_U > 0.0),
            )
            # check if the trial solution is physical (finite and positive, since U = sqrt(A))

            trial_residual_local, trial_norm_sq_local = jax.lax.cond(
                trial_is_physical,
                evaluate_trial_residual,
                reject_trial_residual,
                trial_U,
            )

            armijo_threshold = (
                1.0 - 2.0 * armijo_c * damping
            ) * residual_norm_sq_current
            accept_this = jnp.logical_and(
                ~accepted,
                jnp.logical_and(
                    jnp.isfinite(trial_norm_sq_local),
                    trial_norm_sq_local <= armijo_threshold,
                ),
            )

            U_best = jnp.where(accept_this, trial_U, U_best)
            res_best = jnp.where(accept_this, trial_residual_local, res_best)
            norm_sq_best = jnp.where(accept_this, trial_norm_sq_local, norm_sq_best)
            accepted = jnp.logical_or(accepted, accept_this)
            damping = jnp.where(accepted, damping, damping * 0.5)

            return damping, U_best, res_best, norm_sq_best, accepted

        _, U_new, res_new, norm_sq_new, accepted = jax.lax.fori_loop(
            0,
            max_line_search_steps,
            line_search_body,
            line_search_init,
        )
        # run the line search loop to find the step size that satisfies the Armijo condition

        U_out = jnp.where(accepted, U_new, U_current)
        # update the solution to the new trial solution if accepted, otherwise keep the current solution
        res_out = jnp.where(accepted, res_new, current_residual)
        # update the residual to the new trial residual if accepted, otherwise keep the current residual
        norm_sq_out = jnp.where(accepted, norm_sq_new, residual_norm_sq_current)
        # update the residual norm squared to the new trial value if accepted, otherwise keep the current value
        norm_inf_out = jnp.max(jnp.abs(res_out))
        # compute the infinity norm of the new residual for the convergence check
        converged_out = norm_inf_out <= tolerance
        # check if the new solution is converged based on the infinity norm of the residual

        return (
            U_out,
            res_out,
            norm_sq_out,
            norm_inf_out,
            step + jnp.int32(1),
            converged_out,
        )

    U_final, residual, residual_norm_sq, residual_norm_inf, step, converged = (
        jax.lax.while_loop(newton_cond, newton_body, initial_state)
    )

    U_final = jnp.where(exact_exterior_points, boundary_u, U_final)

    return U_final**2, jnp.bool_(converged), residual_norm_inf


@jax.jit
def euler_step_A(metric, grid, dt, schwarzschild_mass):
    """Predict the next ``A`` field with one explicit Euler step."""

    safe_r = safe_radius(grid.r_full, grid.epsilon)
    dA_dr = compute_metric_radial_derivative(metric.A, schwarzschild_mass, grid)

    # The polar-gauge evolution equation advects A with the shift:
    # d_t A = beta * (d_r A + A / r).
    dt_value = jnp.asarray(dt, dtype=metric.A.dtype)
    metric_rhs = metric.shift * (dA_dr + metric.A / safe_r)
    A_new = metric.A + dt_value * metric_rhs

    # Keep A positive before it is used in density reconstruction.
    A_new = safe_metric_A(A_new)

    # The outer boundary is vacuum and is pinned to the exact Schwarzschild A.
    outer_A = schwarzschild_A(grid.r_full[-1], schwarzschild_mass, grid.epsilon)
    A_new = A_new.at[-1].set(outer_A)

    # The center is regular, so keep its predictor fixed instead of introducing
    # a one-sided drift that would fight the Neumann condition.
    A_new = A_new.at[0].set(metric.A[0])
    return A_new
