import jax
import jax.numpy as jnp

from RadiShPICR.EM.EM_energy_momentum_tensor import compute_EM_energy_density
from RadiShPICR.EM.gauss_law import compute_charge_density_and_radial_electric_field
from RadiShPICR.deposition import (
    last_shape_support_index,
)
from RadiShPICR.deposition.number_density import (
    compute_number_density,
    compute_number_density_metric_derivative,
    compute_number_density_metric_jacobian,
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


def metric_source_terms_from_U(particles, U, grid, shape_mode="nearest"):
    """Build the matter source terms needed by the coupled Newton solve."""

    A = U**2
    number_density = compute_number_density(particles, A, grid, shape_mode=shape_mode)
    dn_dA = compute_number_density_metric_derivative(
        particles,
        A,
        grid,
        shape_mode=shape_mode,
    )
    dn_dA_jacobian = compute_number_density_metric_jacobian(
        particles,
        A,
        grid,
        shape_mode=shape_mode,
    )
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
    # The EM energy density changes the Hamiltonian-constraint source. Its
    # derivative through the Gauss-law solve is intentionally not included in
    # the Newton matrix in this pass.
    drho_dA = particle_mass * dn_dA
    drho_dA_jacobian = particle_mass * dn_dA_jacobian
    source_term = -2.0 * jnp.pi * rho
    source_term_U_derivative = -4.0 * jnp.pi * U * drho_dA
    source_term_U_jacobian = -4.0 * jnp.pi * drho_dA_jacobian * U[jnp.newaxis, :]
    return (
        A,
        rho,
        drho_dA,
        source_term,
        source_term_U_derivative,
        source_term_U_jacobian,
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

    U_current = U_initial
    step = 0
    converged = bool(float(residual_norm_inf) <= tolerance)
    # build initial state for the Newton iteration. The loop stays at Python
    # level because the radial Gauss-law solve inside metric_source_terms_from_U
    # is a SciPy sparse solve, not a JAX operation.

    while (not converged) and step < max_newton_steps:
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


        damping = jnp.asarray(1.0, dtype=U_current.dtype)
        U_best = U_current
        res_best = current_residual
        norm_sq_best = residual_norm_sq_current
        accepted = False
        current_norm_sq_value = float(residual_norm_sq_current)
        # start the line search with a full Newton step and no accepted solution

        for _ in range(max_line_search_steps):
            trial_U = U_current + damping * delta_U
            # compute the trial solution for this line search step

            trial_U = enforce_vaccuum_U(trial_U)
            # enforce the exact Schwarzschild solution for U in the vacuum region on the trial solution

            trial_is_physical = bool(
                jnp.logical_and(
                    jnp.all(jnp.isfinite(trial_U)),
                    jnp.all(trial_U > 0.0),
                )
            )
            # check if the trial solution is physical (finite and positive, since U = sqrt(A))

            if trial_is_physical:
                _, _, _, trial_source_term, _, _ = metric_source_terms_from_U(
                    particles,
                    trial_U,
                    grid,
                    shape_mode=shape_mode,
                )
                # compute the source term for the trial solution to evaluate the residual at that point

                trial_residual_local = nonlinear_residual_u(
                    trial_U, trial_source_term, grid,
                    boundary_u, exact_exterior_points,
                )
                # compute the trial residual at the trial solution

                trial_norm_sq_local = jnp.dot(trial_residual_local, trial_residual_local)
                # compute the trial residual norm squared for the Armijo condition
            else:
                trial_residual_local = jnp.full_like(current_residual, jnp.inf)
                trial_norm_sq_local = jnp.asarray(jnp.inf, dtype=current_residual.dtype)
                # if the trial solution is not physical, return invalid values that will fail the Armijo condition

            armijo_threshold = (
                1.0 - 2.0 * armijo_c * float(damping)
            ) * current_norm_sq_value
            trial_norm_sq_value = float(trial_norm_sq_local)
            accept_this = bool(jnp.isfinite(trial_norm_sq_local)) and (
                trial_norm_sq_value <= armijo_threshold
            )

            # Update best values only on first acceptance
            if bool(accept_this):
                U_best = trial_U
                res_best = trial_residual_local
                norm_sq_best = trial_norm_sq_local
                accepted = True
                break

            # Halve the damping for the next attempt (only matters if not accepted)
            damping = damping * 0.5

        # run the line search loop to find the step size that satisfies the Armijo condition

        # If line search failed, keep current state (converged stays False)
        U_current = U_best if accepted else U_current
        # update the solution to the new trial solution if accepted, otherwise keep the current solution
        residual = res_best if accepted else current_residual
        # update the residual to the new trial residual if accepted, otherwise keep the current residual
        residual_norm_sq = norm_sq_best if accepted else residual_norm_sq_current
        # update the residual norm squared to the new trial value if accepted, otherwise keep the current value
        residual_norm_inf = jnp.max(jnp.abs(residual))
        # compute the infinity norm of the new residual for the convergence check
        converged = bool(float(residual_norm_inf) <= tolerance)
        # check if the new solution is converged based on the infinity norm of the residual

        step = step + 1

    U_final = U_current
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
