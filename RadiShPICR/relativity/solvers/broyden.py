from functools import partial

import jax
import jax.numpy as jnp

from RadiShPICR.deposition import last_shape_support_index
from RadiShPICR.relativity.A import (
    metric_source_term_from_U,
    nonlinear_residual_u,
)
from RadiShPICR.relativity.schwarzschild import schwarzschild_u


def _particle_list(particles):
    if isinstance(particles, (list, tuple)):
        return particles
    return [particles]


def finite_difference_residual_jacobian(
    residual_func,
    U,
    step_size=1.0e-4,
):
    """Central finite-difference Jacobian for a residual written in ``U``."""

    step_value = jnp.asarray(step_size, dtype=U.dtype)
    basis = jnp.eye(U.shape[0], dtype=U.dtype)

    def residual_column(direction):
        residual_plus = residual_func(U + step_value * direction)
        residual_minus = residual_func(U - step_value * direction)
        return (residual_plus - residual_minus) / (2.0 * step_value)

    return jax.vmap(residual_column)(basis).T


def metric_A_residual_from_U(
    particles,
    U,
    grid,
    boundary_u,
    exact_exterior_points,
    shape_mode="nearest",
    EM=True,
):
    """Evaluate the same Hamiltonian-constraint residual used by ``solve_metric_A``."""

    source_term = metric_source_term_from_U(
        particles,
        U,
        grid,
        shape_mode=shape_mode,
        EM=EM,
    )
    return nonlinear_residual_u(
        U,
        source_term,
        grid,
        boundary_u,
        exact_exterior_points,
    )


@partial(
    jax.jit,
    static_argnames=(
        "max_broyden_steps",
        "max_line_search_steps",
        "shape_mode",
        "EM",
    ),
)
def solve_metric_A_broyden(
    particles,
    grid,
    schwarzschild_mass,
    initial_A_guess,
    tolerance=1.0e-9,
    max_broyden_steps=400,
    max_line_search_steps=60,
    finite_difference_step=1.0e-4,
    shape_mode="nearest",
    EM=True,
):
    """Solve the polar-gauge ``A`` equation with a finite-difference Broyden method."""

    species_list = _particle_list(particles)
    last_index = last_shape_support_index(
        species_list[0].r,
        grid,
        shape_mode=shape_mode,
    )
    for species in species_list[1:]:
        species_last_index = last_shape_support_index(
            species.r,
            grid,
            shape_mode=shape_mode,
        )
        last_index = jnp.maximum(last_index, species_last_index)
    # get the outermost grid index touched by the particle deposition support.

    grid_index = jnp.arange(grid.r_full.shape[0], dtype=last_index.dtype)
    exact_exterior_points = grid_index > last_index
    exact_exterior_points = exact_exterior_points.at[-1].set(True)
    # pin the vacuum exterior to the exact Schwarzschild solution, matching the
    # Newton solver's discrete exterior convention.

    boundary_u = schwarzschild_u(grid.r_full, schwarzschild_mass, grid.epsilon)
    enforce_vaccuum_U = lambda U: jnp.where(exact_exterior_points, boundary_u, U)

    initial_A_guess = jnp.where(exact_exterior_points, boundary_u**2, initial_A_guess)
    U_initial = jnp.sqrt(initial_A_guess)
    U_initial = enforce_vaccuum_U(U_initial)

    residual_func = lambda trial_U: metric_A_residual_from_U(
        particles,
        trial_U,
        grid,
        boundary_u,
        exact_exterior_points,
        shape_mode=shape_mode,
        EM=EM,
    )

    residual = residual_func(U_initial)
    residual_norm_sq = jnp.dot(residual, residual)
    residual_norm_inf = jnp.max(jnp.abs(residual))
    jacobian_matrix = finite_difference_residual_jacobian(
        residual_func,
        U_initial,
        step_size=finite_difference_step,
    )
    # Broyden starts from a finite-difference Jacobian of the full nonlinear
    # residual, then updates that matrix with accepted nonlinear steps.

    initial_state = (
        U_initial,
        residual,
        residual_norm_sq,
        residual_norm_inf,
        jacobian_matrix,
        jnp.int32(0),
        residual_norm_inf <= tolerance,
    )

    def broyden_cond(state):
        _, _, _, _, _, step, converged = state
        return jnp.logical_and(~converged, step < max_broyden_steps)

    def broyden_body(state):
        U_current, current_residual, residual_norm_sq_current, _, jacobian, step, _ = (
            state
        )

        delta_U = jnp.linalg.solve(jacobian, -current_residual)
        # solve the current secant linearization for the candidate Broyden step.

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
            trial_U = enforce_vaccuum_U(trial_U)

            def evaluate_trial_residual(candidate_U):
                trial_residual = residual_func(candidate_U)
                trial_norm_sq = jnp.dot(trial_residual, trial_residual)
                return trial_residual, trial_norm_sq

            def reject_trial_residual(_):
                invalid_residual = jnp.full_like(current_residual, jnp.inf)
                invalid_norm_sq = jnp.asarray(jnp.inf, dtype=current_residual.dtype)
                return invalid_residual, invalid_norm_sq

            trial_is_physical = jnp.logical_and(
                jnp.all(jnp.isfinite(trial_U)),
                jnp.all(trial_U > 0.0),
            )

            trial_residual, trial_norm_sq = jax.lax.cond(
                trial_is_physical,
                evaluate_trial_residual,
                reject_trial_residual,
                trial_U,
            )

            accept_this = jnp.logical_and(
                ~accepted,
                jnp.logical_and(
                    jnp.isfinite(trial_norm_sq),
                    trial_norm_sq < residual_norm_sq_current,
                ),
            )

            U_best = jnp.where(accept_this, trial_U, U_best)
            res_best = jnp.where(accept_this, trial_residual, res_best)
            norm_sq_best = jnp.where(accept_this, trial_norm_sq, norm_sq_best)
            accepted = jnp.logical_or(accepted, accept_this)
            damping = jnp.where(accepted, damping, damping * 0.5)

            return damping, U_best, res_best, norm_sq_best, accepted

        _, U_new, res_new, norm_sq_new, accepted = jax.lax.fori_loop(
            0,
            max_line_search_steps,
            line_search_body,
            line_search_init,
        )

        U_out = jnp.where(accepted, U_new, U_current)
        res_out = jnp.where(accepted, res_new, current_residual)
        norm_sq_out = jnp.where(accepted, norm_sq_new, residual_norm_sq_current)
        norm_inf_out = jnp.max(jnp.abs(res_out))

        step_U = U_out - U_current
        residual_change = res_out - current_residual
        step_norm_sq = jnp.dot(step_U, step_U)
        safe_step_norm_sq = jnp.where(step_norm_sq > 0.0, step_norm_sq, 1.0)
        secant_error = residual_change - jacobian @ step_U
        jacobian_update = jnp.outer(secant_error, step_U) / safe_step_norm_sq
        update_jacobian = jnp.logical_and(accepted, step_norm_sq > 0.0)
        jacobian_out = jnp.where(update_jacobian, jacobian + jacobian_update, jacobian)
        # Broyden's rank-one update is only meaningful for an accepted nonzero
        # nonlinear step, since rejected trials do not define a new secant pair.

        converged_out = norm_inf_out <= tolerance

        return (
            U_out,
            res_out,
            norm_sq_out,
            norm_inf_out,
            jacobian_out,
            step + jnp.int32(1),
            converged_out,
        )

    U_final, residual, residual_norm_sq, residual_norm_inf, _, step, converged = (
        jax.lax.while_loop(broyden_cond, broyden_body, initial_state)
    )

    U_final = enforce_vaccuum_U(U_final)

    return U_final**2, jnp.bool_(converged), residual_norm_inf
