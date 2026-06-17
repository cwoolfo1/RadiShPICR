import jax.numpy as jnp

from RadiShPICR.deposition.charge_density import charge_density_at_point
from RadiShPICR.deposition.mass_density import mass_density_at_point
from RadiShPICR.forces.energy_momentum_tensor import Srr_at_point, Sr_at_point
from RadiShPICR.forces.utils import pad_value
from RadiShPICR.forces.vacuum_conditions import rescale_metric_to_vacuum_boundary


def _safe_radius(r, dr):
    return jnp.maximum(r, 0.5 * dr)


def dr_A(U_state):
    A, phi, alpha, Krr, beta_over_r, Er, source_terms, r = U_state
    return 2.0 * phi * jnp.sqrt(A)


def dr_sqrt_phi(U_state, dr=None):
    A, phi, alpha, Krr, beta_over_r, Er, source_terms, r = U_state
    rho, charge_density, Srr, Sr = source_terms
    mass_energy_density = rho + 0.5 * Er**2

    if dr is None:
        safe_r = jnp.where(r == 0.0, 1.0, r)
    else:
        safe_r = _safe_radius(r, dr)

    interior_term = -2.0 * jnp.pi * jnp.sqrt(A) ** 5 * mass_energy_density - 2.0 * phi / safe_r
    center_term = -jnp.pi * jnp.sqrt(A) ** 5 * mass_energy_density / 3.0

    return jnp.where(r == 0.0, center_term, interior_term)


def dr_alpha(U_state, dr=None):
    A, phi, alpha, Krr, beta_over_r, Er, source_terms, r = U_state
    rho, charge_density, Srr, Sr = source_terms
    A_for_denominators = pad_value(A)

    first_term = 4.0 * jnp.pi * alpha * Srr * r * A
    second_term = -2.0 * alpha * phi * jnp.sqrt(A)
    third_term = -2.0 * alpha * phi**2 * r
    denominator = A_for_denominators * (
        1.0 + 2.0 * r * phi / jnp.sqrt(A_for_denominators)
    )

    return (first_term + second_term + third_term) / denominator


def Krr_from_state(U_state):
    A, phi, alpha, Krr, beta_over_r, Er, source_terms, r = U_state
    rho, charge_density, Srr, Sr = source_terms
    A_for_denominators = pad_value(A)

    return 4.0 * jnp.pi * r * Sr / (
        1.0 + 2.0 * r * phi / jnp.sqrt(A_for_denominators)
    )


def dr_beta_over_r(U_state, dr=None):
    A, phi, alpha, Krr, beta_over_r, Er, source_terms, r = U_state
    safe_r = jnp.where(r == 0.0, 1.0, r) if dr is None else _safe_radius(r, dr)
    return jnp.where(r == 0.0, 0.0, alpha * Krr_from_state(U_state) / safe_r)


def dr_Er(U_state, dr=None):
    A, phi, alpha, Krr, beta_over_r, Er, source_terms, r = U_state
    rho, charge_density, Srr, Sr = source_terms
    A_for_denominators = pad_value(A)

    safe_r = jnp.where(r == 0.0, 1.0, r) if dr is None else _safe_radius(r, dr)
    interior_term = (
        charge_density
        - 2.0 * Er / safe_r
        - 2.0 * phi * Er / jnp.sqrt(A_for_denominators)
    )
    center_term = charge_density / 3.0

    return jnp.where(r == 0.0, center_term, interior_term)


def _source_terms_at_point(particles, A_at_point, radial_coordinate, dr):
    mass_density = mass_density_at_point(particles, A_at_point, radial_coordinate, dr)
    charge_density = charge_density_at_point(particles, A_at_point, radial_coordinate, dr)
    Srr = Srr_at_point(particles, A_at_point, radial_coordinate, dr)
    Sr = Sr_at_point(particles, A_at_point, radial_coordinate, dr)

    return mass_density, charge_density, Srr, Sr


def heuns_method(U_state, dr, particles):
    A, phi, alpha, Krr, beta_over_r, Er, source_terms, r = U_state

    dA_dr = dr_A(U_state)
    dphi_dr = dr_sqrt_phi(U_state, dr)
    dalpha_dr = dr_alpha(U_state, dr)
    dE_dr = dr_Er(U_state, dr)

    r_predictor = r + dr
    A_predictor = A + dA_dr * dr
    phi_predictor = phi + dphi_dr * dr
    alpha_predictor = alpha + dalpha_dr * dr
    Er_predictor = Er + dE_dr * dr
    source_terms_predictor = _source_terms_at_point(
        particles, A_predictor, r_predictor, dr
    )
    Krr_predictor = Krr_from_state(
        (A_predictor, phi_predictor, alpha_predictor, Krr, beta_over_r, Er_predictor, source_terms_predictor, r_predictor)
    )
    beta_over_r_predictor = dr_beta_over_r(
        (A_predictor, phi_predictor, alpha_predictor, Krr_predictor, beta_over_r, Er_predictor, source_terms_predictor, r_predictor),
        dr,
    )
    predictor_state = (
        A_predictor,
        phi_predictor,
        alpha_predictor,
        Krr_predictor,
        beta_over_r_predictor,
        Er_predictor,
        source_terms_predictor,
        r_predictor,
    )

    dA_dr_predictor = dr_A(predictor_state)
    dphi_dr_predictor = dr_sqrt_phi(predictor_state, dr)
    dalpha_dr_predictor = dr_alpha(predictor_state, dr)
    dE_dr_predictor = dr_Er(predictor_state, dr)

    A_corrected = A + 0.5 * (dA_dr + dA_dr_predictor) * dr
    phi_corrected = phi + 0.5 * (dphi_dr + dphi_dr_predictor) * dr
    alpha_corrected = alpha + 0.5 * (dalpha_dr + dalpha_dr_predictor) * dr
    Er_corrected = Er + 0.5 * (dE_dr + dE_dr_predictor) * dr
    source_terms_corrected = _source_terms_at_point(
        particles, A_corrected, r_predictor, dr
    )
    Krr_corrected = Krr_from_state(
        (
            A_corrected,
            phi_corrected,
            alpha_corrected,
            Krr,
            beta_over_r,
            Er_corrected,
            source_terms_corrected,
            r_predictor,
        )
    )
    beta_over_r_corrected = dr_beta_over_r(
        (
            A_corrected,
            phi_corrected,
            alpha_corrected,
            Krr_corrected,
            beta_over_r,
            Er_corrected,
            source_terms_corrected,
            r_predictor,
        ),
        dr,
    )

    return (
        A_corrected,
        phi_corrected,
        alpha_corrected,
        Krr_corrected,
        beta_over_r_corrected,
        Er_corrected,
        source_terms_corrected,
        r_predictor,
    )


def calculate_metric(particles, r_grid, dr):
    r_grid = jnp.asarray(r_grid)
    dr = jnp.asarray(dr, dtype=r_grid.dtype)

    initial_A = jnp.asarray(1.0, dtype=r_grid.dtype)
    initial_phi = jnp.asarray(0.0, dtype=r_grid.dtype)
    initial_alpha = jnp.asarray(1.0, dtype=r_grid.dtype)
    initial_Krr = jnp.asarray(0.0, dtype=r_grid.dtype)
    initial_beta_over_r = jnp.asarray(0.0, dtype=r_grid.dtype)
    initial_Er = jnp.asarray(0.0, dtype=r_grid.dtype)
    initial_r = r_grid[0]
    initial_source_terms = _source_terms_at_point(particles, initial_A, initial_r, dr)

    state = (
        initial_A,
        initial_phi,
        initial_alpha,
        initial_Krr,
        initial_beta_over_r,
        initial_Er,
        initial_source_terms,
        initial_r,
    )

    A_values = []
    phi_values = []
    alpha_values = []
    Krr_values = []
    beta_over_r_values = []
    Er_values = []
    mass_density_values = []
    charge_density_values = []
    Srr_values = []
    Sr_values = []

    for grid_index in range(r_grid.shape[0]):
        if grid_index > 0:
            local_dr = r_grid[grid_index] - r_grid[grid_index - 1]
            state = heuns_method(state, local_dr, particles)

        A, phi, alpha, Krr, beta_over_r, Er, source_terms, r = state
        mass_density, charge_density, Srr, Sr = source_terms

        A_values.append(A)
        phi_values.append(phi)
        alpha_values.append(alpha)
        Krr_values.append(Krr)
        beta_over_r_values.append(beta_over_r)
        Er_values.append(Er)
        mass_density_values.append(mass_density)
        charge_density_values.append(charge_density)
        Srr_values.append(Srr)
        Sr_values.append(Sr)

    source_terms = (
        jnp.asarray(mass_density_values),
        jnp.asarray(charge_density_values),
        jnp.asarray(Srr_values),
        jnp.asarray(Sr_values),
    )

    U_state = (
        jnp.asarray(A_values),
        jnp.asarray(phi_values),
        jnp.asarray(alpha_values),
        jnp.asarray(Krr_values),
        jnp.asarray(beta_over_r_values),
        jnp.asarray(Er_values),
        source_terms,
        r_grid,
    )

    return rescale_metric_to_vacuum_boundary(U_state, particles)
