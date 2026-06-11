import jax.numpy as jnp

from RadiShPICR.deposition.charge_density import charge_density_at_point
from RadiShPICR.deposition.mass_density import mass_density_at_point
from RadiShPICR.relativity.energy_momentum_tensor import Srr_at_point, Sr_at_point

def dr_A(U_state):
    A, phi, alpha, Krr_, beta, Er, source_terms, r = U_state
    rho, charge_density, Srr, Sr = source_terms

    return 2 * phi * jnp.sqrt(A)

def dr_sqrt_phi(U_state):
    A, phi, alpha, Krr_, beta, Er, source_terms, r = U_state
    rho, charge_density, Srr, Sr = source_terms


    mass_energy_density = rho + Er**2 / 2.0

    if r == 0.0:
        first_term = -1.0 / 3.0 * jnp.pi * jnp.sqrt(A)**5 * mass_energy_density
        second_term = 0.0
    else:
        first_term = -2.0 * jnp.pi * jnp.sqrt(A)**5 * mass_energy_density
        second_term = -2.0 * phi / r

    return first_term + second_term

def dr_alpha(U_state):
    A, phi, alpha, Krr_, beta, Er, source_terms, r = U_state
    rho, charge_density, Srr, Sr = source_terms

    first_term  = 4.0 * jnp.pi * alpha * Srr * r * A
    second_term = -2.0 * alpha * phi * jnp.sqrt(A)
    third_term  = -2.0 * alpha * phi**2 * r

    denominator = A * ( 1 + 2 * r * phi / jnp.sqrt(A) )

    return ( first_term + second_term + third_term ) / denominator


def Krr(U_state):
    A, phi, alpha, Krr_, beta, Er, source_terms, r = U_state
    rho, charge_density, Srr, Sr = source_terms

    return 4 * jnp.pi * r * Sr / (  1 + 2*r*phi / jnp.sqrt(A) )

def dr_beta_over_r(U_state):
    A, phi, alpha, Krr_, beta, Er, source_terms, r = U_state
    rho, charge_density, Srr, Sr = source_terms

    return alpha * Krr(U_state) / r

def dr_Er(U_state):
    A, phi, alpha, Krr_, beta, Er, source_terms, r = U_state
    rho, charge_density, Srr, Sr = source_terms

    if r == 0.0:
        return charge_density / 3.0
    else:
        return charge_density - 2.0 * Er / r - 2.0 * phi * Er / jnp.sqrt(A)
    

# Heuns method for solving the coupled ODEs for A and phi
def heuns_method(U_state, dr, particles):
    A, phi, alpha, Krr_, beta, Er, source_terms, r = U_state

    dA_dr = dr_A(U_state)
    dphi_dr = dr_sqrt_phi(U_state)
    dalpha_dr = dr_alpha(U_state)
    dE_dr = dr_Er(U_state)

    # Predictor step
    A_predictor = A + dA_dr * dr
    phi_predictor = phi + dphi_dr * dr
    alpha_predictor = alpha + dalpha_dr * dr
    Er_predictor = Er + dE_dr * dr

    U_predictor = (A_predictor, phi_predictor, alpha_predictor, Krr_, beta, Er_predictor, source_terms, r + dr)
    
    dA_dr_predictor = dr_A(U_predictor)
    dphi_dr_predictor = dr_sqrt_phi(U_predictor)
    dalpha_dr_predictor = dr_alpha(U_predictor)
    dE_dr_predictor = dr_Er(U_predictor)
    # Corrector step
    A_corrected = A + 0.5 * (dA_dr + dA_dr_predictor) * dr
    phi_corrected = phi + 0.5 * (dphi_dr + dphi_dr_predictor) * dr
    alpha_corrected = alpha + 0.5 * (dalpha_dr + dalpha_dr_predictor) * dr
    Er_corrected = Er + 0.5 * (dE_dr + dE_dr_predictor) * dr

    beta_over_r_corrected = dr_beta_over_r((A_corrected, phi_corrected, alpha_corrected, Krr_, beta, Er_corrected, source_terms, r + dr))
    Krr_corrected = Krr((A_corrected, phi_corrected, alpha_corrected, Krr_, beta, Er_corrected, source_terms, r + dr))

    # source terms
    charge_density_corrected = charge_density_at_point(particles, A_corrected, r + dr, dr)
    mass_density_corrected = mass_density_at_point(particles, A_corrected, r + dr, dr)
    Srr_corrected = Srr_at_point(particles, A_corrected, r + dr, dr)
    Sr_corrected = Sr_at_point(particles, A_corrected, r + dr, dr)
    source_terms_corrected = (mass_density_corrected, charge_density_corrected, Srr_corrected, Sr_corrected)


    return A_corrected, phi_corrected, alpha_corrected, beta_over_r_corrected, Krr_corrected, Er_corrected, source_terms_corrected, r + dr




def calculate_metric(particles, r_grid, dr):

    initial_A = 1.0
    initial_phi = 0.0
    initial_alpha = 1.0
    initial_Krr = 0.0
    initial_beta_over_r = 0.0
    initial_Er = 0.0
    # initial values for the metric fields at r = 0.0


    charge_density = charge_density_at_point(particles, initial_A, 0, dr)
    mass_density = mass_density_at_point(particles, initial_A, 0, dr)
    Srr = Srr_at_point(particles, initial_A, 0, dr)
    Sr = Sr_at_point(particles, initial_A, 0, dr)
    # compuute the source terms at r = 0.0

    source_terms = (mass_density, charge_density, Srr, Sr)
    # pack the initial state into a tuple to pass to Heun's method
    state = (initial_A,                       \
            initial_phi, initial_alpha,       \
            initial_beta_over_r, initial_Krr, \
            initial_Er, source_terms, 0.0     )
    # A = 1.0, phi = 0.0, alpha = 1.0, beta_over_r = 0.0, Krr = 0.0, Er = 0.0, source_terms, r = 0.0 at the start

    A_values = []
    phi_values = []
    alpha_values = []
    beta_over_r_values = []
    Krr_values = []
    Er_values = []
    mass_density_values = []
    charge_density_values = []
    Srr_values = []
    Sr_values = []

    for r in r_grid:
        state = heuns_method(state, dr, particles)
        # integrate the values forward in r using Heun's method

        A, phi, alpha, beta_over_r, Krr, Er, source_terms, r = state
        mass_density, charge_density, Srr, Sr = source_terms

        A_values.append(A)
        phi_values.append(phi)
        alpha_values.append(alpha)
        beta_over_r_values.append(beta_over_r)
        Krr_values.append(Krr)
        Er_values.append(Er)
        mass_density_values.append(mass_density)
        charge_density_values.append(charge_density)
        Srr_values.append(Srr)
        Sr_values.append(Sr)
        # store the values for diagnostics and plotting

    A_values = jnp.array(A_values)
    phi_values = jnp.array(phi_values)
    alpha_values = jnp.array(alpha_values)
    beta_over_r_values = jnp.array(beta_over_r_values)
    Krr_values = jnp.array(Krr_values)
    Er_values = jnp.array(Er_values)
    mass_density_values = jnp.array(mass_density_values)
    charge_density_values = jnp.array(charge_density_values)
    Srr_values = jnp.array(Srr_values)
    Sr_values = jnp.array(Sr_values)
    # convert lists to arrays for easier plotting and analysis

    source_terms = (mass_density_values, charge_density_values, Srr_values, Sr_values)

    return A_values, phi_values, alpha_values, beta_over_r_values, Krr_values, Er_values, source_terms