import jax.numpy as jnp


def total_particle_mass(particles):
    """Total macro-particle mass from the particle getter contract."""

    return jnp.sum(jnp.ones_like(particles.r) * particles.get_mass())


def total_particle_charge(particles):
    """Total macro-particle charge from the particle getter contract."""

    return jnp.sum(jnp.ones_like(particles.r) * particles.get_charge())


def reissner_nordstrom_A(r, mass, charge):
    rQ = charge**2 / (4.0 * jnp.pi)

    return (1.0 + mass / (2.0 * r))**2 - rQ**2 / (4.0 * r**2)


def reissner_nordstrom_lapse(r, mass, charge):
    rQ = charge**2 / (4.0 * jnp.pi)

    numerator = (1.0 - mass / (2.0 * r)) * (1.0 + mass / (2.0 * r))
    numerator = numerator + rQ**2 / (4.0 * r**2)

    return numerator / reissner_nordstrom_A(r, mass, charge)


def vacuum_rescale_factors(A_outer, alpha_outer, r_outer, mass, charge):
    """Coordinate rescaling that matches the vacuum solution at the outer cell."""

    rQ = charge**2 / (4.0 * jnp.pi)

    linear_coefficient = mass / r_outer - A_outer
    constant_coefficient = (mass**2 - rQ**2) / (4.0 * r_outer**2)
    discriminant = linear_coefficient**2 - 4.0 * constant_coefficient
    X_r = 0.5 * (-linear_coefficient + jnp.sqrt(discriminant))

    rescaled_outer_radius = X_r * r_outer
    X_t = alpha_outer / reissner_nordstrom_lapse(
        rescaled_outer_radius,
        mass,
        charge,
    )

    return X_r, X_t


def rescale_metric_to_vacuum_boundary(U_state, particles):
    A, phi, alpha, Krr, beta_over_r, Er, source_terms, r_grid = U_state
    mass_density, charge_density, Srr, Sr = source_terms

    mass = total_particle_mass(particles)
    charge = total_particle_charge(particles)
    X_r, X_t = vacuum_rescale_factors(
        A[-1],
        alpha[-1],
        r_grid[-1],
        mass,
        charge,
    )

    A = A / X_r
    phi = phi / X_r ** (3.0 / 2.0)
    alpha = alpha / X_t
    Krr = Krr * X_r**2 / X_t
    beta_over_r = beta_over_r / X_t
    Er = X_r * Er
    r_grid = X_r * r_grid

    Srr = Srr * (X_r / X_t) ** 2
    Sr = Sr * X_r / X_t
    source_terms = (mass_density, charge_density, Srr, Sr)

    return (
        A,
        phi,
        alpha,
        Krr,
        beta_over_r,
        Er,
        source_terms,
        r_grid,
    )
