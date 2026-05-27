from functools import partial

import jax
import jax.numpy as jnp

from RadiShPICR.deposition.particle_shapes import (
    interpolate_field_to_particles,
    radial_shape_stencil,
)
from RadiShPICR.relativity.utils import safe_metric_A, safe_radius


@partial(jax.jit, static_argnames=["shape_mode"])
def compute_number_density(particles, A, grid, shape_mode="nearest"):
    """
    Compute the number density of particles on a radial grid in relativistic electrostatic PIC.
    This function deposits particle number density onto a grid using shape function weighting,
    accounting for relativistic Lorentz factors and the spherical metric in radial coordinates.
    Parameters
    ----------
    particles : object
        Particle object containing:
        - r : array_like
            Radial positions of particles
        - u_r : array_like
            Radial velocity components of particles
        - u_phi : array_like
            Azimuthal velocity components of particles
    A : array_like
        Metric factor A(r) representing the lapse function in the radial direction
    grid : object
        Grid object containing:
        - r_full : array_like
            Full radial grid points
        - dr : float
            Radial grid spacing
        - epsilon : float
            Small parameter to avoid singularities at r=0
    shape_mode : str, optional
        Shape function mode for particle weighting. Options are:
        - "nearest" : nearest-grid-point (NGP) weighting (default)
        - other : higher-order shape functions with interpolation
    Returns
    -------
    number_density : array_like
        Number density deposited on the radial grid, with boundary values
        set to zero at r=0 and the outer boundary.
    """

    indices, weights = radial_shape_stencil(particles.r, grid, shape_mode=shape_mode)
    # get the stencil for the particle shape and the corresponding weights

    if shape_mode == "nearest":
        A_at_particle = safe_metric_A(A)[indices[0, :]]
    else:
        A_at_particle = interpolate_field_to_particles(
            safe_metric_A(A),
            particles.r,
            grid,
            shape_mode=shape_mode,
        )
    # interpolate the metric factor A to the particle positions using the appropriate stencil

    safe_r_particle = safe_radius(particles.r, grid.epsilon)
    # compute the safe radius for the particles to avoid singularities at r=0
    kinetic_term = particles.u_r**2 + particles.u_phi**2 / safe_r_particle**2
    # compute the kinetic term in the lorentz factor W.
    W = jnp.sqrt(1.0 + kinetic_term / A_at_particle**2)
    # compute the lorentz factor W for each particle using the kinetic term and the metric factor A

    number_density = jnp.zeros_like(grid.r_full)
    # initialize the number density array on the grid to zeros

    dV_spherical = 4.0 * jnp.pi * grid.r_full[indices] ** 2 * grid.dr
    # compute the spherical volume element at the grid points corresponding to the particle positions

    dn = (
        weights
        * W[jnp.newaxis, :]
        / (dV_spherical * A_at_particle[jnp.newaxis, :] ** 3)
    )
    # compute the contribution to the number density from each particle, accounting for the shape weights

    number_density = number_density.at[indices].add(dn)
    # deposit the contributions to the number density onto the grid using the shape stencil indices

    number_density = number_density.at[0].set(0.0)
    number_density = number_density.at[-1].set(0.0)
    # enforce boundary conditions in number density at the inner and outer boundaries of the grid

    return number_density


@partial(jax.jit, static_argnames=("shape_mode",))
def compute_number_density_metric_derivative(particles, A, grid, shape_mode="nearest"):
    """Diagonal derivative of number density with respect to metric ``A``."""

    jacobian = compute_number_density_metric_jacobian(
        particles,
        A,
        grid,
        shape_mode=shape_mode,
    )
    return jnp.diag(jacobian)


@partial(jax.jit, static_argnames=("shape_mode",))
def compute_number_density_metric_jacobian(particles, A, grid, shape_mode="nearest"):
    """
    Compute the Jacobian matrix of the number density with respect to the metric factor A in a spherical, relativistic particle-in-cell (PIC) simulation.
    This function calculates the derivative of the deposited number density on the grid with respect to the metric factor A, accounting for the relativistic Lorentz factor and the particle shape function. The Jacobian is assembled by depositing contributions from each particle using the specified shape function and enforcing boundary conditions.
    Parameters
    ----------
    particles : object
        An object containing particle data, with at least the following attributes:
            - r: array-like, particle radial positions
            - u_r: array-like, particle radial momenta
            - u_phi: array-like, particle angular momenta
    A : array-like
        The metric factor A evaluated on the grid.
    grid : object
        An object containing grid data, with at least the following attributes:
            - r_full: array-like, grid point positions
            - dr: float, grid spacing
            - epsilon: float, small value to regularize radius near zero
    shape_mode : str, optional
        The particle shape function to use for deposition and interpolation. Supported values include "nearest" and others as implemented in `radial_shape_stencil`.
    Returns
    -------
    dn_dA : ndarray
        The Jacobian matrix of the number density with respect to the metric factor A, with shape (N_grid, N_grid), where N_grid is the number of grid points. Boundary rows and columns are set to zero to enforce boundary conditions.
    Notes
    -----
    - The function avoids singularities at r=0 by using a "safe" radius.
    - The Jacobian is assembled using the particle shape stencil and weights.
    - Boundary conditions are enforced by zeroing out the first and last rows and columns of the Jacobian.
    """

    indices, weights = radial_shape_stencil(particles.r, grid, shape_mode=shape_mode)
    # get the stencil for the particle shape and the corresponding weights

    if shape_mode == "nearest":
        A_at_particle = safe_metric_A(A)[indices[0, :]]
    else:
        A_at_particle = interpolate_field_to_particles(
            safe_metric_A(A),
            particles.r,
            grid,
            shape_mode=shape_mode,
        )
    # interpolate the metric factor A to the particle positions using the appropriate stencil

    safe_r_particle = safe_radius(particles.r, grid.epsilon)
    # compute the safe radius for the particles to avoid singularities at r=0
    kinetic_term = particles.u_r**2 + particles.u_phi**2 / safe_r_particle**2
    # compute the kinetic term in the lorentz factor W.
    W = jnp.sqrt(1.0 + kinetic_term / A_at_particle**2)
    # compute the lorentz factor W for each particle using the kinetic term and the metric factor A

    dV_spherical = 4.0 * jnp.pi * grid.r_full[indices] ** 2 * grid.dr
    # compute the spherical volume element at the grid points corresponding to the particle positions

    dn_dA = jnp.zeros((grid.r_full.shape[0], grid.r_full.shape[0]), dtype=A.dtype)
    # initialize the Jacobian matrix for the number density with respect to the metric factor A to zeros

    dn_dA_per_particle = weights * (
        -3.0
        * W[jnp.newaxis, :]
        / (dV_spherical * A_at_particle[jnp.newaxis, :] ** 4)
        - kinetic_term[jnp.newaxis, :]
        / (
            dV_spherical
            * A_at_particle[jnp.newaxis, :] ** 6
            * W[jnp.newaxis, :]
        )
    )
    # compute the contribution to the Jacobian from each particle, accounting for the shape weights and the dependence on A

    row_indices = indices[:, jnp.newaxis, :]
    column_indices = indices[jnp.newaxis, :, :]
    # compute the row and column indices for the Jacobian matrix based on the shape stencil indices

    jacobian_values = (
        dn_dA_per_particle[:, jnp.newaxis, :] * weights[jnp.newaxis, :, :]
    )
    # compute the values to be added to the Jacobian matrix for each particle contribution

    row_indices = jnp.broadcast_to(row_indices, jacobian_values.shape)
    column_indices = jnp.broadcast_to(column_indices, jacobian_values.shape)
    # broadcast the row and column indices to match the shape of the Jacobian values for proper indexing

    dn_dA = dn_dA.at[(row_indices.ravel(), column_indices.ravel())].add(
        jacobian_values.ravel()
    )
    # deposit the contributions to the Jacobian matrix using the shape stencil indices

    dn_dA = dn_dA.at[0, :].set(0.0)
    dn_dA = dn_dA.at[-1, :].set(0.0)
    dn_dA = dn_dA.at[:, 0].set(0.0)
    dn_dA = dn_dA.at[:, -1].set(0.0)
    # enforce boundary conditions in the Jacobian matrix at the inner and outer boundaries of the grid

    return dn_dA
