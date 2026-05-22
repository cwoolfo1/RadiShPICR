from __future__ import annotations
from functools import partial
from typing import NamedTuple
import jax
import jax.numpy as jnp


from src.utils import ( centered_first_derivative, safe_radius,
                       compute_metric_radial_derivative )
from src.matter_source_terms import interpolate_to_particle


class ParticleDerivativeState(NamedTuple):
    """Time derivatives for the evolved orbit variables."""

    dr_dt: jnp.ndarray
    dphi_dt: jnp.ndarray
    du_r_dt: jnp.ndarray


class ParticleState(NamedTuple):

    mass: jnp.ndarray
    r: jnp.ndarray
    theta: jnp.ndarray
    phi: jnp.ndarray
    u_r: jnp.ndarray
    u_theta: jnp.ndarray
    u_phi: jnp.ndarray

    @classmethod
    def initialize_uniform(
        cls,
        num_particles: int,
        epsilon: float,
        r_max: float,
        particle_mass: float = 1.0,
    ) -> "ParticleState":
        """Place particles uniformly in radius and theta with zero velocity.

        The particles are placed at cell centers inside the domain rather than
        directly on the regular center or the outer boundary.
        """

        if num_particles < 1:
            raise ValueError("num_particles must be at least 1")

        radial_spacing = float(r_max) / float(num_particles)
        # place the particle centers uniformly across the physical interval 0 <= r <= r_max
        radial_positions = radial_spacing * (jnp.arange(int(num_particles)) + 0.5)
        # keep the particles off the exact center and off the exact outer boundary by using cell centers
        theta_positions = jnp.linspace(0.0, jnp.pi, int(num_particles))
        zeros = jnp.zeros_like(radial_positions)
        masses = jnp.full_like(radial_positions, float(particle_mass))

        return cls(
            mass=masses,
            r=radial_positions,
            theta=theta_positions,
            phi=zeros,
            u_r=zeros,
            u_theta=zeros,
            u_phi=zeros,
        )

    def count(self) -> int:
        """Return the current number of active particles."""

        return int(self.r.shape[0])

    def with_updated_radial_state(
        self,
        radial_positions: jnp.ndarray,
        radial_momentum: jnp.ndarray,
    ) -> "ParticleState":
        """Replace the evolved radial variables while keeping angular labels."""

        return ParticleState(
            mass=self.mass,
            r=radial_positions,
            theta=self.theta,
            phi=self.phi,
            u_r=radial_momentum,
            u_theta=self.u_theta,
            u_phi=self.u_phi,
        )

    def with_updated_orbital_state(
        self,
        radial_positions: jnp.ndarray,
        azimuthal_angles: jnp.ndarray,
        radial_momentum: jnp.ndarray,
    ) -> "ParticleState":
        """Replace the evolved orbit variables used in fixed-metric tests.

        The Schwarzschild validation notebook keeps ``u_phi`` fixed while
        evolving ``(r, phi, u_r)``. This helper keeps that update readable.
        """

        return ParticleState(
            mass=self.mass,
            r=radial_positions,
            theta=self.theta,
            phi=azimuthal_angles,
            u_r=radial_momentum,
            u_theta=self.u_theta,
            u_phi=self.u_phi,
        )

    def apply_absorbing_boundary(self, r_min: float, r_max: float) -> "ParticleState":
        """Delete particles that have moved outside the physical radial domain."""

        keep_particle = jnp.logical_and(self.r >= float(r_min), self.r <= float(r_max))

        return ParticleState(
            mass=self.mass[keep_particle],
            r=self.r[keep_particle],
            theta=self.theta[keep_particle],
            phi=self.phi[keep_particle],
            u_r=self.u_r[keep_particle],
            u_theta=self.u_theta[keep_particle],
            u_phi=self.u_phi[keep_particle],
        )




def compute_dphi_dt( lapse_at_particle, azimuthal_momentum,
    metric_A_at_particle, radial_position, lorentz_factor):

    return lapse_at_particle * azimuthal_momentum / (
        radial_position**2 * metric_A_at_particle**2 * lorentz_factor
    )


@partial(jax.jit, static_argnames=("shape_mode",))
def compute_particle_derivatives(
    particles,
    fields,
    grid,
    schwarzschild_mass,
    shape_mode="nearest",
):

    A_at_particle = interpolate_to_particle(fields.A, particles.r, grid, shape_mode=shape_mode)
    lapse_at_particle = interpolate_to_particle(fields.lapse, particles.r, grid, shape_mode=shape_mode)
    shift_at_particle = interpolate_to_particle(fields.shift, particles.r, grid, shape_mode=shape_mode)
    # interpolate quantities to particle positions

    dA_dr = compute_metric_radial_derivative( fields.A, schwarzschild_mass, grid, exact_exterior_points = None )
    # compute the radial derivative of the metric function A, which is needed for the particle acceleration

    d_lapse_dr = centered_first_derivative(fields.lapse, grid.dr)
    d_shift_dr = centered_first_derivative(fields.shift, grid.dr)
    dA_dr_at_particle = interpolate_to_particle(dA_dr, particles.r, grid, shape_mode=shape_mode)
    d_lapse_dr_at_particle = interpolate_to_particle(d_lapse_dr, particles.r, grid, shape_mode=shape_mode)
    d_shift_dr_at_particle = interpolate_to_particle(d_shift_dr, particles.r, grid, shape_mode=shape_mode)
    # compute the radial derivatives of the metric functions and interpolate to particle positions

    safe_r_particle = safe_radius(particles.r, grid.epsilon)
    # ensure r values are not too close to zero to avoid numerical issues

    W = jnp.sqrt( 1.0 + particles.u_r**2 / A_at_particle**2
        + particles.u_phi**2 / (safe_r_particle**2 * A_at_particle**2) )
    # compute the lorzentz factor for the particles

    dr_dt = lapse_at_particle * particles.u_r / (A_at_particle**2 * W) - shift_at_particle
    # compute the radial velocity of the particles using the geodesic equations in the given metric

    dphi_dt = compute_dphi_dt( lapse_at_particle, particles.u_phi, A_at_particle, 
            safe_r_particle, W )
    # compute the azimuthal velocity of the particles using the geodesic equations in the given metric

    du_r_dt = -W * d_lapse_dr_at_particle + particles.u_r * d_shift_dr_at_particle
    du_r_dt = du_r_dt + ( lapse_at_particle * particles.u_r**2 * dA_dr_at_particle / (A_at_particle**3 * W))
    du_r_dt = du_r_dt + (lapse_at_particle * particles.u_phi**2 / W
        * (1.0 / (safe_r_particle**3 * A_at_particle**2)
            + dA_dr_at_particle / (safe_r_particle**2 * A_at_particle**3)) )
    # compute the radial acceleration of the particles using the geodesic equations in the given metric

    return ParticleDerivativeState(dr_dt=dr_dt, dphi_dt=dphi_dt, du_r_dt=du_r_dt)
