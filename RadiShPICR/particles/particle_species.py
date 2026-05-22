import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
class particle_species:
    """Spherical 1D2V particle species stored as a JAX pytree.

    The evolved orbit variables are ``r``, ``phi``, and ``u_r``.  The
    azimuthal momentum ``u_phi`` is carried as a conserved particle label.
    """

    def __init__(
        self,
        name,
        number_of_particles,
        charge,
        mass,
        temperature,
        r,
        phi,
        u_r,
        u_phi,
        weight=1.0,
        r_min=0.0,
        r_max=None,
        dr=None,
        shape=1,
        dt=0.0,
    ):
        self.name = name
        self.number_of_particles = int(number_of_particles)
        self.N_particles = self.number_of_particles
        self.charge = charge
        self.mass = mass
        self.temperature = temperature
        self.T = temperature
        self.weight = weight
        self.r_min = r_min
        self.r_max = r_max
        self.dr = dr
        self.shape = shape
        self.dt = dt

        self.r = jnp.asarray(r)
        self.phi = jnp.asarray(phi, dtype=self.r.dtype)
        self.u_r = jnp.asarray(u_r, dtype=self.r.dtype)
        self.u_phi = jnp.asarray(u_phi, dtype=self.r.dtype)

        self._validate_array_shapes()

    @classmethod
    def initialize_uniform(
        cls,
        name,
        number_of_particles,
        charge,
        mass,
        temperature,
        r_max,
        weight=1.0,
        r_min=0.0,
        dr=None,
        u_phi=0.0,
        shape=1,
        dt=0.0,
    ):
        """Place particles uniformly in radius with zero radial momentum."""

        if number_of_particles < 1:
            raise ValueError("number_of_particles must be at least 1")
        if not (float(r_max) > float(r_min)):
            raise ValueError("r_max must be greater than r_min")

        radial_spacing = (float(r_max) - float(r_min)) / float(number_of_particles)
        radial_positions = float(r_min) + radial_spacing * (
            jnp.arange(int(number_of_particles)) + 0.5
        )
        zeros = jnp.zeros_like(radial_positions)
        azimuthal_momentum = jnp.full_like(radial_positions, float(u_phi))

        return cls(
            name=name,
            number_of_particles=number_of_particles,
            charge=charge,
            mass=mass,
            temperature=temperature,
            r=radial_positions,
            phi=zeros,
            u_r=zeros,
            u_phi=azimuthal_momentum,
            weight=weight,
            r_min=r_min,
            r_max=r_max,
            dr=dr,
            shape=shape,
            dt=dt,
        )

    def _validate_array_shapes(self):
        expected_shape = self.r.shape
        for field_name in ("phi", "u_r", "u_phi"):
            field_value = getattr(self, field_name)
            if field_value.shape != expected_shape:
                raise ValueError(
                    f"{field_name} must have shape {expected_shape}, "
                    f"got {field_value.shape}"
                )
        if self.number_of_particles != int(self.r.shape[0]):
            raise ValueError(
                "number_of_particles must match the length of the particle arrays: "
                f"expected {self.r.shape[0]}, got {self.number_of_particles}"
            )

    def count(self):
        """Return the number of particles in this species."""

        return int(self.r.shape[0])

    def get_name(self):
        return self.name

    def get_charge(self):
        return self.charge * self.weight

    def get_mass(self):
        return self.mass * self.weight

    def get_weight(self):
        return self.weight

    def get_number_of_particles(self):
        return self.number_of_particles

    def get_temperature(self):
        return self.temperature

    def get_shape(self):
        return self.shape

    def get_position(self):
        return self.r, self.phi

    def get_forward_position(self):
        return self.get_position()

    def get_velocity(self):
        return self.u_r, self.u_phi

    def get_index(self):
        """Return nearest interior radial deposition indices.

        The first and last grid cells are reserved as vacuum cells by the
        relativity deposition routines, so particles map only to indices
        ``1`` through ``N - 2`` when domain metadata is available.
        """

        if self.dr is None or self.r_max is None:
            raise ValueError("get_index requires dr and r_max to be set")

        floating_index = (self.r - float(self.r_min)) / float(self.dr)
        nearest = jnp.rint(floating_index).astype(jnp.int32)
        num_points = int(round((float(self.r_max) - float(self.r_min)) / float(self.dr))) + 1
        return jnp.clip(nearest, 1, num_points - 2)

    def set_position(self, r, phi):
        self.r = jnp.asarray(r, dtype=self.r.dtype)
        self.phi = jnp.asarray(phi, dtype=self.r.dtype)
        self._validate_array_shapes()

    def set_velocity(self, u_r, u_phi):
        self.u_r = jnp.asarray(u_r, dtype=self.r.dtype)
        self.u_phi = jnp.asarray(u_phi, dtype=self.r.dtype)
        self._validate_array_shapes()

    def set_mass(self, mass):
        self.mass = mass

    def set_weight(self, weight):
        self.weight = weight

    def with_updated_radial_state(self, radial_positions, radial_momentum):
        """Replace ``r`` and ``u_r`` while preserving angular labels."""

        return self._replace(
            r=radial_positions,
            u_r=radial_momentum,
        )

    def with_updated_orbital_state(
        self,
        radial_positions,
        azimuthal_angles,
        radial_momentum,
    ):
        """Replace the evolved 1D2V orbit variables.

        ``u_phi`` is intentionally preserved by this first spherical
        implementation.
        """

        return self._replace(
            r=radial_positions,
            phi=azimuthal_angles,
            u_r=radial_momentum,
        )

    def boundary_conditions(self):
        """Boundary handling is intentionally deferred for the first pass."""

        return self

    def update_position(self):
        """Update positions using stored ``dt`` only for non-GR callers."""

        self.r = self.r + self.u_r * self.dt
        return self

    def _replace(self, **updates):
        values = {
            "r": self.r,
            "phi": self.phi,
            "u_r": self.u_r,
            "u_phi": self.u_phi,
        }
        values.update(updates)

        return type(self)(
            name=self.name,
            number_of_particles=self.number_of_particles,
            charge=self.charge,
            mass=self.mass,
            temperature=self.temperature,
            r=values["r"],
            phi=values["phi"],
            u_r=values["u_r"],
            u_phi=values["u_phi"],
            weight=self.weight,
            r_min=self.r_min,
            r_max=self.r_max,
            dr=self.dr,
            shape=self.shape,
            dt=self.dt,
        )

    def tree_flatten(self):
        children = (self.r, self.phi, self.u_r, self.u_phi)
        aux_data = (
            self.name,
            self.number_of_particles,
            self.charge,
            self.mass,
            self.temperature,
            self.weight,
            self.r_min,
            self.r_max,
            self.dr,
            self.shape,
            self.dt,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        r, phi, u_r, u_phi = children
        (
            name,
            number_of_particles,
            charge,
            mass,
            temperature,
            weight,
            r_min,
            r_max,
            dr,
            shape,
            dt,
        ) = aux_data

        return cls(
            name=name,
            number_of_particles=number_of_particles,
            charge=charge,
            mass=mass,
            temperature=temperature,
            r=r,
            phi=phi,
            u_r=u_r,
            u_phi=u_phi,
            weight=weight,
            r_min=r_min,
            r_max=r_max,
            dr=dr,
            shape=shape,
            dt=dt,
        )


ParticleSpecies = particle_species
