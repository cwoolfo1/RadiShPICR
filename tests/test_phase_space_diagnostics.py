import numpy as np
import jax.numpy as jnp

from RadiShPICR.diagnostics import write_phase_space
from RadiShPICR.particles.particle_species import particle_species


def test_write_phase_space_writes_radial_snapshot(tmp_path):
    particles = particle_species(
        name="ions",
        number_of_particles=3,
        charge=1.0,
        mass=2.0,
        temperature=0.0,
        r=jnp.array([0.2, 0.4, 0.6]),
        phi=jnp.zeros(3),
        u_r=jnp.array([0.01, -0.02, 0.03]),
        u_phi=jnp.zeros(3),
    )

    output_path = write_phase_space(
        particles,
        tmp_path,
        step=12,
        time=0.6,
    )

    assert output_path.name == "phase_space_ions_step_000012.npz"
    assert output_path.parent == tmp_path

    with np.load(output_path) as snapshot:
        assert np.allclose(snapshot["r"], np.asarray(particles.r))
        assert np.allclose(snapshot["vr"], np.asarray(particles.u_r))
        assert int(snapshot["step"]) == 12
        assert np.isclose(float(snapshot["time"]), 0.6)
        assert str(snapshot["species_name"]) == "ions"
