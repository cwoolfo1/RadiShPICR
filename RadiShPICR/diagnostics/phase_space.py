from pathlib import Path

import numpy as np


def write_phase_space(particles, output_folder, step, time=None):
    """Write one radial phase-space snapshot for a particle species.

    The timestepper evolves ``r`` and ``u_r`` directly.  This diagnostic writes
    those current particle arrays without changing the particle state, so it can
    be called after any timestep in a loop.
    """

    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    species_name = particles.get_name()
    filename = f"phase_space_{species_name}_step_{int(step):06d}.npz"
    snapshot_path = output_path / filename

    # Store u_r under the short phase-space name vr for plotting convenience.
    radial_positions = np.asarray(particles.r)
    radial_velocities = np.asarray(particles.u_r)
    output_time = np.nan if time is None else float(time)

    np.savez_compressed(
        snapshot_path,
        r=radial_positions,
        vr=radial_velocities,
        step=int(step),
        time=output_time,
        species_name=species_name,
    )

    return snapshot_path
