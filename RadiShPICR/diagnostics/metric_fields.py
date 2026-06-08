from pathlib import Path

import numpy as np


def write_metric_fields(metric, grid, output_folder, step, time=None):
    """Write one lapse and radial metric snapshot.

    The metric solve already places ``A`` and ``lapse`` on ``grid.r_full``.
    This diagnostic records those current grid fields without changing the
    metric state, so it can be called after any timestep in a loop.
    """

    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    filename = f"metric_fields_step_{int(step):06d}.npz"
    snapshot_path = output_path / filename

    radial_grid = np.asarray(grid.r_full)
    A = np.asarray(metric.A)
    lapse = np.asarray(metric.lapse)
    output_time = np.nan if time is None else float(time)

    np.savez_compressed(
        snapshot_path,
        r=radial_grid,
        A=A,
        lapse=lapse,
        step=int(step),
        time=output_time,
    )

    return snapshot_path
