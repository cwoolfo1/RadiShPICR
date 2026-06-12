from pathlib import Path

import numpy as np


def write_metric_fields(U_state, output_folder, step, time=None):
    A, phi, alpha, Krr, beta_over_r, Er, source_terms, r_grid = U_state
    mass_density, charge_density, Srr, Sr = source_terms

    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    filename = f"metric_fields_step_{int(step):06d}.npz"
    snapshot_path = output_path / filename
    output_time = np.nan if time is None else float(time)

    np.savez_compressed(
        snapshot_path,
        r=np.asarray(r_grid),
        A=np.asarray(A),
        phi=np.asarray(phi),
        alpha=np.asarray(alpha),
        Krr=np.asarray(Krr),
        beta_over_r=np.asarray(beta_over_r),
        Er=np.asarray(Er),
        mass_density=np.asarray(mass_density),
        charge_density=np.asarray(charge_density),
        Srr=np.asarray(Srr),
        Sr=np.asarray(Sr),
        step=int(step),
        time=output_time,
    )

    return snapshot_path
