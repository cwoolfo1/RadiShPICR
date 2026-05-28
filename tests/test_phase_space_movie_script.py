from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT_PATH = (
    Path(__file__).resolve().parents[3]
    / "runs"
    / "TWO_STREAM_WITH_GR_MAY27"
    / "make_phase_space_movie.py"
)


def load_movie_script_module():
    spec = importlib.util.spec_from_file_location("make_phase_space_movie", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_snapshot(species_dir: Path, step: int, radial_positions, radial_momenta):
    species_dir.mkdir(parents=True, exist_ok=True)
    snapshot = np.column_stack([radial_positions, radial_momenta])
    np.save(species_dir / f"phase_space_step_{step:06d}.npy", snapshot)


def test_discover_common_steps_and_species(tmp_path):
    module = load_movie_script_module()
    phase_space_dir = tmp_path / "phase_space"

    write_snapshot(phase_space_dir / "electrons", 0, [0.1, 0.2], [0.01, -0.02])
    write_snapshot(phase_space_dir / "electrons", 2, [0.3, 0.4], [0.03, -0.04])
    write_snapshot(phase_space_dir / "protons", 0, [0.15, 0.25], [0.0, 0.0])
    write_snapshot(phase_space_dir / "protons", 1, [0.35, 0.45], [0.0, 0.0])

    species_names, common_steps = module.discover_species_and_steps(phase_space_dir)

    assert species_names == ["electrons", "protons"]
    assert common_steps == [0]


def test_load_phase_space_frames_reads_common_snapshots(tmp_path):
    module = load_movie_script_module()
    phase_space_dir = tmp_path / "phase_space"

    write_snapshot(phase_space_dir / "electrons", 0, [0.1, 0.2], [0.01, -0.02])
    write_snapshot(phase_space_dir / "electrons", 1, [0.3, 0.4], [0.03, -0.04])
    write_snapshot(phase_space_dir / "protons", 0, [0.15, 0.25], [0.0, 0.0])
    write_snapshot(phase_space_dir / "protons", 1, [0.35, 0.45], [0.0, 0.0])

    frames = module.load_phase_space_frames(phase_space_dir)

    assert len(frames) == 2
    assert frames[0]["step"] == 0
    assert frames[1]["step"] == 1
    assert np.allclose(frames[0]["species"]["electrons"], np.array([[0.1, 0.01], [0.2, -0.02]]))
    assert np.allclose(frames[1]["species"]["protons"], np.array([[0.35, 0.0], [0.45, 0.0]]))


def test_compute_axis_limits_uses_all_species_and_frames(tmp_path):
    module = load_movie_script_module()
    phase_space_dir = tmp_path / "phase_space"

    write_snapshot(phase_space_dir / "electrons", 0, [1.0, 2.0], [-0.1, 0.2])
    write_snapshot(phase_space_dir / "protons", 0, [0.5, 3.0], [-0.3, 0.4])

    frames = module.load_phase_space_frames(phase_space_dir)
    xlim, ylim = module.compute_axis_limits(frames)

    assert np.allclose(xlim, (0.5, 3.0))
    assert np.allclose(ylim, (-0.3, 0.4))
