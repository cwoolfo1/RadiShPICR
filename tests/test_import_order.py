import os
import subprocess
import sys


def test_package_defaults_to_jax_x64_when_environment_is_unset():
    env = os.environ.copy()
    env.pop("JAX_ENABLE_X64", None)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import RadiShPICR, jax; "
                "print(jax.config.read('jax_enable_x64'))"
            ),
        ],
        cwd=os.path.dirname(os.path.dirname(__file__)),
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "True"


def test_package_respects_explicit_jax_x64_environment_setting():
    env = os.environ.copy()
    env["JAX_ENABLE_X64"] = "0"

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import RadiShPICR, jax; "
                "print(jax.config.read('jax_enable_x64'))"
            ),
        ],
        cwd=os.path.dirname(os.path.dirname(__file__)),
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "False"


def test_radial_electric_solver_imports_without_relativity_metric_cycle():
    from RadiShPICR.evolve import step, step_rk4
    from RadiShPICR.ConstraintBasedRelativity import calculate_metric

    assert callable(step)
    assert callable(step_rk4)
    assert callable(calculate_metric)
