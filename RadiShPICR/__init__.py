"""RadiShPICR package."""

import os

import jax


if "JAX_ENABLE_X64" not in os.environ:
    jax.config.update("jax_enable_x64", True)
    # The metric and field solves are elliptic constraints; use 64-bit JAX
    # arithmetic by default unless the runtime environment explicitly opts out.
