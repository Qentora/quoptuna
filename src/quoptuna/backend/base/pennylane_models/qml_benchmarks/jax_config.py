"""Central JAX precision configuration for all quantum models.

Importing this module (for its side effect) applies the precision setting
exactly once, replacing the per-model-file ``jax.config.update`` calls.

64-bit precision is the historical default. Set ``QUOPTUNA_JAX_X64=0`` before
quoptuna is imported to train in float32, which halves memory and speeds up
simulation with typically negligible accuracy impact for classification.
"""

import os

import jax

_FALSY = {"0", "false", "no", "off"}


def jax_x64_enabled() -> bool:
    """Whether 64-bit JAX is requested (``QUOPTUNA_JAX_X64``, default on)."""
    return os.environ.get("QUOPTUNA_JAX_X64", "1").strip().lower() not in _FALSY


jax.config.update("jax_enable_x64", jax_x64_enabled())
