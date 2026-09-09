"""Central JAX precision configuration for all quantum models.

Importing this module (for its side effect) applies the precision setting
exactly once, replacing the per-model-file ``jax.config.update`` calls.

64-bit precision is the historical default. Set ``QUOPTUNA_JAX_X64=0`` before
quoptuna is imported to train in float32, which halves memory and speeds up
simulation with typically negligible accuracy impact for classification.

Shot-based models additionally need :func:`sampling_precision`; see below.
"""

import contextlib
import os

import jax
import numpy as np

_FALSY = {"0", "false", "no", "off"}
_TRUTHY = {"1", "true", "yes", "on"}


def jax_x64_enabled() -> bool:
    """Whether 64-bit JAX is requested (``QUOPTUNA_JAX_X64``, default on)."""
    return os.environ.get("QUOPTUNA_JAX_X64", "1").strip().lower() not in _FALSY


jax.config.update("jax_enable_x64", jax_x64_enabled())


def _numpy_default_int_is_64bit() -> bool:
    """Whether NumPy's platform default integer is 64-bit.

    True on Linux and macOS (including Apple silicon), False on Windows,
    where NumPy's default integer is int32 regardless of the interpreter.
    """
    return np.array([0]).dtype == np.int64


def sampling_needs_float32() -> bool:
    """Whether shot-based circuits must run with 64-bit JAX disabled.

    PennyLane evaluates finite-shot measurements (``qml.sample``) through a
    host callback whose result dtype is declared from JAX's x64 setting: with
    x64 enabled it declares int64. NumPy produces the actual samples, and on
    Windows NumPy's default integer is int32, so the callback rejects its own
    result with "Incorrect output dtype for return value #0: Expected: int64,
    Actual: int32". The mismatch is dtype-driven, not platform-driven, so it
    is detected rather than keyed off ``sys.platform``: it can only occur when
    x64 is on *and* NumPy's default integer is narrower than that.

    ``QUOPTUNA_SAMPLING_FLOAT32`` overrides the probe (``1``/``0``); the
    default, ``auto``, detects it.
    """
    override = os.environ.get("QUOPTUNA_SAMPLING_FLOAT32", "auto").strip().lower()
    if override in _TRUTHY:
        return True
    if override in _FALSY:
        return False
    return bool(jax.config.jax_enable_x64) and not _numpy_default_int_is_64bit()


def sampling_precision():
    """Context manager under which shot-based circuits can be evaluated.

    Disables 64-bit JAX for the duration where :func:`sampling_needs_float32`
    says it is required, and is a no-op everywhere else — so models keep full
    x64 precision on platforms that support sampling under it. The global
    setting is restored on exit.
    """
    if sampling_needs_float32():
        from jax.experimental import disable_x64

        return disable_x64()
    return contextlib.nullcontext()
