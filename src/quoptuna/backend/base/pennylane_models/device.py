"""Quantum simulator device selection with a safe fallback."""

import logging

import pennylane as qml

logger = logging.getLogger(__name__)

DEFAULT_DEV_TYPE = "default.qubit"


def resolve_dev_type(requested: str) -> str:
    """Return ``requested`` if that PennyLane device can be constructed.

    Falls back to ``default.qubit`` with a warning otherwise (e.g. the
    lightning plugin is missing from the environment). Probe once per run,
    not per trial — device construction is not free.
    """
    if requested == DEFAULT_DEV_TYPE:
        return requested
    try:
        qml.device(requested, wires=1)
    except Exception as exc:  # noqa: BLE001 - any plugin failure means fallback
        logger.warning(
            "Quantum device %r unavailable (%s); falling back to %r.",
            requested,
            exc,
            DEFAULT_DEV_TYPE,
        )
        return DEFAULT_DEV_TYPE
    return requested
