"""Task-type (binary vs multiclass) single source of truth.

A ``TaskSpec`` captures everything downstream code needs to branch on the
class structure of the target: the kind of task, the number of classes, the
original label names in encoded order, and (for multiclass) which class the
user designated as the favorable outcome for fairness auditing.

Encoding conventions:
- binary: labels are encoded to {-1, +1} (quantum-model convention);
  ``class_labels`` is ``[neg, pos]`` so index 0 -> -1 and index 1 -> +1.
- multiclass: labels are encoded to integer codes 0..K-1 in sorted order of
  their string representation; ``class_labels[code]`` is the original name.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

BINARY_N_CLASSES = 2


@dataclass(frozen=True)
class TaskSpec:
    kind: Literal["binary", "multiclass"]
    n_classes: int
    # Original label names (as strings), index == encoded code for multiclass,
    # [neg, pos] for binary.
    class_labels: tuple[str, ...]
    favorable_class: str | None = None
    favorable_code: int | None = None

    @property
    def is_multiclass(self) -> bool:
        return self.kind == "multiclass"

    def encoded_classes(self) -> list[int]:
        """Encoded label values in class order."""
        if self.kind == "binary":
            return [-1, 1]
        return list(range(self.n_classes))

    def display_label(self, code: Any) -> str:
        """Original label name for an encoded value."""
        code = int(code)
        if self.kind == "binary":
            return self.class_labels[0] if code == -1 else self.class_labels[1]
        return self.class_labels[code]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["class_labels"] = list(self.class_labels)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> TaskSpec:
        return cls(
            kind=data["kind"],
            n_classes=int(data["n_classes"]),
            class_labels=tuple(str(c) for c in data["class_labels"]),
            favorable_class=data.get("favorable_class"),
            favorable_code=data.get("favorable_code"),
        )

    @classmethod
    def from_target(
        cls,
        y: Any,
        label_mapping: dict | None = None,
        favorable_class: Any = None,
    ) -> TaskSpec:
        """Derive a TaskSpec from the ORIGINAL (pre-encoding) target values.

        Deterministic: multiclass codes follow sorted order of the string
        representation of the unique values, so the spec can be re-derived
        from the same data after a restart.
        """
        values = np.asarray(y).ravel()
        unique = sorted({str(v) for v in values})
        n_classes = len(unique)

        if n_classes < BINARY_N_CLASSES:
            msg = f"Target must have at least 2 classes, found {n_classes}"
            raise ValueError(msg)

        if n_classes == BINARY_N_CLASSES:
            if label_mapping:
                neg, pos = str(label_mapping["neg"]), str(label_mapping["pos"])
            else:
                neg, pos = unique[0], unique[1]
            return cls(kind="binary", n_classes=2, class_labels=(neg, pos))

        favorable = str(favorable_class) if favorable_class is not None else None
        favorable_code = None
        if favorable is not None:
            if favorable not in unique:
                msg = f"favorable_class '{favorable}' not found in target values {unique}"
                raise ValueError(msg)
            favorable_code = unique.index(favorable)
        return cls(
            kind="multiclass",
            n_classes=n_classes,
            class_labels=tuple(unique),
            favorable_class=favorable,
            favorable_code=favorable_code,
        )

    def encode(self, y: Any) -> np.ndarray:
        """Encode original target values to the task's integer codes."""
        as_str = np.asarray(y).astype(str).ravel()
        if self.kind == "binary":
            return np.where(as_str == self.class_labels[1], 1, -1)
        code_of = {label: code for code, label in enumerate(self.class_labels)}
        try:
            return np.array([code_of[v] for v in as_str])
        except KeyError as e:
            msg = f"Target value {e} not in known classes {list(self.class_labels)}"
            raise ValueError(msg) from e
