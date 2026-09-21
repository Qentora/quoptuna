"""Surface SHAP's per-row progress to callers instead of only the terminal.

``shap.Explainer.__call__`` walks its rows through ``show_progress``, which
renders a tqdm bar to stdout and offers no hook for anything else. On a
variational quantum model an explanation is minutes of work, so that bar is the
only sign of life — and it is visible to whoever started the server, not to the
person waiting in the browser.

Wrapping that iterator gives exact per-row progress with no change to what SHAP
computes: rows are still explained one at a time, in the same order.

The wrapper reads its sink from a :class:`~contextvars.ContextVar`, so
concurrent analyses each report to their own job rather than to whichever
patched last. The patch itself is applied once and left in place; with no sink
set it delegates untouched.
"""

from __future__ import annotations

from contextlib import contextmanager, suppress
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from collections.abc import Iterator

#: ``(rows_done, rows_total)``. Called once per explained row.
ProgressSink = Callable[[int, int], None]

_sink: ContextVar[ProgressSink | None] = ContextVar("shap_progress_sink", default=None)

#: Marks our wrapper so a second install is a no-op rather than a wrapper
#: around a wrapper.
_MARKER = "_quoptuna_tracked"


def _install() -> bool:
    """Patch SHAP's progress iterator. Returns whether progress can be reported.

    Pinned to shap 0.46's private module layout. If that moves, progress simply
    is not reported — an analysis must not fail over a progress bar.
    """
    try:
        from shap.explainers import _explainer as shap_explainer

        original = shap_explainer.show_progress
    except (ImportError, AttributeError):
        return False
    if getattr(original, _MARKER, False):
        return True

    def tracked(iterable: Any, total: Any = None, *args: Any, **kwargs: Any) -> Iterator[Any]:
        inner = original(iterable, total, *args, **kwargs)
        sink = _sink.get()
        if sink is None:
            return inner

        def counting() -> Iterator[Any]:
            for done, item in enumerate(inner, start=1):
                # Yield first: the row is only finished once the caller's body
                # has run and asked for the next one.
                yield item
                with suppress(Exception):  # reporting must never break the run
                    sink(done, int(total or 0))

        return counting()

    setattr(tracked, _MARKER, True)
    shap_explainer.show_progress = tracked
    return True


@contextmanager
def report_progress(sink: ProgressSink) -> Iterator[None]:
    """Report each explained row to ``sink`` for the duration of the block."""
    if not _install():
        yield
        return
    token = _sink.set(sink)
    try:
        yield
    finally:
        _sink.reset(token)
