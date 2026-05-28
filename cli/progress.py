"""Progress bars for the ``csemsearch`` CLI.

Wraps ``tqdm`` if installed, falls back to a plain ``[N/T]`` line printer
when it isn't. Either way, output goes to stderr so it never pollutes a
machine-readable stdout (e.g. piping ``csemsearch search`` results).

The wrapper exists so command code can write ``with ProgressBar(...)``
once and have a working bar in both modes — the alternative would be
``if HAS_TQDM`` branches scattered through every verb implementation.
"""

from __future__ import annotations

import sys
from types import TracebackType
from typing import Any, Type

try:
    from tqdm import tqdm  # type: ignore[import-untyped]
    HAS_TQDM = True
except ImportError:  # pragma: no cover - exercised in test_progress.py
    tqdm = None  # type: ignore[assignment]
    HAS_TQDM = False


def _truncate_middle(s: str, width: int) -> str:
    """Middle-truncate ``s`` to fit in ``width`` characters with an ellipsis,
    keeping the start and the end visible. The end is the more important
    part for paths (the filename and extension live there)."""
    if width <= 1 or len(s) <= width:
        return s
    if width <= 3:
        return s[-width:]
    keep = width - 1  # one char for the ellipsis "…"
    # Bias toward keeping the tail (filename); split the kept chars 1/3 head,
    # 2/3 tail.
    head = max(1, keep // 3)
    tail = keep - head
    return s[:head] + "…" + s[-tail:]


class ProgressBar:
    """tqdm bar with a plain-stderr-counter fallback.

    Use as a context manager. ``update(n)`` advances the bar by ``n`` ticks
    and accepts a ``postfix`` string (rendered next to the bar by tqdm, on
    the same line by the fallback). ``set_to(processed)`` jumps the bar to
    an absolute position — useful when the underlying progress source
    reports cumulative counts rather than deltas.
    """

    POSTFIX_MAX_WIDTH = 72

    def __init__(self, total: int, *, desc: str = "", unit: str = "file") -> None:
        self.total = total
        self.desc = desc
        self.unit = unit
        self.n = 0
        self._bar: Any | None = None
        if HAS_TQDM:
            self._bar = tqdm(
                total=total,
                desc=desc,
                unit=unit,
                file=sys.stderr,
                dynamic_ncols=True,
                leave=False,
            )

    def update(self, n: int = 1, *, postfix: str | None = None) -> None:
        self.n += n
        if self._bar is not None:
            if postfix is not None:
                self._bar.set_postfix_str(
                    _truncate_middle(postfix, self.POSTFIX_MAX_WIDTH),
                    refresh=False,
                )
            self._bar.update(n)
            return
        line = f"[{self.n}/{self.total}]"
        if self.desc:
            line = f"{self.desc} {line}"
        if postfix is not None:
            line = f"{line} {_truncate_middle(postfix, self.POSTFIX_MAX_WIDTH)}"
        print(line, file=sys.stderr)

    def set_to(self, processed: int, *, postfix: str | None = None) -> None:
        delta = processed - self.n
        if delta > 0:
            self.update(delta, postfix=postfix)
        elif postfix is not None and self._bar is not None:
            self._bar.set_postfix_str(
                _truncate_middle(postfix, self.POSTFIX_MAX_WIDTH),
                refresh=True,
            )

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None

    def __enter__(self) -> "ProgressBar":
        return self

    def __exit__(
        self,
        exc_type: Type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
