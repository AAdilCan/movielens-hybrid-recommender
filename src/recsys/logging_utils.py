"""Small logging helper so every module logs consistently.

I prefer a single ``get_logger`` over ``logging.basicConfig`` scattered around
because the CLI, the library code and the tests can all share one format
without fighting over the root logger.
"""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False
_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%H:%M:%S"


def configure_logging(level: int = logging.INFO) -> None:
    """Attach a single stream handler to the package root logger once."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    root = logging.getLogger("recsys")
    root.setLevel(level)
    root.addHandler(handler)
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger namespaced under ``recsys``.

    Accepts either a bare module name or a dotted ``__name__``; the result is
    always a child of the ``recsys`` logger so :func:`configure_logging` controls
    formatting and level for the whole package.
    """
    configure_logging()
    short = name.split(".")[-1] if name != "__main__" else name
    return logging.getLogger(f"recsys.{short}")
