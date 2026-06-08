"""
log.py — one place that decides how livescore talks to the console.

The split follows the standard Python convention: **libraries emit, the
application configures.**

- Engine/background modules (mrt_controller, llm_style_director, voice_analyzer,
  engine) call ``get_logger(name)`` and just emit. They NEVER attach handlers or
  set levels — that's not their job.
- The entrypoints (main.py, app.py) call ``configure()`` exactly once to attach a
  single clean console handler.

Why this keeps two very different audiences happy:

- **Live terminal** — ``configure()`` formats records as just ``%(message)s`` and
  the messages still carry their ``[MRT2]`` / ``[LLM]`` tags, so a performer sees
  exactly what they always have. ``--verbose`` flips INFO→DEBUG to surface the
  chatty re-seed / hold-scene lines.
- **pytest** — the suite never calls ``configure()``. The package logger carries a
  ``NullHandler`` (added at import), so with no configuration every record is
  swallowed: clean test stdout, no ``[MRT2] FATAL …`` bleed. ``caplog`` still sees
  the records, so resilience tests can assert on them.
"""

from __future__ import annotations

import logging
import threading

_ROOT = "livescore"

# A NullHandler on the package logger means "no configuration => silent" (the
# library default). This is what keeps pytest stdout clean without any test-side
# setup, while still letting caplog capture records.
logging.getLogger(_ROOT).addHandler(logging.NullHandler())

_configured = False
_configure_lock = threading.Lock()


def get_logger(name: str) -> logging.Logger:
    """A namespaced logger, e.g. get_logger("mrt") -> 'livescore.mrt'.

    Use the short module tag (mrt, llm, voice, engine); the visible ``[MRT2]``
    style prefix stays in the message text so the live output is unchanged."""
    return logging.getLogger(f"{_ROOT}.{name}")


def configure(verbose: bool = False) -> None:
    """Attach one clean console handler to the livescore logger. Idempotent.

    Called by the application entrypoints (main.py / app.py), never by library
    modules or tests. ``verbose=True`` lowers the threshold to DEBUG so the
    chatty operational lines (re-seed, scene-hold) become visible. Idempotent
    about the handler (added once), but a repeat call DOES re-apply the level, so
    a later ``configure(verbose=True)`` can loosen an earlier INFO setup."""
    global _configured
    logger = logging.getLogger(_ROOT)
    with _configure_lock:
        # Level can be re-tightened/loosened on repeat calls (e.g. a later
        # --verbose); the handler is only ever added once.
        logger.setLevel(logging.DEBUG if verbose else logging.INFO)
        if _configured:
            return
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        # Don't also bubble up to the root logger (which a host app like Flask may
        # have configured) — that would double-print every line.
        logger.propagate = False
        _configured = True
