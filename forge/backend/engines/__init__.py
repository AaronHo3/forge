"""
engines - pluggable music-generation backends for Forge.

`make_backend(name)` is the one place that knows which engines exist. Everything
else (ForgeCore, games, server) depends only on the `Backend` protocol, so adding
Stable Audio 3 or Suno is: write engines/<x>.py, register it here, done.

Engine selection precedence:
    explicit arg  >  $FORGE_BACKEND env var  >  "mrt2" default

Imports are lazy (inside the branch) on purpose: a heavy engine like Stable Audio
3 pulls in torch, and the Suno engine needs network config - we must not pay that
cost just to construct a different engine.
"""

from __future__ import annotations

import os

from .base import Backend, Embedding

__all__ = ["Backend", "Embedding", "make_backend"]


def make_backend(name: str | None = None) -> Backend:
    """Construct the selected generation engine. See module docstring for order."""
    name = (name or os.environ.get("FORGE_BACKEND") or "mrt2").strip().lower()

    if name in ("mrt2", "mrt", "magenta"):
        from ..mrt_worker import MRTWorker  # noqa: PLC0415
        return MRTWorker()

    if name in ("sa3", "stableaudio", "stable-audio"):
        from .sa3 import SA3Backend  # noqa: PLC0415
        return SA3Backend()

    # Planned engines - wired in during their respective challenge tracks:
    #   if name in ("suno",): from .suno import SunoBackend ...

    raise ValueError(
        f"unknown FORGE_BACKEND '{name}' (known: mrt2, sa3; planned: suno)"
    )
