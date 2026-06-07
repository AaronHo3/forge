"""
engines/base.py - the generation-engine contract.

Forge is engine-agnostic. The workbench, the games, the judge, storage, and the
whole web layer only ever speak `PromptSpec -> Clip`; they never name a model.
THIS file is the seam that makes that true: any music generator that can turn a
spec into a WAV on disk can be a Forge engine.

Implementations (each in its own module under engines/):
  - mrt2  - Magenta RealTime 2, local MLX, live style morphing   (engines/mrt.py)
  - sa3   - Stable Audio 3, local/open-weights, fast one-shots    (planned)
  - suno  - Suno API, cloud, finished songs + audio remix         (planned)

Because this is a `typing.Protocol`, engines do NOT inherit from it - they just
need the right shape (structural typing). That keeps each engine free of any
Forge import beyond the shared `PromptSpec`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..models import PromptSpec

# A style/content vector used for novelty scoring. Optional: engines with no such
# notion (e.g. a cloud API) return None, and novelty features simply no-op.
Embedding = tuple[float, ...] | None


@runtime_checkable
class Backend(Protocol):
    """One music-generation engine, behind a single blocking render call.

    The single-worker / serial-queue discipline (see mrt_worker.py) is an engine
    *implementation* detail, not part of this contract - a cloud engine may run
    fully concurrently. Forge's turn-based pacing does not depend on it.
    """

    #: short, stable identifier ("mrt2", "sa3", "suno") - used in logs/UI.
    name: str

    def start(self) -> None:
        """Prepare the engine: load weights, spin a worker thread, open a client.

        Must be safe to call more than once (idempotent). Cloud engines with
        nothing to warm up may treat this as a no-op. Heavy local models should
        load lazily on first render rather than here, so startup stays fast.
        """
        ...

    def render_blocking(
        self, spec: PromptSpec, out_path: str, timeout: float = 300.0,
    ) -> tuple[str, Embedding]:
        """Render `spec` to a WAV file at `out_path`, blocking until done.

        Returns (wav_path, embedding). `embedding` is an optional style vector for
        novelty scoring; return None when the engine has no embedding to offer.
        Raise on failure (TimeoutError on timeout) - callers handle it.
        """
        ...
