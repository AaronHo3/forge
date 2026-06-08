"""
voice_analyzer.py
-----------------
Captures microphone input in real time and extracts acoustic features
that describe the emotional and physical qualities of the narrator's voice.

Features extracted every ~100ms:
  - energy      : how loud/intense the voice is (RMS)
  - pitch        : fundamental frequency (Hz), 0 if unvoiced
  - speech_rate  : how fast words are coming (onset strength)
  - brightness   : spectral centroid — bright/excited vs dark/calm
  - is_silent    : True when the narrator pauses
"""

import queue
import threading
from collections import deque
import numpy as np
import sounddevice as sd
import librosa

from log import get_logger

log = get_logger("voice")

# MRT2 requires 48kHz — match here so we don't have to resample
SAMPLE_RATE = 48_000
# ~100ms of audio per analysis window
BLOCK_SIZE = 4_800


class VoiceFeatures:
    """Snapshot of voice features at a moment in time."""
    def __init__(self):
        self.energy: float = 0.0       # 0.0 – 1.0 (normalised RMS)
        self.pitch: float = 0.0        # Hz, 0 if unvoiced/silent
        self.speech_rate: float = 0.0  # 0.0 – 1.0
        self.brightness: float = 0.0   # 0.0 – 1.0
        self.is_silent: bool = True

    def __repr__(self):
        bar = lambda v: "█" * int(v * 20) + "░" * (20 - int(v * 20))
        return (
            f"energy    {bar(self.energy)}  {self.energy:.3f}\n"
            f"pitch     {self.pitch:6.1f} Hz\n"
            f"rate      {bar(self.speech_rate)}  {self.speech_rate:.3f}\n"
            f"bright    {bar(self.brightness)}  {self.brightness:.3f}\n"
            f"silent    {'yes' if self.is_silent else 'no'}"
        )


class VoiceAnalyzer:
    """
    Continuously captures microphone audio and provides the latest
    VoiceFeatures via `get_features()`.

    Usage:
        analyzer = VoiceAnalyzer()
        analyzer.start()
        features = analyzer.get_features()   # call in your main loop
        analyzer.stop()
    """

    # Silence threshold: below this RMS the narrator is considered paused.
    SILENCE_THRESHOLD = 0.005

    # Rough spectral centroid range for speech (Hz).
    CENTROID_MIN = 200.0
    CENTROID_MAX = 3500.0

    def __init__(self, device=None):
        self._device = device
        self._queue: queue.Queue = queue.Queue(maxsize=8)
        self._features = VoiceFeatures()
        self._lock = threading.Lock()
        self._running = False
        self._stream = None
        self._thread = None
        # Rolling raw audio for Whisper transcription (50 blocks ≈ 5 seconds).
        # _raw_lock guards it: the mic thread appends while the LLM thread
        # snapshots, and `list(deque)` during a concurrent append raises
        # "deque mutated during iteration" in CPython without it.
        self._raw_blocks: deque = deque(maxlen=50)
        self._raw_lock = threading.Lock()
        # Latest sounddevice status flag, set (plain atomic assignment) by the
        # audio thread and logged by the worker thread — the callback itself must
        # stay lock-free, so it never touches the logging machinery directly.
        self._pending_status = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        """Open the mic stream and begin feature extraction."""
        self._running = True
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            channels=1,
            dtype="float32",
            device=self._device,
            callback=self._audio_callback,
        )
        self._stream.start()
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()
        log.info(f"[VoiceAnalyzer] Listening at {SAMPLE_RATE} Hz, "
                 f"block size {BLOCK_SIZE} samples (~{BLOCK_SIZE/SAMPLE_RATE*1000:.0f}ms)")

    def stop(self):
        """Shut down the mic stream."""
        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()

    def get_features(self) -> VoiceFeatures:
        """Return the most recently computed feature snapshot (thread-safe)."""
        with self._lock:
            # Return a shallow copy so the caller can't mutate our state
            f = VoiceFeatures()
            f.energy = self._features.energy
            f.pitch = self._features.pitch
            f.speech_rate = self._features.speech_rate
            f.brightness = self._features.brightness
            f.is_silent = self._features.is_silent
            return f

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _append_raw(self, block: np.ndarray) -> None:
        """Append a raw audio block under the lock, so a concurrent snapshot in
        get_audio_for_transcription never iterates the deque mid-mutation."""
        with self._raw_lock:
            self._raw_blocks.append(block)

    def get_audio_for_transcription(self, seconds: float = 4.0) -> np.ndarray:
        """Return the last `seconds` of raw mono 48kHz audio for Whisper."""
        n_blocks = int(seconds * 10)  # 10 blocks per second at 100ms each
        with self._raw_lock:
            blocks = list(self._raw_blocks)[-n_blocks:]
        if not blocks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(blocks)

    def _audio_callback(self, indata, frames, time_info, status):
        """Called by sounddevice on the audio thread — keep it fast and lock-free.
        We must NOT call logging here (it takes locks / may do blocking I/O, which
        PortAudio forbids in the callback); just hand the status off for the worker
        thread to log."""
        if status:
            self._pending_status = str(status)   # plain assignment: atomic, lock-free
        block = indata[:, 0].copy()
        self._append_raw(block)   # locked: safe against concurrent snapshots
        try:
            self._queue.put_nowait(block)
        except queue.Full:
            pass  # Drop frame — processing is behind; not critical

    def _process_loop(self):
        """Worker thread: pulls audio blocks and extracts features."""
        last_logged_status = None
        while self._running:
            # Log any audio-stream warning the callback flagged — off the audio
            # thread, and only on change so a sustained overflow doesn't spam.
            status = self._pending_status
            if status and status != last_logged_status:
                log.warning(f"[VoiceAnalyzer] sounddevice status: {status}")
                last_logged_status = status
            try:
                block = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            features = self._extract(block)
            with self._lock:
                self._features = features

    def _extract(self, block: np.ndarray) -> VoiceFeatures:
        f = VoiceFeatures()

        # ── Silence / Energy ──────────────────────────────────────────
        rms = float(np.sqrt(np.mean(block ** 2)))
        f.is_silent = rms < self.SILENCE_THRESHOLD

        # Normalise: typical spoken voice peaks around 0.1–0.3 RMS.
        # We clip at 0.3 so shouting still maps to 1.0.
        f.energy = float(np.clip(rms / 0.3, 0.0, 1.0))

        if f.is_silent:
            # Don't bother with pitch/rate when silent — avoids noise artifacts
            return f

        # ── Pitch (fundamental frequency) ────────────────────────────
        # librosa.pyin gives reliable voiced/unvoiced detection.
        try:
            f0, voiced_flag, _ = librosa.pyin(
                block,
                fmin=librosa.note_to_hz("C2"),   # ~65 Hz
                fmax=librosa.note_to_hz("C6"),   # ~1047 Hz
                sr=SAMPLE_RATE,
                fill_na=0.0,
            )
            voiced_f0 = f0[voiced_flag] if voiced_flag is not None else []
            f.pitch = float(np.mean(voiced_f0)) if len(voiced_f0) > 0 else 0.0
        except Exception:
            f.pitch = 0.0

        # ── Speech rate (onset strength) ──────────────────────────────
        # onset_strength returns a 1-D array; its mean is a proxy for
        # how many syllable-like events occurred per unit time.
        onset_env = librosa.onset.onset_strength(y=block, sr=SAMPLE_RATE)
        # Typical range 0–5; normalise to 0–1
        f.speech_rate = float(np.clip(float(np.mean(onset_env)) / 5.0, 0.0, 1.0))

        # ── Brightness (spectral centroid) ────────────────────────────
        centroid = float(np.mean(
            librosa.feature.spectral_centroid(y=block, sr=SAMPLE_RATE)
        ))
        f.brightness = float(np.clip(
            (centroid - self.CENTROID_MIN) / (self.CENTROID_MAX - self.CENTROID_MIN),
            0.0, 1.0,
        ))

        return f


# ------------------------------------------------------------------
# Quick test — run this file directly to see live feature output
# ------------------------------------------------------------------
if __name__ == "__main__":
    import time

    print("Speak into your microphone. Press Ctrl+C to stop.\n")
    analyzer = VoiceAnalyzer()
    analyzer.start()

    try:
        while True:
            f = analyzer.get_features()
            # Clear terminal line and print features
            print("\033[2J\033[H")  # clear screen
            print("── Score the Story: Voice Analyzer ──\n")
            print(f)
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        analyzer.stop()
        print("\nStopped.")
