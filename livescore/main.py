"""
main.py — Score the Story
--------------------------
A narrator speaks. MRT2 composes the score in real time.

Run:
    python main.py              # native magenta-rt real-time engine (default)
    python main.py --mode midi  # legacy: drive the MRT2 AU plugin in a DAW via MIDI

Controls (keyboard, while running):
    t <scene> + Enter  →  inject a scene description (bypasses mic)
    s         + Enter  →  print a one-shot status snapshot
    c/a/d/f/g + Enter  →  hold a chord (MIDI mode)
    r         + Enter  →  release chord
    q         + Enter  →  quit

The terminal stays calm — nothing repaints in place. Type `s` whenever
you want to see the current voice features and MRT2 parameters.
"""

import argparse
import threading
import time

from voice_analyzer    import VoiceAnalyzer
from feature_mapper    import FeatureMapper
from mrt_controller    import MRTController
from llm_style_director import LLMStyleDirector
from telemetry         import Telemetry
from presets           import PRESETS, get as get_preset
from speaker_signature import SpeakerSignature
from keepsake          import SessionLog
from session_summary   import build_summary, write_summary

import log
import paths


# ── Story arc: starting chord suggestions ─────────────────────────────
# Edit these to match whatever key/mood you want for your performance.
CHORDS = {
    "c": ["C3", "E3", "G3"],          # C major — open, neutral
    "a": ["A2", "C3", "E3"],          # A minor — introspective
    "d": ["D3", "F3", "A3"],          # D minor — melancholic
    "f": ["F3", "A3", "C4"],          # F major — warm, resolved
    "g": ["G2", "B2", "D3"],          # G major — hopeful, rising
}

UPDATE_HZ = 20   # How many times per second we push new params to MRT2


def print_status(analyzer, mapper, detector=None, controller=None):
    """One-shot snapshot — printed on demand, never repainted."""
    print("\n── Voice ──────────────────────────")
    print(analyzer.get_features())
    print("\n── MRT2 Parameters ────────────────")
    print(mapper._params)
    if detector:
        print(f"\nllm:  [{detector.last_status}]"
              + ("  (mic muted)" if detector.muted else ""))
        print(f"A: {detector.current_a}")
        print(f"B: {detector.current_b}")
    if controller:
        health = controller.health() if hasattr(controller, "health") else None
        if health and not health["ok"]:
            print("\n── Engine ─────────────────────────")
            print(f"⚠  STOPPED — {health['fault']}")
        perf = controller.perf_stats()
        if perf:
            flag = "ok" if perf["realtime_ok"] else "SLOWER THAN REALTIME"
            print(f"\n── Performance ────────────────────")
            print(f"generate: {perf['gen_ms_per_chunk']:.0f} ms/chunk ({flag})")
            print(f"starves (stutters): {perf.get('starves', 0)}   underruns: {perf['underruns']}")
            if health and health.get("last_chunk_age_s") is not None:
                print(f"last chunk: {health['last_chunk_age_s']:.1f}s ago")
    print()


def main():
    parser = argparse.ArgumentParser(description="Score the Story")
    parser.add_argument(
        "--mode", choices=["midi", "python"], default="python",
        help="python = native magenta-rt real-time engine (default); "
             "midi = legacy: send MIDI CC to the MRT2 AU plugin in a DAW"
    )
    parser.add_argument(
        "--no-dashboard", action="store_true",
        help="disable the live telemetry web dashboard + JSONL logging"
    )
    parser.add_argument(
        "--port", type=int, default=8765, help="telemetry dashboard port"
    )
    parser.add_argument(
        "--preset", choices=list(PRESETS.keys()), default="storytelling",
        help="tuning bundle for the use case (timing, mapping, genre bias)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="show the chatty engine debug lines (re-seeds, scene-holds)"
    )
    args = parser.parse_args()

    # Route the engine's [MRT2]/[LLM]/[voice] diagnostics through logging. The
    # terminal UI below stays plain print(); this only governs the engine chatter.
    log.configure(verbose=args.verbose)

    preset = get_preset(args.preset)
    print(f"\n🎙  Score the Story — {args.mode.upper()} mode · preset: {preset.name}\n")

    # ── Telemetry first, so a port failure degrades to None BEFORE the
    #    components capture it (otherwise they'd hold a half-started server). ──
    telemetry = None if args.no_dashboard else Telemetry(port=args.port)
    if telemetry:
        try:
            telemetry.start()
        except Exception as e:
            print(f"⚠  Telemetry dashboard unavailable (port {args.port}): {e}")
            print("   Continuing without the live dashboard.")
            telemetry = None

    # ── Init components (all tuned by the chosen preset) ───────────────
    analyzer   = VoiceAnalyzer()
    mapper     = FeatureMapper(smoothing=preset.smoothing,
                               drums_threshold=preset.drums_threshold)
    controller = MRTController(mode=args.mode, morph_step=preset.morph_step,
                               default_a=preset.default_a, default_b=preset.default_b,
                               default_key=preset.default_key, telemetry=telemetry,
                               enable_drums=preset.enable_drums, anchor=preset.anchor,
                               axis_strength=preset.axis_strength)
    signature   = SpeakerSignature()           # voice fingerprint → song identity
    session_log = SessionLog()                  # captures the telling for the keepsake
    detector   = (LLMStyleDirector(analyzer, controller,
                                   transcribe_interval=preset.transcribe_interval,
                                   audio_window=preset.audio_window,
                                   cooldown=preset.cooldown,
                                   style_hint=preset.style_hint,
                                   telemetry=telemetry,
                                   signature=signature,
                                   session_log=session_log)
                  if args.mode == "python" else None)

    try:
        analyzer.start()      # opens the mic — the most likely live failure
    except Exception as e:
        print(f"\n✗ Could not open the microphone: {e}")
        print("  Check a mic is connected and this app has mic permission, then retry.")
        return
    controller.start()
    if detector:
        detector.start()

    # ── Control loop (runs at UPDATE_HZ, silent) ──────────────────────
    stop_event = threading.Event()

    def control_loop():
        interval = 1.0 / UPDATE_HZ
        health_ticks = 0
        while not stop_event.is_set():
            t0 = time.monotonic()
            features = analyzer.get_features()
            params   = mapper.update(features)
            controller.update(params)
            if telemetry:
                out = controller.output_stats() or {}
                telemetry.record(features, params,
                                 out.get("level", 0.0), out.get("bright", 0.0))
                # ~1Hz engine-health push (separate from the 20Hz tick stream) so
                # the dashboard can show a STOPPED/SLOW engine, not just silence.
                health_ticks += 1
                if health_ticks >= UPDATE_HZ:
                    health_ticks = 0
                    telemetry.health(controller.health(), controller.perf_stats())
            elapsed = time.monotonic() - t0
            time.sleep(max(0.0, interval - elapsed))

    threading.Thread(target=control_loop, daemon=True).start()

    # ── Keyboard input (main thread) ──────────────────────────────────
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   Score the Story  —  Live                                 ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print("Commands:")
    print("  t <scene>   inject a scene, e.g.  t the hero charged into battle")
    print("  m           mute/unmute the mic (mute before typing injections)")
    print("  s           show current voice + MRT2 status")
    print("  c/a/d/f/g   hold a chord     r  release     q  quit")
    if telemetry:
        print(f"  live charts → http://localhost:{args.port}  (open in a browser)")
    print()

    try:
        while True:
            raw = input("scene> ").strip()
            cmd = raw.lower()

            if cmd == "q":
                break
            elif cmd == "m":
                if detector:
                    muted = detector.toggle_mute()
                    print(f"[mic] {'MUTED — type t <scene> to inject' if muted else 'LIVE — listening again'}")
                else:
                    print("m command only available in --mode python")
            elif cmd == "s":
                print_status(analyzer, mapper, detector, controller)
            elif cmd == "r":
                controller.release_chord()
                print("Chord released.")
            elif cmd[:1] == "t" and (len(cmd) == 1 or cmd[1] == " "):
                if detector:
                    scene = raw[1:].strip()
                    if scene:
                        detector.inject(scene)
                        print(f"[inject] → \"{scene[:70]}\"  (Claude is composing…)")
                    else:
                        print("Usage: t <scene>   e.g.: t the hero charged into battle")
                else:
                    print("t command only available in --mode python")
            elif cmd in CHORDS:
                notes = CHORDS[cmd]
                controller.hold_chord(notes)
                print(f"Holding chord: {notes}")
            elif cmd:
                print(f"Unknown command '{cmd}'. Try: t <scene>  m  s  c a d f g  r  q")

    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        print("\nShutting down...")
        stop_event.set()
        analyzer.stop()
        controller.stop()
        if detector:
            detector.stop()
        if telemetry:
            telemetry.stop()
        # Save the telling so it can be rendered into a keepsake song.
        if args.mode == "python":
            # The keepsake's identity = the BLEND of every voice from the moment.
            if signature.windows > 0:
                session_log.set_signature(signature.signature())
                print(f"\n🎙  Song identity (blended from {signature.windows} "
                      f"voiced moments): {signature.describe()}")
            path = session_log.save()
            if path:
                print(f"🎵 Session captured → {path}")
                try:
                    summary = build_summary(
                        duration_s=session_log.elapsed(),
                        scenes=session_log.scenes(),
                        perf=controller.perf_stats(),
                        health=controller.health(),
                    )
                    spath = write_summary(summary, paths.summary_for(path))
                    print(f"📊 Session report → {spath}")
                except Exception as e:
                    print(f"   (session report skipped: {e})")
                print(f"   Render the keepsake song with:")
                print(f"     python3 keepsake.py render {path}")
                print(f"   ...or turn it into an editable, multiplayer Audiotool project:")
                print(f"     python3 audiotool_arranger.py {path}")
                print(f"     (then: cd audiotool_export && npm install && see its README)")
        print("Done.")


if __name__ == "__main__":
    main()
