"""
trainer.py - Ear Trainer (Phase 4), expanded.

Two kinds of question, each with the right tool:
  - THEORY (intervals, chords, scales) → precise SYNTHESIZED tones (synth.py).
    Randomized root every time, so questions never repeat and clips are instant.
  - VIBE (mood / tempo / energy / instrument / genre / texture) → MRT2 clips
    (longer, ~6s), the perceptual layer MRT2 is actually good at.

Each question is generated on the fly; its answer is stashed under a token and
checked on submit. Feedback names the concept - and for intervals, leans on
familiar songs ("a Perfect 5th - like the start of 'Twinkle, Twinkle'").
"""

from __future__ import annotations

import os
import random
import uuid

from . import synth
from .forge_core import ForgeCore
from .models import PromptSpec
from .storage import Storage

# ── Theory data ──────────────────────────────────────────────────────────────
INTERVAL_NAMES = {1: "Minor 2nd", 2: "Major 2nd", 3: "Minor 3rd", 4: "Major 3rd",
                  5: "Perfect 4th", 6: "Tritone", 7: "Perfect 5th", 8: "Minor 6th",
                  9: "Major 6th", 10: "Minor 7th", 11: "Major 7th", 12: "Octave"}
# Song hooks for the common ones (ascending).
INTERVAL_SONGS = {
    2: "the first two notes of 'Happy Birthday'",
    3: "the opening of 'Greensleeves'",
    4: "the start of 'When the Saints Go Marching In'",
    5: "'Here Comes the Bride'",
    7: "the start of 'Twinkle, Twinkle, Little Star'",
    9: "the 'My Bonnie' melody",
    12: "'Some-where' in 'Over the Rainbow'",
}
CHORDS_EASY = [("Major", [0, 4, 7]), ("Minor", [0, 3, 7])]
CHORDS_ALL = CHORDS_EASY + [("Diminished", [0, 3, 6]), ("Augmented", [0, 4, 8])]
CHORD_WHY = {
    "Major": "a Major chord - bright and stable (a major 3rd on the bottom).",
    "Minor": "a Minor chord - darker, sadder (a minor 3rd on the bottom).",
    "Diminished": "a Diminished chord - tense and unstable (both 3rds minor).",
    "Augmented": "an Augmented chord - dreamy and unsettled (both 3rds major).",
}
MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11, 12]
MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10, 12]

VIBE_DECK = [
    {"prompt": "happy bright major-key piano, cheerful", "dimension": "Mood",
     "question": "Does this feel happy or sad?", "options": ["Happy", "Sad"], "answer": 0,
     "why": "That bright, lifted feeling is a MAJOR key."},
    {"prompt": "sad slow minor-key piano, melancholy", "dimension": "Mood",
     "question": "Does this feel happy or sad?", "options": ["Happy", "Sad"], "answer": 1,
     "why": "That darker, heavier feeling is a MINOR key."},
    {"prompt": "fast upbeat energetic electronic beat, driving", "dimension": "Tempo",
     "question": "Fast or slow?", "options": ["Fast", "Slow"], "answer": 0,
     "why": "A quick pulse = fast TEMPO."},
    {"prompt": "slow calm spacious ambient pad, drifting", "dimension": "Tempo",
     "question": "Fast or slow?", "options": ["Fast", "Slow"], "answer": 1,
     "why": "An unhurried pulse = slow TEMPO."},
    {"prompt": "loud intense powerful orchestral hit, dramatic", "dimension": "Energy",
     "question": "Gentle or intense?", "options": ["Gentle", "Intense"], "answer": 1,
     "why": "Loud and forceful = high-energy DYNAMICS."},
    {"prompt": "solo acoustic grand piano, clear", "dimension": "Instrument",
     "question": "Which instrument leads?", "options": ["Piano", "Guitar", "Drums", "Synth"],
     "answer": 0, "why": "That struck-string, resonant tone is a PIANO."},
    {"prompt": "fingerpicked nylon acoustic guitar, warm", "dimension": "Instrument",
     "question": "Which instrument leads?", "options": ["Piano", "Guitar", "Flute", "Bass"],
     "answer": 1, "why": "That plucked, woody warmth is a GUITAR."},
    {"prompt": "lo-fi hip hop beat, mellow, vinyl crackle", "dimension": "Genre",
     "question": "Which genre?", "options": ["Lo-fi hip hop", "Heavy metal", "Classical", "EDM"],
     "answer": 0, "why": "Mellow beat + dusty texture = LO-FI HIP HOP."},
    {"prompt": "epic cinematic orchestra, soaring strings and brass", "dimension": "Genre",
     "question": "Which genre?", "options": ["Orchestral", "Reggae", "Techno", "Lo-fi"],
     "answer": 0, "why": "Big strings and brass = ORCHESTRAL / cinematic."},
]

ROOT_LO, ROOT_HI = 52, 63   # comfortable root range (E3–Eb4)


def _shuffle(options: list[str], answer_text: str) -> tuple[list[str], int]:
    opts = options[:]
    random.shuffle(opts)
    return opts, opts.index(answer_text)


def gen_interval(semis: list[int]) -> dict:
    root = random.randint(ROOT_LO, ROOT_HI)
    correct = random.choice(semis)
    midis = [root, root + correct]
    audio = synth.join(synth.melodic(midis, 0.75, 0.06), synth.harmonic(midis, 1.5))
    options = [INTERVAL_NAMES[s] for s in sorted(semis)]   # sorted = learnable order
    name = INTERVAL_NAMES[correct]
    song = INTERVAL_SONGS.get(correct)
    article = "an" if name[0] in "AEIOU" else "a"
    why = f"That's {article} {name}" + (f" - like {song}." if song else ".")
    return {"audio": audio, "question": "Which interval do you hear?", "options": options,
            "answer": options.index(name), "answer_text": name, "why": why, "dimension": "Interval"}


def gen_chord(qualities: list[tuple[str, list[int]]]) -> dict:
    root = random.randint(ROOT_LO, ROOT_HI - 2)
    name, shape = random.choice(qualities)
    midis = [root + i for i in shape]
    audio = synth.join(synth.melodic(midis, 0.4, 0.04), synth.harmonic(midis, 1.7))
    options, answer = _shuffle([q[0] for q in qualities], name)
    return {"audio": audio, "question": "What kind of chord is this?", "options": options,
            "answer": answer, "answer_text": name, "why": "That's " + CHORD_WHY[name],
            "dimension": "Chord"}


def gen_scale() -> dict:
    root = random.randint(ROOT_LO, ROOT_HI - 2)
    name, shape = random.choice([("Major", MAJOR_SCALE), ("Minor", MINOR_SCALE)])
    audio = synth.melodic([root + i for i in shape], 0.38, 0.02)
    options, answer = _shuffle(["Major", "Minor"], name)
    why = ("a MAJOR scale - bright and happy all the way up."
           if name == "Major" else "a MINOR scale - note the darker, sadder 3rd step.")
    return {"audio": audio, "question": "Major or minor scale?", "options": options,
            "answer": answer, "answer_text": name, "why": "That's " + why, "dimension": "Scale"}


_CATS = {
    "interval5oct": lambda: gen_interval([7, 12]),
    "intervaleasy": lambda: gen_interval([4, 5, 7, 12]),
    "intervalall": lambda: gen_interval(list(range(1, 13))),
    "chord": lambda: gen_chord(CHORDS_EASY),
    "chordall": lambda: gen_chord(CHORDS_ALL),
    "scale": gen_scale,
}
_MIXED = ["interval5oct", "chord", "scale"]


class Trainer:
    def __init__(self, forge: ForgeCore, storage: Storage, audio_dir: str = "outputs/train"):
        self._forge = forge
        self._storage = storage
        self._dir = audio_dir
        os.makedirs(audio_dir, exist_ok=True)
        self._q: dict[str, dict] = {}      # token -> {answer, answer_text, why}
        self._paths: dict[str, str] = {}   # token -> synth wav path

    def question(self, category: str = "mixed") -> dict:
        cat = category if (category in _CATS or category == "vibe") else "mixed"
        if cat == "mixed":
            cat = random.choice(_MIXED)
        token = uuid.uuid4().hex[:12]
        if cat == "vibe":
            item = random.choice(VIBE_DECK)
            clip = self._forge.generate(PromptSpec(text_a=item["prompt"], chunks=8),
                                        created_by="trainer")
            self._storage.put_clip(clip)
            spec = {**item, "answer_text": item["options"][item["answer"]]}
            clip_url = f"/clips/{clip.id}.wav"
        else:
            spec = _CATS[cat]()
            path = os.path.join(self._dir, f"{token}.wav")
            synth.write_wav(path, spec["audio"])
            self._paths[token] = path
            clip_url = f"/train/audio/{token}.wav"
        self._q[token] = {"answer": spec["answer"], "answer_text": spec["answer_text"],
                          "why": spec["why"]}
        return {"token": token, "clip_url": clip_url, "question": spec["question"],
                "options": spec["options"], "dimension": spec["dimension"], "category": cat}

    def answer(self, token: str, choice: int) -> dict:
        q = self._q.get(token)
        if q is None:
            raise ValueError("unknown or expired question")
        return {"correct": choice == q["answer"], "answer": q["answer"],
                "answer_text": q["answer_text"], "why": q["why"]}

    def audio_path(self, token: str) -> str | None:
        return self._paths.get(token)
