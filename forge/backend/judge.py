"""
judge.py - the AI judge (a tutor, not just a scorekeeper).

Scores a player's free-text guess against the TRUE prompt that generated the clip
they heard, and returns a score PLUS a short explanation. The explanation is the
point: a wrong guess should teach.

Two backends:
  - Claude (claude-haiku) when ANTHROPIC_API_KEY is set - reuses the cached
    system-prompt pattern from the livescore project's llm_style_director.py.
  - A word-overlap heuristic fallback when there's no key (so the game always runs).

Difficulty (beginner|advanced) swaps the RUBRIC, not the code.

STATUS: Phase 2 - score() implemented. Embedding/novelty still Phase 3.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .models import Skill

_RUBRIC = {
    "beginner": "genre, overall mood (happy/sad), and tempo (fast/slow)",
    "advanced": ("subgenre, era, instrumentation incl. articulation, production "
                 "technique, rhythmic feel/meter, and harmonic/modal flavor"),
}

_SYSTEM = """\
You judge how well a player's GUESS describes the same music as a TRUE prompt for
a generative music model. Both are short style descriptions (instruments, mood,
genre, etc.). Score how close the GUESS is to the TRUE prompt's *sound*, focusing
on: {rubric}.

Be encouraging but honest. Reward specificity; a vague guess that happens to
overlap should not score high. The tutor_note is ONE short sentence that teaches
- name what they got right and the single most important thing they missed.

Output ONLY JSON: {{"score": <0-100 int>, "tutor_note": "<one sentence>"}}"""

_BRIEF_SYSTEM = """\
You judge a PROMPT a player wrote for a generative music model, trying to evoke a
SCENE (a brief). Score 0-100 how well this prompt would produce INSTRUMENTAL music
that fits the brief's mood, energy, and setting.

Reward concrete musical translation: instruments, genre, tempo/rhythm, texture,
and harmonic or mood-in-sound choices that suit the scene. PENALIZE a prompt that
merely restates the brief or swaps in SYNONYMS of the brief's words instead of
describing actual sound; renaming the scene is not prompting, score it low.

The tutor_note is ONE short sentence that teaches: name what worked and the single
best improvement to their prompting.

Output ONLY JSON: {"score": <0-100 int>, "tutor_note": "<one sentence>"}"""


@dataclass(frozen=True)
class Verdict:
    score: float                       # 0..100
    tutor_note: str                    # the teaching/flavor line shown to the player
    breakdown: dict = field(default_factory=dict)


class Judge:
    def __init__(self, api_key: str | None = None):
        self._api_key = api_key
        self._client = None            # lazy anthropic.Anthropic

    def score(self, guess: str, truth: str, skill: Skill = "advanced") -> Verdict:
        """Judge a guess against the true prompt under the skill-appropriate rubric."""
        guess = (guess or "").strip()
        if not guess:
            return Verdict(0.0, "No guess submitted.")
        if self._api_key:
            try:
                return self._claude(guess, truth, skill)
            except Exception as e:  # noqa: BLE001 - never let scoring crash a round
                print(f"[judge] Claude failed ({e}); using heuristic")
        return self._heuristic(guess, truth)

    # ── Claude backend ─────────────────────────────────────────────────────────
    def _get_client(self):
        if self._client is None:
            import anthropic  # noqa: PLC0415
            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def _claude(self, guess: str, truth: str, skill: Skill) -> Verdict:
        client = self._get_client()
        system = _SYSTEM.format(rubric=_RUBRIC.get(skill, _RUBRIC["advanced"]))
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=150,
            system=[{"type": "text", "text": system,
                     "cache_control": {"type": "ephemeral"}}],
            messages=[
                {"role": "user",
                 "content": f"TRUE prompt: {truth}\nPLAYER'S guess: {guess}"},
                {"role": "assistant", "content": '{"score":'},
            ],
        )
        raw = '{"score":' + resp.content[0].text
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        data = json.loads(match.group(0) if match else raw)
        score = max(0.0, min(100.0, float(data["score"])))
        return Verdict(score, str(data.get("tutor_note", "")).strip())

    # ── Prompt Party: judge a music PROMPT against a SCENE brief ────────────────
    def score_brief(self, prompt: str, brief: str) -> Verdict:
        """Score how well a player's music-generation PROMPT would evoke a SCENE.
        Rewards translating the scene into SOUND (instruments, genre, tempo, mood),
        and penalizes merely restating the brief or using synonyms of its words."""
        prompt = (prompt or "").strip()
        if not prompt:
            return Verdict(0.0, "No prompt submitted.")
        if self._api_key:
            try:
                client = self._get_client()
                resp = client.messages.create(
                    model="claude-haiku-4-5-20251001", max_tokens=150,
                    system=[{"type": "text", "text": _BRIEF_SYSTEM,
                             "cache_control": {"type": "ephemeral"}}],
                    messages=[
                        {"role": "user",
                         "content": f"BRIEF (the scene to evoke in music): {brief}\nPLAYER'S prompt: {prompt}"},
                        {"role": "assistant", "content": '{"score":'},
                    ],
                )
                raw = '{"score":' + resp.content[0].text
                match = re.search(r"\{.*\}", raw, re.DOTALL)
                data = json.loads(match.group(0) if match else raw)
                return Verdict(max(0.0, min(100.0, float(data["score"]))),
                               str(data.get("tutor_note", "")).strip())
            except Exception as e:  # noqa: BLE001 - never crash a round
                print(f"[judge] brief scoring failed ({e}); using heuristic")
        # fallback (no key): reward specificity (distinct descriptive words)
        n = len(self._words(prompt))
        return Verdict(float(min(100, 25 + n * 9)),
                       "Name instruments, a genre, and a tempo to paint the scene in sound.")

    # ── Heuristic fallback (no API key) ─────────────────────────────────────────
    @staticmethod
    def _words(s: str) -> set[str]:
        stop = {"the", "a", "an", "and", "with", "of", "in", "to", "very", "some"}
        return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in stop}

    def _heuristic(self, guess: str, truth: str) -> Verdict:
        g, t = self._words(guess), self._words(truth)
        if not t:
            return Verdict(0.0, "Nothing to compare against.")
        hit = g & t
        jac = len(hit) / len(g | t) if (g | t) else 0.0
        score = round(100 * jac)
        if hit:
            note = f"Matched: {', '.join(sorted(hit))}. Listen for what you missed."
        else:
            note = "No shared words - try naming the instruments and the mood."
        return Verdict(float(score), note, {"overlap": score})

    # ── Novelty (Phase 3 - Forge Battle) ────────────────────────────────────────
    @staticmethod
    def novelty(embedding, references) -> float:
        """How original a submission is: 1 − cosine similarity to its NEAREST other
        submission (0 = identical to someone, 1 = maximally different). Uses the
        MusicCoCa style embeddings attached to each clip."""
        import numpy as np  # noqa: PLC0415
        refs = [r for r in (references or []) if r]
        if not embedding or not refs:
            return 0.0
        v = np.asarray(embedding, dtype=float)
        vn = np.linalg.norm(v) or 1.0
        best = -1.0
        for r in refs:
            r = np.asarray(r, dtype=float)
            rn = np.linalg.norm(r) or 1.0
            best = max(best, float(np.dot(v, r) / (vn * rn)))
        return max(0.0, 1.0 - best)
