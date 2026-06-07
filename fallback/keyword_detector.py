"""
keyword_detector.py
-------------------
Transcribes the narrator's speech with Whisper tiny every few seconds,
detects story-significant keywords, and switches MRT2's musical style poles.

Two layers of control:
  - Acoustic features (energy, brightness) → continuous micro-variation via blend
  - Keywords → macro-level style switch (changes what the two poles ARE)

When a keyword fires, the active prompts change. The feature mapper's blend
continues to work between the new poles — so the music shifts territory AND
stays reactive to the narrator's voice within that new territory.

Edit KEYWORD_MAP and STYLE_PRESETS freely to match your story.
"""

import threading
import time
import numpy as np

SAMPLE_RATE = 48_000   # matches VoiceAnalyzer


# ── Musical style presets ────────────────────────────────────────────────────
# Each preset is (prompt_A, prompt_B).
# Prompt A = the "darker/lower" pole; Prompt B = the "brighter/higher" pole.
# The narrator's voice continuously blends between the two within the preset.

STYLE_PRESETS = {
    # default: ambient neutral starting point
    'default': (
        "dark ambient electronic drone, minimal, atmospheric",
        "warm lo-fi hip hop beat, chill, gentle piano",
    ),

    # action / superhero / fight scenes
    'action': (
        "epic superhero action music, massive orchestra, Hans Zimmer, pounding drums, intense",
        "driving electronic rock, distorted electric guitar, heavy synth bass, energetic",
    ),

    # battle / war
    'battle': (
        "intense war drums, low brass, heavy percussion, aggressive orchestral action",
        "powerful metal electric guitar riff, distorted bass, hard rock battle music",
    ),

    # pursuit / chase
    'chase': (
        "fast paced chase music, staccato strings, urgent brass, Hans Zimmer Inception style",
        "electronic techno chase, fast drum machine, synth pulse, high tension",
    ),

    # grief / death / tragedy
    'grief': (
        "sorrowful solo cello, slow minor key, silence and grief",
        "sad indie folk acoustic guitar, minimal, heartfelt, emotional",
    ),

    # hope / dawn / new beginning
    'hope': (
        "gentle ambient awakening, soft synthesizer pads, peaceful sunrise",
        "uplifting indie pop, acoustic guitar, bright piano, hopeful major key",
    ),

    # fear / horror / dread
    'fear': (
        "horror film score, dissonant strings, creeping dread, dark ambient",
        "industrial noise, eerie synth, unsettling electronic, suspense",
    ),

    # love / romance / tender
    'love': (
        "intimate acoustic guitar solo, warm and tender, fingerpicked",
        "romantic jazz piano trio, soft brushed drums, dreamy, bossa nova",
    ),

    # mystery / detective / unknown
    'mystery': (
        "noir jazz, muted trumpet, sparse piano, smoky and mysterious",
        "ambient electronic mystery, subtle synth textures, curious, searching",
    ),

    # triumph / victory / celebration
    'triumph': (
        "triumphant superhero fanfare, full orchestra, soaring brass, epic victory",
        "euphoric electronic dance music, uplifting synth lead, festival crowd energy",
    ),

    # sadness / melancholy
    'sad': (
        "melancholy piano solo, slow, introspective, minor key",
        "sad singer-songwriter acoustic, emotional, quiet strings",
    ),

    # danger / threat / villain
    'danger': (
        "villain theme, dark orchestral, low brass stabs, ominous",
        "tense electronic thriller, pulsing synth bass, cat and mouse suspense",
    ),

    # magic / fantasy / wonder
    'magic': (
        "magical fantasy orchestra, shimmering strings, harp glissando, wonder",
        "ethereal ambient electronic, floating synth pads, mystical and otherworldly",
    ),

    # space / sci-fi / futuristic
    'space': (
        "sci-fi ambient synthesizer, space exploration, Interstellar Hans Zimmer",
        "futuristic electronic, glitchy synth, cyberpunk, Blade Runner atmosphere",
    ),
}


# ── Keyword → preset mapping ─────────────────────────────────────────────────
# Keys are single lowercase words. Add/remove to match your story.

KEYWORD_MAP = {
    # Action / superhero
    'superhero': 'action', 'hero': 'action', 'powers': 'action', 'cape': 'action',
    'punched': 'action', 'punch': 'action', 'kicked': 'action', 'slammed': 'action',
    'flying': 'action', 'flew': 'action', 'laser': 'action', 'exploded': 'action',
    'explosion': 'action', 'crash': 'action', 'smash': 'action', 'epic': 'action',

    # Battle / war
    'battle': 'battle', 'fight': 'battle', 'fighting': 'battle', 'war': 'battle',
    'fought': 'battle', 'army': 'battle', 'warrior': 'battle', 'attack': 'battle',
    'attacked': 'battle', 'clash': 'battle', 'enemy': 'battle', 'soldiers': 'battle',
    'weapon': 'battle', 'sword': 'battle', 'gun': 'battle', 'shot': 'battle',

    # Chase / pursuit
    'running': 'chase', 'ran': 'chase', 'chasing': 'chase', 'chase': 'chase',
    'escape': 'chase', 'escaped': 'chase', 'fleeing': 'chase', 'fled': 'chase',
    'pursuit': 'chase', 'caught': 'chase', 'sprinting': 'chase', 'rushed': 'chase',

    # Danger / villain / threat
    'villain': 'danger', 'evil': 'danger', 'threat': 'danger', 'danger': 'danger',
    'dark': 'danger', 'dangerous': 'danger', 'trap': 'danger', 'ambush': 'danger',
    'lurking': 'danger', 'shadow': 'danger', 'menacing': 'danger',

    # Grief / death
    'death': 'grief', 'died': 'grief', 'dead': 'grief', 'killed': 'grief',
    'fallen': 'grief', 'grief': 'grief', 'sorrow': 'grief', 'mourning': 'grief',
    'loss': 'grief', 'tragedy': 'grief', 'heartbroken': 'grief', 'crying': 'grief',

    # Sad / melancholy
    'sad': 'sad', 'alone': 'sad', 'lonely': 'sad', 'broken': 'sad',
    'tears': 'sad', 'wept': 'sad', 'hurt': 'sad', 'pain': 'sad',

    # Hope / dawn
    'hope': 'hope', 'light': 'hope', 'dawn': 'hope', 'morning': 'hope',
    'sunrise': 'hope', 'new': 'hope', 'rise': 'hope', 'beginning': 'hope',
    'awakened': 'hope', 'reborn': 'hope', 'bright': 'hope',

    # Fear / horror
    'fear': 'fear', 'afraid': 'fear', 'terrified': 'fear', 'horror': 'fear',
    'scared': 'fear', 'monster': 'fear', 'creature': 'fear', 'haunted': 'fear',
    'nightmare': 'fear', 'dread': 'fear', 'scream': 'fear', 'screamed': 'fear',

    # Love / romance
    'love': 'love', 'loved': 'love', 'kiss': 'love', 'kissed': 'love',
    'embrace': 'love', 'together': 'love', 'heart': 'love', 'romantic': 'love',
    'tender': 'love', 'beautiful': 'love', 'gentle': 'love',

    # Mystery / detective
    'mystery': 'mystery', 'secret': 'mystery', 'hidden': 'mystery',
    'unknown': 'mystery', 'clue': 'mystery', 'discovered': 'mystery',
    'whispered': 'mystery', 'strange': 'mystery', 'ancient': 'mystery',
    'forbidden': 'mystery', 'encrypted': 'mystery',

    # Triumph / victory
    'victory': 'triumph', 'triumph': 'triumph', 'won': 'triumph', 'win': 'triumph',
    'saved': 'triumph', 'prevailed': 'triumph', 'celebrated': 'triumph',
    'glory': 'triumph', 'champion': 'triumph', 'achieved': 'triumph',

    # Magic / fantasy
    'magic': 'magic', 'spell': 'magic', 'wizard': 'magic', 'dragon': 'magic',
    'enchanted': 'magic', 'mystical': 'magic', 'portal': 'magic', 'realm': 'magic',
    'sorcerer': 'magic', 'powers': 'magic', 'potion': 'magic',

    # Space / sci-fi
    'space': 'space', 'galaxy': 'space', 'planet': 'space', 'spaceship': 'space',
    'robot': 'space', 'alien': 'space', 'future': 'space', 'cyberpunk': 'space',
    'starship': 'space', 'cosmos': 'space', 'orbit': 'space',
}


class KeywordDetector:
    """
    Runs Whisper tiny in a background thread. Listens every TRANSCRIBE_INTERVAL
    seconds, checks the transcript for keywords, and calls controller.set_prompts()
    when a new story mood is detected.
    """

    TRANSCRIBE_INTERVAL = 3.5   # seconds between transcription passes
    AUDIO_WINDOW        = 4.0   # seconds of audio fed to Whisper each pass

    def __init__(self, analyzer, controller):
        self._analyzer   = analyzer
        self._controller = controller
        self._stop_event = threading.Event()
        self._current_preset = 'default'

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def stop(self):
        self._stop_event.set()

    # ── Internal ─────────────────────────────────────────────────────────────

    def _run(self):
        import whisper
        import librosa

        print("[Keyword] Loading Whisper tiny...")
        model = whisper.load_model('tiny')
        print("[Keyword] Ready — listening for story keywords")
        print(f"[Keyword] Active preset: {self._current_preset}")

        while not self._stop_event.is_set():
            time.sleep(self.TRANSCRIBE_INTERVAL)

            audio_48k = self._analyzer.get_audio_for_transcription(self.AUDIO_WINDOW)
            if len(audio_48k) < SAMPLE_RATE // 2:
                continue  # not enough audio yet

            # Whisper expects 16kHz mono float32
            audio_16k = librosa.resample(audio_48k, orig_sr=SAMPLE_RATE, target_sr=16_000)

            try:
                result = model.transcribe(audio_16k, fp16=False, language='en')
                text = result['text'].lower().strip()
                if text:
                    print(f"[Keyword] \"{text[:80]}\"")
                    self._check_keywords(text)
            except Exception as e:
                print(f"[Keyword] transcription error: {e}")

    def _check_keywords(self, text: str):
        words = text.replace(',', ' ').replace('.', ' ').replace("'", ' ').split()
        for word in words:
            word = word.strip('!"?;:')
            if word in KEYWORD_MAP:
                preset = KEYWORD_MAP[word]
                if preset != self._current_preset:
                    self._current_preset = preset
                    prompt_a, prompt_b = STYLE_PRESETS[preset]
                    print(f"[Keyword] '{word}' → '{preset}' style")
                    self._controller.set_prompts(prompt_a, prompt_b)
                    return   # one switch per transcription window
