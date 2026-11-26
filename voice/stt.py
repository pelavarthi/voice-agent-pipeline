"""Speech-to-text adapters: interface, fake transcriber, streaming sessions.

A Transcriber turns audio into text. It works two ways: one-shot, for a
whole utterance at once, and streaming, for a flow of audio chunks that
yields partial results while the speaker is still talking and a final
result when the utterance ends. Real STT (Day 10) plugs in behind the
same interface; the fake transcriber reads from a script, so the whole
pipeline is testable with no API key and no microphone.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from voice.audio import (
    DEFAULT_SAMPLE_RATE,
    frame_duration_ms,
    validate_sample_rate,
)

DEFAULT_SCRIPT = (
    "hello world",
    "what time is it",
    "tell me a joke",
    "how is the weather today",
)
DEFAULT_MS_PER_WORD = 200  # how fast the fake stream reveals the next word


@dataclass
class Transcription:
    """One STT result: a partial guess or a final transcript."""

    text: str
    start_ms: float = 0.0
    end_ms: float = 0.0
    is_final: bool = True
    confidence: float = 1.0

    @property
    def duration_ms(self):
        return self.end_ms - self.start_ms


class TranscriptionStream(ABC):
    """One streaming session: audio chunks in, results out."""

    @abstractmethod
    def push(self, samples):
        """Feed one audio chunk; returns new results (partials or finals)."""

    @abstractmethod
    def finish(self):
        """Close the stream; returns any final results left."""


class Transcriber(ABC):
    """Turns audio into text, one-shot or streaming."""

    @abstractmethod
    def transcribe(self, samples, sample_rate=DEFAULT_SAMPLE_RATE):
        """Transcribe a whole utterance at once; returns a Transcription."""

    @abstractmethod
    def stream(self, sample_rate=DEFAULT_SAMPLE_RATE):
        """Start a streaming session; returns a TranscriptionStream."""


class FakeTranscriber(Transcriber):
    """A transcriber that reads its answers from a script.

    Each utterance — one-shot or one stream — takes the next phrase in
    the script, cycling back to the start when it runs out, so a demo
    never stalls for lack of lines. The streaming session reveals one
    word per DEFAULT_MS_PER_WORD of audio, the way a real streaming
    recognizer refines its guess as more speech arrives.
    """

    def __init__(self, script=DEFAULT_SCRIPT, ms_per_word=DEFAULT_MS_PER_WORD):
        if not script:
            raise ValueError("script must have at least one phrase")
        if ms_per_word <= 0:
            raise ValueError("ms_per_word must be positive")
        self.script = tuple(script)
        self.ms_per_word = ms_per_word
        self._next = 0

    def _take_phrase(self):
        phrase = self.script[self._next]
        self._next = (self._next + 1) % len(self.script)
        return phrase

    def transcribe(self, samples, sample_rate=DEFAULT_SAMPLE_RATE):
        validate_sample_rate(sample_rate)
        duration = frame_duration_ms(len(samples), sample_rate)
        if not samples:
            return Transcription(text="", end_ms=0.0)
        return Transcription(text=self._take_phrase(), end_ms=duration)

    def stream(self, sample_rate=DEFAULT_SAMPLE_RATE):
        validate_sample_rate(sample_rate)
        return _FakeStream(self._take_phrase(), sample_rate, self.ms_per_word)


class _FakeStream(TranscriptionStream):
    """Streaming session for one scripted phrase."""

    def __init__(self, phrase, sample_rate, ms_per_word):
        self._words = phrase.split()
        self._sample_rate = sample_rate
        self._ms_per_word = ms_per_word
        self._elapsed_ms = 0.0
        self._revealed = 0
        self._finished = False

    def _partial(self):
        return Transcription(
            text=" ".join(self._words[:self._revealed]),
            end_ms=self._elapsed_ms,
            is_final=False,
        )

    def push(self, samples):
        if self._finished or not samples:
            return []
        self._elapsed_ms += frame_duration_ms(len(samples), self._sample_rate)
        revealed = min(
            len(self._words), int(self._elapsed_ms // self._ms_per_word)
        )
        if revealed == self._revealed:
            return []
        self._revealed = revealed
        return [self._partial()]

    def finish(self):
        if self._finished:
            return []
        self._finished = True
        if self._elapsed_ms == 0:
            return []
        return [
            Transcription(
                text=" ".join(self._words),
                end_ms=self._elapsed_ms,
                is_final=True,
            )
        ]


def create_transcriber(kind="fake", **kwargs):
    """Build a transcriber by name; Day 10 adds real ones behind keys."""
    if kind == "fake":
        return FakeTranscriber(**kwargs)
    raise ValueError(f"unknown transcriber: {kind}")


def transcribe_segment(segment, transcriber,
                       sample_rate=DEFAULT_SAMPLE_RATE):
    """Transcribe one VAD segment; timestamps are shifted onto the segment."""
    result = transcriber.transcribe(segment.samples, sample_rate)
    return Transcription(
        text=result.text,
        start_ms=segment.start_ms + result.start_ms,
        end_ms=segment.start_ms + result.end_ms,
        is_final=result.is_final,
        confidence=result.confidence,
    )


def transcribe_frames(frames, transcriber, sample_rate=DEFAULT_SAMPLE_RATE):
    """Drive a stream over audio frames; returns partials then the final."""
    stream = transcriber.stream(sample_rate)
    results = []
    for frame in frames:
        results.extend(stream.push(frame))
    results.extend(stream.finish())
    return results
