"""Energy-based voice activity detection.

Listens to fixed-size audio frames and decides when someone is speaking.
A frame counts as speech when its RMS energy clears a threshold; a few
loud frames in a row start a segment, and a stretch of quiet frames ends
it. The detector keeps a short pre-roll of audio before speech starts,
so the first syllable is not clipped off.

Later chunks feed the segments into the STT adapter, and the live
speech/silence signal drives barge-in and turn-taking.
"""

import math
from collections import deque
from dataclasses import dataclass

from voice.audio import (
    DEFAULT_FRAME_MS,
    DEFAULT_SAMPLE_RATE,
    frame_duration_ms,
    samples_for_ms,
    validate_sample_rate,
)

DEFAULT_THRESHOLD = 400.0  # RMS energy; a normal voice at the mic is ~1000+
DEFAULT_MIN_SPEECH_MS = 100  # ignore blips shorter than this
DEFAULT_SILENCE_TIMEOUT_MS = 400  # quiet this long ends a segment
DEFAULT_PRE_ROLL_MS = 100  # audio kept from just before speech starts


@dataclass
class SpeechSegment:
    """One stretch of detected speech, with its audio attached."""

    start_ms: float
    end_ms: float
    samples: list

    @property
    def duration_ms(self):
        return self.end_ms - self.start_ms


@dataclass
class VadEvent:
    """Something the detector noticed: speech started or a segment ended."""

    kind: str  # "start" or "end"
    start_ms: float
    end_ms: float = None
    segment: SpeechSegment = None


def frame_rms(samples):
    """Root-mean-square energy of a frame; 0 for silence."""
    if not samples:
        return 0.0
    return math.sqrt(sum(s * s for s in samples) / len(samples))


class VoiceActivityDetector:
    """State machine over audio frames: idle -> speaking -> idle."""

    def __init__(self, sample_rate=DEFAULT_SAMPLE_RATE, frame_ms=DEFAULT_FRAME_MS,
                 threshold=DEFAULT_THRESHOLD,
                 min_speech_ms=DEFAULT_MIN_SPEECH_MS,
                 silence_timeout_ms=DEFAULT_SILENCE_TIMEOUT_MS,
                 pre_roll_ms=DEFAULT_PRE_ROLL_MS):
        validate_sample_rate(sample_rate)
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.threshold = threshold
        self._frame_samples = samples_for_ms(frame_ms, sample_rate)
        self._min_speech_frames = max(1, round(min_speech_ms / frame_ms))
        self._silence_frames = max(1, round(silence_timeout_ms / frame_ms))
        self._pre_roll_ms = pre_roll_ms
        # Enough history to cover the pre-roll plus the frames needed to
        # confirm speech, so the segment starts before the first loud frame.
        history = max(0, round(pre_roll_ms / frame_ms)) + self._min_speech_frames
        self._pre_roll = deque(maxlen=history)
        self._elapsed_ms = 0.0
        self._reset()

    def _reset(self):
        self._state = "idle"
        self._speech_run = 0
        self._silence_run = 0
        self._run_start_ms = 0.0
        self._segment_start_ms = 0.0
        self._segment_samples = []

    @property
    def speaking(self):
        return self._state == "speaking"

    def calibrate(self, frames):
        """Set the threshold from frames of background noise.

        The threshold becomes four times the average frame energy, with
        a floor of 200 RMS so a dead-quiet room does not make the
        detector hair-trigger.
        """
        if not frames:
            return
        floor = sum(frame_rms(f) for f in frames) / len(frames)
        self.threshold = max(200.0, 4 * floor)

    def process(self, frame):
        """Feed one frame; returns a VadEvent or None."""
        frame = list(frame)
        is_speech = frame_rms(frame) >= self.threshold
        start_ms = self._elapsed_ms
        self._elapsed_ms += frame_duration_ms(len(frame), self.sample_rate)
        end_ms = self._elapsed_ms

        if self._state == "idle":
            self._pre_roll.append(frame)
            if is_speech:
                self._speech_run += 1
                if self._speech_run == 1:
                    self._run_start_ms = start_ms
                if self._speech_run >= self._min_speech_frames:
                    self._state = "speaking"
                    self._silence_run = 0
                    self._segment_start_ms = max(
                        0.0, self._run_start_ms - self._pre_roll_ms
                    )
                    self._segment_samples = [
                        s for f in self._pre_roll for s in f
                    ]
                    return VadEvent(kind="start",
                                    start_ms=self._segment_start_ms)
            else:
                self._speech_run = 0
            return None

        self._segment_samples.extend(frame)
        if is_speech:
            self._silence_run = 0
        else:
            self._silence_run += 1
            if self._silence_run >= self._silence_frames:
                # The segment ends at the last loud frame; the trailing
                # quiet frames were only the timeout, not speech.
                end = end_ms - self._silence_run * self.frame_ms
                trim = self._silence_run * self._frame_samples
                samples = self._segment_samples[:-trim] if trim else \
                    self._segment_samples
                segment = SpeechSegment(self._segment_start_ms, end, samples)
                self._reset()
                return VadEvent(kind="end", start_ms=segment.start_ms,
                                end_ms=segment.end_ms, segment=segment)
        return None

    def flush(self):
        """Close an open segment at the end of the stream, if there is one."""
        if self._state != "speaking":
            self._speech_run = 0
            return None
        segment = SpeechSegment(
            self._segment_start_ms, self._elapsed_ms, self._segment_samples
        )
        self._reset()
        return VadEvent(kind="end", start_ms=segment.start_ms,
                        end_ms=segment.end_ms, segment=segment)

    def segments(self, frames):
        """Run frames through the detector; returns the segments found."""
        found = []
        for frame in frames:
            event = self.process(frame)
            if event is not None and event.kind == "end":
                found.append(event.segment)
        tail = self.flush()
        if tail is not None:
            found.append(tail.segment)
        return found


def detect_speech(samples, sample_rate=DEFAULT_SAMPLE_RATE,
                  frame_ms=DEFAULT_FRAME_MS, **kwargs):
    """Split raw samples into frames and return the speech segments."""
    detector = VoiceActivityDetector(sample_rate, frame_ms, **kwargs)
    frame_len = samples_for_ms(frame_ms, sample_rate)
    frames = [samples[i:i + frame_len]
              for i in range(0, len(samples), frame_len)]
    return detector.segments(frames)
