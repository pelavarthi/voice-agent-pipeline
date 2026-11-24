"""A fake microphone source for tests and demos.

Plays back recorded or generated samples in fixed-size frames, exactly
like a real mic would, but with no hardware and no timing surprises.
Later chunks feed it into the VAD and the STT adapter the same way they
will feed a real capture device.
"""

from voice.audio import DEFAULT_SAMPLE_RATE, sine_wave, validate_sample_rate
from voice.wav_io import read_wav


class FakeMicSource:
    """Yields PCM16 frames from an in-memory sample list."""

    def __init__(self, samples, sample_rate=DEFAULT_SAMPLE_RATE, loop=False):
        validate_sample_rate(sample_rate)
        self.samples = list(samples)
        self.sample_rate = sample_rate
        self.loop = loop
        self._pos = 0

    @classmethod
    def from_wav(cls, path, loop=False):
        samples, sample_rate = read_wav(path)
        return cls(samples, sample_rate, loop=loop)

    @classmethod
    def from_tone(cls, freq_hz=440, duration_ms=1000,
                  sample_rate=DEFAULT_SAMPLE_RATE, loop=False):
        return cls(sine_wave(freq_hz, duration_ms, sample_rate),
                   sample_rate, loop=loop)

    @classmethod
    def silence(cls, duration_ms=1000, sample_rate=DEFAULT_SAMPLE_RATE):
        from voice.audio import samples_for_ms

        return cls([0] * samples_for_ms(duration_ms, sample_rate),
                   sample_rate)

    @property
    def exhausted(self):
        return not self.loop and self._pos >= len(self.samples)

    def read(self, num_samples):
        """Next num_samples, padding with silence at the end (or looping)."""
        out = []
        while len(out) < num_samples:
            if self._pos >= len(self.samples):
                if self.loop and self.samples:
                    self._pos = 0
                else:
                    out.extend([0] * (num_samples - len(out)))
                    break
            take = min(num_samples - len(out), len(self.samples) - self._pos)
            out.extend(self.samples[self._pos:self._pos + take])
            self._pos += take
        return out

    def chunks(self, frame_ms=20):
        """Yield fixed-size frames until the source is exhausted."""
        from voice.audio import samples_for_ms

        frame = samples_for_ms(frame_ms, self.sample_rate)
        while not self.exhausted:
            yield self.read(frame)
