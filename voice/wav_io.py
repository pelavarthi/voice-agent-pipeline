"""WAV file I/O for 16-bit PCM mono audio.

The stdlib `wave` module does the work; this is a thin wrapper that pins
the format the pipeline uses everywhere and keeps the call sites short.
"""

import wave

from voice.audio import bytes_to_samples, samples_to_bytes, validate_sample_rate


def write_wav(path, samples, sample_rate=16000):
    """Write mono PCM16 samples to a WAV file."""
    validate_sample_rate(sample_rate)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(samples_to_bytes(samples))


def read_wav(path):
    """Read a WAV file. Returns (samples, sample_rate).

    Only 16-bit mono files are supported; anything else raises ValueError
    so a stereo or 24-bit file fails loudly instead of being misread.
    """
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1:
            raise ValueError(f"expected mono WAV, got {wav.getnchannels()} channels")
        if wav.getsampwidth() != 2:
            raise ValueError(f"expected 16-bit WAV, got {wav.getsampwidth() * 8}-bit")
        return bytes_to_samples(wav.readframes(wav.getnframes())), wav.getframerate()
