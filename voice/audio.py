"""Sample-rate helpers and PCM16 conversion.

Everything in the pipeline is mono 16-bit PCM. Samples travel as lists of
ints in [-32768, 32767]; bytes on the wire are little-endian int16.
"""

import math
import struct

SUPPORTED_SAMPLE_RATES = (8000, 16000, 22050, 24000, 44100, 48000)
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_FRAME_MS = 20


def validate_sample_rate(rate):
    """Raise ValueError unless rate is a sample rate the pipeline supports."""
    if rate not in SUPPORTED_SAMPLE_RATES:
        raise ValueError(f"unsupported sample rate: {rate}")
    return rate


def samples_for_ms(milliseconds, sample_rate=DEFAULT_SAMPLE_RATE):
    """How many samples fit in a duration at a sample rate."""
    return int(round(milliseconds * sample_rate / 1000))


def frame_duration_ms(num_samples, sample_rate=DEFAULT_SAMPLE_RATE):
    """Duration of a frame in milliseconds."""
    return num_samples * 1000 / sample_rate


def samples_to_bytes(samples):
    """PCM16 ints -> little-endian bytes."""
    return struct.pack(f"<{len(samples)}h", *samples)


def bytes_to_samples(data):
    """Little-endian PCM16 bytes -> list of ints. Trailing odd byte is dropped."""
    usable = len(data) - (len(data) % 2)
    return list(struct.unpack(f"<{usable // 2}h", data[:usable]))


def resample_linear(samples, from_rate, to_rate):
    """Resample mono PCM16 by linear interpolation.

    Good enough for moving between the rates the pipeline supports (8k
    telephone audio up to 48k mic audio). Not a substitute for a real
    resampler when quality matters.
    """
    if from_rate == to_rate:
        return list(samples)
    if not samples:
        return []
    ratio = from_rate / to_rate
    out_len = int(round(len(samples) * to_rate / from_rate))
    out = []
    for i in range(out_len):
        pos = i * ratio
        lo = int(pos)
        frac = pos - lo
        if lo + 1 >= len(samples):
            out.append(samples[-1])
        else:
            value = samples[lo] * (1 - frac) + samples[lo + 1] * frac
            out.append(int(round(value)))
    return out


def sine_wave(freq_hz, duration_ms, sample_rate=DEFAULT_SAMPLE_RATE, amplitude=12000):
    """A pure tone, handy for tests and for checking a mic path end to end."""
    count = samples_for_ms(duration_ms, sample_rate)
    return [
        int(round(amplitude * math.sin(2 * math.pi * freq_hz * i / sample_rate)))
        for i in range(count)
    ]


def list_input_devices():
    """Names of available mic devices, or [] when sounddevice is absent.

    sounddevice is optional: the pipeline runs its tests and its fake
    sources without it, and real capture only needs it on the machine
    that actually has a microphone.
    """
    try:
        import sounddevice as sd
    except ImportError:
        return []
    return [
        device["name"]
        for device in sd.query_devices()
        if device["max_input_channels"] > 0
    ]
