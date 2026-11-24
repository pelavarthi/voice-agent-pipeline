import wave

import pytest

from voice.audio import sine_wave
from voice.wav_io import read_wav, write_wav


def test_wav_round_trip_preserves_samples_and_rate(tmp_path):
    path = tmp_path / "tone.wav"
    samples = sine_wave(440, 250, 16000)
    write_wav(path, samples, 16000)
    back, rate = read_wav(path)
    assert rate == 16000
    assert back == samples


def test_wav_round_trip_at_8k(tmp_path):
    path = tmp_path / "low.wav"
    samples = [100, -100, 0, 500]
    write_wav(path, samples, 8000)
    back, rate = read_wav(path)
    assert (back, rate) == (samples, 8000)


def test_read_wav_rejects_stereo(tmp_path):
    path = tmp_path / "stereo.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00\x00\x00")
    with pytest.raises(ValueError, match="mono"):
        read_wav(path)


def test_write_wav_rejects_unknown_sample_rate(tmp_path):
    with pytest.raises(ValueError):
        write_wav(tmp_path / "bad.wav", [1, 2, 3], 12345)
