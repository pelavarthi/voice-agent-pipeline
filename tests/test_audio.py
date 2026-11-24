import pytest

from voice.audio import (
    bytes_to_samples,
    frame_duration_ms,
    resample_linear,
    samples_for_ms,
    samples_to_bytes,
    sine_wave,
    validate_sample_rate,
)


def test_samples_for_ms_and_frame_duration_round_trip():
    assert samples_for_ms(20, 16000) == 320
    assert samples_for_ms(1000, 8000) == 8000
    assert frame_duration_ms(320, 16000) == pytest.approx(20.0)


def test_validate_sample_rate_rejects_unknown_rates():
    assert validate_sample_rate(16000) == 16000
    with pytest.raises(ValueError):
        validate_sample_rate(12345)


def test_pcm_bytes_round_trip():
    samples = [0, 1, -1, 32767, -32768, 12345]
    assert bytes_to_samples(samples_to_bytes(samples)) == samples


def test_bytes_to_samples_drops_trailing_odd_byte():
    assert bytes_to_samples(b"\x01\x00\xff") == [1]


def test_resample_same_rate_is_identity():
    samples = [10, 20, 30]
    assert resample_linear(samples, 16000, 16000) == samples


def test_resample_down_halves_length():
    samples = list(range(0, 1600, 10))
    out = resample_linear(samples, 16000, 8000)
    assert len(out) == len(samples) // 2
    assert out[0] == samples[0]


def test_resample_up_doubles_length():
    samples = [0, 100, 200, 300]
    out = resample_linear(samples, 8000, 16000)
    assert len(out) == 8
    assert out[0] == 0
    assert out[-1] == 300


def test_sine_wave_has_expected_length_and_range():
    wave = sine_wave(440, 100, 16000)
    assert len(wave) == 1600
    assert max(wave) <= 12000
    assert min(wave) >= -12000
    assert max(wave) > 10000  # actually oscillates, not flat
