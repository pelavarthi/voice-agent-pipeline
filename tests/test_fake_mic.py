from voice.audio import samples_for_ms, sine_wave
from voice.fake_mic import FakeMicSource
from voice.wav_io import write_wav


def test_read_returns_samples_then_silence_padding():
    mic = FakeMicSource([10, 20, 30])
    assert mic.read(2) == [10, 20]
    assert mic.read(4) == [30, 0, 0, 0]
    assert mic.exhausted


def test_looping_source_wraps_around():
    mic = FakeMicSource([1, 2], loop=True)
    assert mic.read(5) == [1, 2, 1, 2, 1]
    assert not mic.exhausted


def test_chunks_yield_fixed_20ms_frames():
    mic = FakeMicSource(sine_wave(440, 100, 16000))
    frames = list(mic.chunks(frame_ms=20))
    assert len(frames) == 5
    assert all(len(f) == samples_for_ms(20, 16000) for f in frames)


def test_from_tone_matches_sine_wave():
    mic = FakeMicSource.from_tone(440, 100, 16000)
    assert mic.read(1600) == sine_wave(440, 100, 16000)


def test_silence_source_is_all_zeros():
    mic = FakeMicSource.silence(50, 16000)
    assert mic.read(800) == [0] * 800


def test_from_wav_loads_a_file(tmp_path):
    path = tmp_path / "tone.wav"
    write_wav(path, [5, 6, 7, 8], 16000)
    mic = FakeMicSource.from_wav(path)
    assert mic.sample_rate == 16000
    assert mic.read(4) == [5, 6, 7, 8]
