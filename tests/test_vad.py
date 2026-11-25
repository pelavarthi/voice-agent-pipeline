from voice.audio import samples_for_ms, sine_wave
from voice.fake_mic import FakeMicSource
from voice.vad import VoiceActivityDetector, detect_speech, frame_rms

RATE = 16000
FRAME_MS = 20


def make_stream(*parts):
    """Build samples from ("silence"|"tone"|"quiet", milliseconds) parts."""
    out = []
    for kind, ms in parts:
        if kind == "silence":
            out += [0] * samples_for_ms(ms, RATE)
        elif kind == "tone":
            out += sine_wave(440, ms, RATE, amplitude=12000)
        elif kind == "quiet":
            out += sine_wave(440, ms, RATE, amplitude=100)
        elif kind == "noise":
            out += sine_wave(440, ms, RATE, amplitude=800)
    return out


def frames_of(samples, frame_ms=FRAME_MS, rate=RATE):
    n = samples_for_ms(frame_ms, rate)
    return [samples[i:i + n] for i in range(0, len(samples), n)]


def test_frame_rms_silence_is_zero():
    assert frame_rms([0] * 320) == 0.0
    assert frame_rms([]) == 0.0


def test_frame_rms_tone_is_loud():
    rms = frame_rms(sine_wave(440, 20, RATE, amplitude=12000))
    assert 8000 < rms < 9000  # sine RMS is amplitude / sqrt(2)


def test_silence_produces_no_segments():
    detector = VoiceActivityDetector()
    assert detector.segments(frames_of(make_stream(("silence", 2000)))) == []


def test_tone_burst_is_one_segment_with_pre_roll():
    samples = make_stream(("silence", 500), ("tone", 300), ("silence", 1000))
    segments = detect_speech(samples)
    assert len(segments) == 1
    segment = segments[0]
    # Starts 100 ms before the tone (pre-roll), ends with the tone.
    assert segment.start_ms == 400
    assert segment.end_ms == 800
    assert segment.duration_ms == 400
    assert len(segment.samples) == samples_for_ms(400, RATE)


def test_segment_samples_hold_the_tone_and_the_silence_before_it():
    samples = make_stream(("silence", 500), ("tone", 300), ("silence", 1000))
    segment = detect_speech(samples)[0]
    assert segment.samples[:samples_for_ms(100, RATE)] == [0] * samples_for_ms(100, RATE)
    assert max(segment.samples) > 10000


def test_short_blip_below_min_speech_is_ignored():
    samples = make_stream(("silence", 500), ("tone", 40), ("silence", 1000))
    assert detect_speech(samples) == []


def test_100ms_burst_is_detected():
    samples = make_stream(("silence", 500), ("tone", 100), ("silence", 1000))
    assert len(detect_speech(samples)) == 1


def test_quiet_sound_below_threshold_is_ignored():
    samples = make_stream(("silence", 500), ("quiet", 500), ("silence", 500))
    assert detect_speech(samples) == []


def test_short_silence_gap_keeps_one_segment():
    samples = make_stream(
        ("silence", 500), ("tone", 300), ("silence", 200),
        ("tone", 300), ("silence", 1000),
    )
    segments = detect_speech(samples)
    assert len(segments) == 1
    assert segments[0].start_ms == 400
    assert segments[0].end_ms == 1300


def test_long_silence_gap_splits_into_two_segments():
    samples = make_stream(
        ("silence", 500), ("tone", 300), ("silence", 600),
        ("tone", 300), ("silence", 1000),
    )
    segments = detect_speech(samples)
    assert len(segments) == 2
    assert segments[0].start_ms == 400
    assert segments[0].end_ms == 800
    assert segments[1].start_ms == 1300
    assert segments[1].end_ms == 1700


def test_flush_closes_a_segment_at_the_end_of_the_stream():
    samples = make_stream(("silence", 500), ("tone", 300))
    segments = detect_speech(samples)
    assert len(segments) == 1
    assert segments[0].start_ms == 400
    assert segments[0].end_ms == 800


def test_start_and_end_events_arrive_in_order():
    samples = make_stream(("silence", 500), ("tone", 300), ("silence", 1000))
    detector = VoiceActivityDetector()
    events = [e for f in frames_of(samples) if (e := detector.process(f))]
    tail = detector.flush()
    if tail is not None:
        events.append(tail)
    assert [e.kind for e in events] == ["start", "end"]
    assert events[0].start_ms == 400
    assert events[1].end_ms == 800
    assert events[1].segment.start_ms == 400


def test_speaking_flag_tracks_the_state_machine():
    detector = VoiceActivityDetector()
    assert not detector.speaking
    for frame in frames_of(sine_wave(440, 100, RATE, amplitude=12000)):
        detector.process(frame)
    assert detector.speaking
    for frame in frames_of([0] * samples_for_ms(500, RATE)):
        detector.process(frame)
    assert not detector.speaking


def test_streaming_from_a_fake_mic():
    samples = make_stream(("silence", 500), ("tone", 300), ("silence", 1000))
    mic = FakeMicSource(samples)
    detector = VoiceActivityDetector()
    found = detector.segments(mic.chunks(frame_ms=20))
    assert len(found) == 1
    assert found[0].start_ms == 400
    assert found[0].end_ms == 800


def test_calibrate_raises_the_threshold_above_room_noise():
    samples = make_stream(("silence", 500), ("noise", 500), ("silence", 500))
    # Noisy room (amplitude 800, RMS ~566) trips the default threshold.
    assert len(detect_speech(samples)) == 1
    detector = VoiceActivityDetector()
    detector.calibrate(frames_of(sine_wave(440, 500, RATE, amplitude=800)))
    assert detector.threshold > 2000
    assert detector.segments(frames_of(samples)) == []
    # But a real voice still gets through.
    loud = make_stream(("silence", 500), ("tone", 300), ("silence", 1000))
    assert len(detector.segments(frames_of(loud))) == 1


def test_detector_works_at_8khz():
    samples = [0] * samples_for_ms(500, 8000)
    samples += sine_wave(440, 300, 8000, amplitude=12000)
    samples += [0] * samples_for_ms(1000, 8000)
    segments = detect_speech(samples, sample_rate=8000)
    assert len(segments) == 1
    assert segments[0].start_ms == 400
    assert segments[0].end_ms == 800
