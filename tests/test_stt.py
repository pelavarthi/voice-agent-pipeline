import pytest

from voice.audio import samples_for_ms, sine_wave
from voice.stt import (
    FakeTranscriber,
    Transcription,
    create_transcriber,
    transcribe_frames,
    transcribe_segment,
)
from voice.vad import SpeechSegment

RATE = 16000
FRAME_MS = 20


def tone(ms=1000, amplitude=12000):
    return sine_wave(440, ms, RATE, amplitude=amplitude)


def frames_of(samples, frame_ms=FRAME_MS, rate=RATE):
    n = samples_for_ms(frame_ms, rate)
    return [samples[i:i + n] for i in range(0, len(samples), n)]


def test_one_shot_returns_first_script_phrase():
    result = FakeTranscriber().transcribe(tone())
    assert result.text == "hello world"
    assert result.is_final
    assert result.start_ms == 0
    assert result.end_ms == 1000
    assert result.duration_ms == 1000


def test_one_shot_advances_through_the_script():
    transcriber = FakeTranscriber()
    texts = [transcriber.transcribe(tone()).text for _ in range(4)]
    assert texts == [
        "hello world",
        "what time is it",
        "tell me a joke",
        "how is the weather today",
    ]


def test_script_cycles_back_to_the_start():
    transcriber = FakeTranscriber(script=("one", "two"))
    texts = [transcriber.transcribe(tone()).text for _ in range(3)]
    assert texts == ["one", "two", "one"]


def test_custom_script_is_used():
    result = FakeTranscriber(script=("custom phrase",)).transcribe(tone())
    assert result.text == "custom phrase"


def test_empty_audio_gives_an_empty_final():
    result = FakeTranscriber().transcribe([])
    assert result == Transcription(text="", end_ms=0.0)
    assert result.is_final


def test_empty_script_is_rejected():
    with pytest.raises(ValueError):
        FakeTranscriber(script=())


def test_unsupported_sample_rate_is_rejected():
    transcriber = FakeTranscriber()
    with pytest.raises(ValueError):
        transcriber.transcribe(tone(), sample_rate=11025)
    with pytest.raises(ValueError):
        transcriber.stream(sample_rate=11025)


def test_streaming_reveals_words_as_audio_arrives():
    stream = FakeTranscriber(script=("hello world",)).stream()
    results = []
    for frame in frames_of(tone()):
        results.extend(stream.push(frame))
    # One word per 200 ms: "hello" at 200 ms, "hello world" at 400 ms.
    assert [r.text for r in results] == ["hello", "hello world"]
    assert all(not r.is_final for r in results)


def test_streaming_finish_returns_the_final():
    stream = FakeTranscriber(script=("hello world",)).stream()
    for frame in frames_of(tone()):
        stream.push(frame)
    finals = stream.finish()
    assert len(finals) == 1
    assert finals[0].text == "hello world"
    assert finals[0].is_final
    assert finals[0].end_ms == 1000


def test_partials_are_prefixes_of_the_final():
    stream = FakeTranscriber(script=("one two three",)).stream()
    partials = []
    for frame in frames_of(tone()):
        partials.extend(stream.push(frame))
    final = stream.finish()[0]
    for partial in partials:
        assert final.text.startswith(partial.text)
    assert partials[-1].text == final.text


def test_finish_with_no_audio_returns_nothing():
    stream = FakeTranscriber().stream()
    assert stream.finish() == []


def test_finish_twice_returns_nothing_the_second_time():
    stream = FakeTranscriber(script=("hello world",)).stream()
    stream.push(tone())
    assert len(stream.finish()) == 1
    assert stream.finish() == []


def test_push_after_finish_returns_nothing():
    stream = FakeTranscriber(script=("hello world",)).stream()
    stream.push(tone())
    stream.finish()
    assert stream.push(tone()) == []


def test_empty_pushes_yield_no_results():
    stream = FakeTranscriber(script=("hello world",)).stream()
    assert stream.push([]) == []
    assert stream.finish() == []


def test_each_stream_takes_the_next_phrase():
    transcriber = FakeTranscriber(script=("first", "second"))
    first = transcriber.stream()
    second = transcriber.stream()
    first.push(tone())
    second.push(tone())
    assert first.finish()[0].text == "first"
    assert second.finish()[0].text == "second"


def test_transcribe_frames_returns_partials_then_final():
    results = transcribe_frames(
        frames_of(tone()), FakeTranscriber(script=("hello world",))
    )
    assert [r.text for r in results] == ["hello", "hello world", "hello world"]
    assert [r.is_final for r in results] == [False, False, True]
    assert results[-1].end_ms == 1000


def test_transcribe_frames_with_no_frames_returns_nothing():
    assert transcribe_frames([], FakeTranscriber()) == []


def test_transcribe_segment_shifts_timestamps():
    segment = SpeechSegment(start_ms=400, end_ms=800, samples=tone(ms=400))
    result = transcribe_segment(segment, FakeTranscriber(script=("hello",)))
    assert result.text == "hello"
    assert result.start_ms == 400
    assert result.end_ms == 800


def test_create_transcriber_factory():
    assert isinstance(create_transcriber("fake"), FakeTranscriber)
    assert isinstance(
        create_transcriber("fake", script=("hi",)), FakeTranscriber
    )
    with pytest.raises(ValueError):
        create_transcriber("whisper")


def test_stream_works_at_8khz():
    stream = FakeTranscriber(script=("hello world",)).stream(sample_rate=8000)
    samples = sine_wave(440, 1000, 8000, amplitude=12000)
    results = []
    for frame in frames_of(samples, rate=8000):
        results.extend(stream.push(frame))
    results.extend(stream.finish())
    assert results[-1].text == "hello world"
    assert results[-1].is_final
    assert results[-1].end_ms == 1000
