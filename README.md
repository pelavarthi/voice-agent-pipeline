# voice-agent-pipeline

Real-time voice agent pipeline in Python: microphone capture, streaming
speech-to-text, an LLM turn loop, text-to-speech, barge-in, and latency
measurement.

## Day 1: scaffold + audio primitives

- `voice/ring_buffer.py` — thread-safe ring buffer the mic writes to and the
  pipeline reads from; drops the oldest samples instead of falling behind.
- `voice/wav_io.py` — read/write 16-bit mono WAV files.
- `voice/audio.py` — sample-rate helpers, PCM16 conversion, a linear
  resampler, and a tone generator for tests.
- `voice/fake_mic.py` — a fake mic source that plays back recorded or
  generated samples in fixed frames, so every later chunk is testable
  without hardware.

Audio is mono 16-bit PCM throughout, 16 kHz by default. `sounddevice` is
optional: it is only needed on a machine with a real microphone, and
`list_input_devices()` returns `[]` when it is not installed.

## Day 2: voice activity detection

- `voice/vad.py` — energy-based VAD: a frame counts as speech when its RMS
  energy clears a threshold; five loud frames in a row start a segment, and
  400 ms of quiet ends it. Short blips are ignored, short pauses inside a
  sentence do not split it, and 100 ms of pre-roll audio is kept so the
  first syllable is not clipped. `calibrate()` measures room noise and
  raises the threshold above it.

## Day 3: STT adapter interface + fake transcriber

- `voice/stt.py` — a `Transcriber` interface with two modes: one-shot
  `transcribe()` for a whole utterance, and `stream()` for chunked audio
  that yields partial results while the speaker is still talking and a
  final result when the utterance ends. `FakeTranscriber` reads its
  answers from a script, revealing one word per 200 ms of audio, so the
  pipeline is testable with no API key. `transcribe_segment()` runs a
  VAD segment through a transcriber, `transcribe_frames()` drives a
  stream over frames, and `create_transcriber()` builds one by name
  (real adapters arrive on Day 10).

## Day 4: LLM turn loop

- `voice/llm.py` — a `LanguageModel` interface with one-shot `reply()` and
  word-by-word `stream_reply()`, a `FakeLanguageModel` that answers a few
  stock questions and echoes anything else, and a `Conversation` that
  keeps the message history, runs one turn at a time, and trims old
  turns past `max_turns` while always keeping the system prompt first.
  `run_conversation()` drives a list of transcripts through a fresh
  conversation. Real models plug in behind the same interface on Day 10.

## Run the tests

```sh
pip install pytest
python -m pytest
```
