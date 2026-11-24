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

## Run the tests

```sh
pip install pytest
python -m pytest
```
