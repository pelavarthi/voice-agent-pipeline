"""Real-time voice agent pipeline: mic capture, streaming STT, LLM turn loop, TTS."""

from voice.audio import (
    bytes_to_samples,
    frame_duration_ms,
    resample_linear,
    samples_for_ms,
    samples_to_bytes,
    validate_sample_rate,
)
from voice.fake_mic import FakeMicSource
from voice.ring_buffer import RingBuffer
from voice.llm import (
    ChatMessage,
    Conversation,
    FakeLanguageModel,
    LanguageModel,
    Turn,
    create_language_model,
    run_conversation,
)
from voice.stt import (
    FakeTranscriber,
    Transcription,
    TranscriptionStream,
    Transcriber,
    create_transcriber,
    transcribe_frames,
    transcribe_segment,
)
from voice.vad import (
    SpeechSegment,
    VadEvent,
    VoiceActivityDetector,
    detect_speech,
    frame_rms,
)
from voice.wav_io import read_wav, write_wav

__all__ = [
    "bytes_to_samples",
    "frame_duration_ms",
    "resample_linear",
    "samples_for_ms",
    "samples_to_bytes",
    "validate_sample_rate",
    "FakeMicSource",
    "RingBuffer",
    "ChatMessage",
    "Conversation",
    "FakeLanguageModel",
    "LanguageModel",
    "Turn",
    "create_language_model",
    "run_conversation",
    "FakeTranscriber",
    "Transcription",
    "TranscriptionStream",
    "Transcriber",
    "create_transcriber",
    "transcribe_frames",
    "transcribe_segment",
    "SpeechSegment",
    "VadEvent",
    "VoiceActivityDetector",
    "detect_speech",
    "frame_rms",
    "read_wav",
    "write_wav",
]
