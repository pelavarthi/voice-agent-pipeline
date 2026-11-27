"""The LLM turn loop: take a transcript, build a reply, hold state.

A LanguageModel turns a chat history into a reply. It works two ways:
one-shot, for a full reply at once, and streaming, for a reply that
arrives word by word so TTS can start speaking before the sentence is
done. The fake model answers from a few canned rules and echoes anything
else, so the whole loop is testable with no API key. A real model plugs
in behind the same interface on Day 10.

Conversation is the state: it keeps the message history, adds each
user transcript and each reply to it, and trims old turns so the
history never grows without bound.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful voice assistant. Keep replies short, one or two "
    "sentences, because they will be spoken out loud."
)
DEFAULT_MAX_TURNS = 10


@dataclass
class ChatMessage:
    """One message in the conversation: system, user, or assistant."""

    role: str
    content: str


@dataclass
class Turn:
    """One round trip: what the user said and what the agent replied."""

    user_text: str
    reply_text: str


class LanguageModel(ABC):
    """Turns a chat history into a reply, one-shot or streaming."""

    @abstractmethod
    def reply(self, messages):
        """Reply to the history; returns the assistant's text."""

    @abstractmethod
    def stream_reply(self, messages):
        """Reply word by word; yields text chunks, last one ends the reply."""


class FakeLanguageModel(LanguageModel):
    """A model that answers from a few canned rules and echoes the rest.

    Greetings, thanks, and a couple of stock questions get a fixed
    answer; anything else is echoed back, which keeps the turn loop
    honest — the reply always comes from the transcript, never from
    nowhere. Empty transcripts get no reply at all.
    """

    def __init__(self, canned=None):
        self.canned = canned or {
            "hello": "Hello! How can I help you today?",
            "hi": "Hi there! What can I do for you?",
            "how are you": "I'm doing well, thanks for asking!",
            "thank you": "You're welcome!",
            "thanks": "You're welcome!",
            "tell me a joke": (
                "Why did the programmer quit their job? "
                "Because they didn't get arrays."
            ),
            "what time is it": (
                "I don't have a clock handy, but I'm listening."
            ),
            "how is the weather today": (
                "I can't check the weather yet — that arrives on Day 12."
            ),
        }

    def _last_user_text(self, messages):
        for message in reversed(messages):
            if message.role == "user":
                return message.content.strip()
        raise ValueError("no user message in history")

    def reply(self, messages):
        text = self._last_user_text(messages)
        if not text:
            return ""
        return self.canned.get(text.lower(), f"You said: {text}")

    def stream_reply(self, messages):
        for word in self.reply(messages).split(" "):
            yield word + " "


def create_language_model(kind="fake", **kwargs):
    """Build a language model by name; Day 10 adds real ones behind keys."""
    if kind == "fake":
        return FakeLanguageModel(**kwargs)
    raise ValueError(f"unknown language model: {kind}")


class Conversation:
    """Holds the chat history and runs one turn at a time.

    Each turn appends the user's transcript and the model's reply to
    the history. Old turns are trimmed once the history passes
    max_turns, but the system prompt is always kept first.
    """

    def __init__(self, model=None, system_prompt=DEFAULT_SYSTEM_PROMPT,
                 max_turns=DEFAULT_MAX_TURNS):
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        self.model = model or FakeLanguageModel()
        self.max_turns = max_turns
        self._messages = [ChatMessage("system", system_prompt)]
        self.turns = []

    @property
    def messages(self):
        return list(self._messages)

    @property
    def turn_count(self):
        return len(self.turns)

    def _trim(self):
        keep = self.max_turns * 2
        if len(self._messages) > keep + 1:
            self._messages = [self._messages[0]] + self._messages[-keep:]

    def turn(self, transcript):
        """One turn: transcript in, reply out. Empty input is skipped."""
        text = transcript.strip()
        if not text:
            return ""
        self._messages.append(ChatMessage("user", text))
        reply = self.model.reply(self._messages)
        self._messages.append(ChatMessage("assistant", reply))
        self.turns.append(Turn(user_text=text, reply_text=reply))
        self._trim()
        return reply

    def stream_turn(self, transcript):
        """One turn with the reply streamed word by word.

        Yields each chunk as it arrives, then records the full reply
        in the history once the stream ends.
        """
        text = transcript.strip()
        if not text:
            return
        self._messages.append(ChatMessage("user", text))
        reply = ""
        for chunk in self.model.stream_reply(self._messages):
            reply += chunk
            yield chunk
        reply = reply.strip()
        self._messages.append(ChatMessage("assistant", reply))
        self.turns.append(Turn(user_text=text, reply_text=reply))
        self._trim()

    def reset(self):
        """Start a fresh conversation, keeping the system prompt."""
        system = self._messages[0]
        self._messages = [system]
        self.turns = []


def run_conversation(transcripts, model=None, **kwargs):
    """Run a list of transcripts through a fresh conversation.

    Returns the list of turns, so a demo or a test can drive several
    exchanges without managing the conversation object itself.
    """
    conversation = Conversation(model=model, **kwargs)
    for transcript in transcripts:
        conversation.turn(transcript)
    return list(conversation.turns)
