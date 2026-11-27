import pytest

from voice.llm import (
    ChatMessage,
    Conversation,
    FakeLanguageModel,
    Turn,
    create_language_model,
    run_conversation,
)


def history(*texts):
    return [ChatMessage("user", text) for text in texts]


def test_greeting_gets_a_canned_reply():
    model = FakeLanguageModel()
    assert model.reply(history("hello")) == "Hello! How can I help you today?"


def test_reply_is_case_insensitive():
    model = FakeLanguageModel()
    assert model.reply(history("HELLO")) == "Hello! How can I help you today?"


def test_joke_gets_the_canned_joke():
    model = FakeLanguageModel()
    assert "programmer" in model.reply(history("tell me a joke"))


def test_unknown_input_is_echoed_back():
    model = FakeLanguageModel()
    assert model.reply(history("I like turtles")) == "You said: I like turtles"


def test_empty_transcript_gets_no_reply():
    model = FakeLanguageModel()
    assert model.reply(history("   ")) == ""


def test_no_user_message_raises():
    model = FakeLanguageModel()
    with pytest.raises(ValueError):
        model.reply([ChatMessage("system", "be helpful")])


def test_stream_yields_the_reply_word_by_word():
    model = FakeLanguageModel()
    chunks = list(model.stream_reply(history("hello")))
    assert "".join(chunks).strip() == "Hello! How can I help you today?"
    assert all(chunk.endswith(" ") for chunk in chunks)


def test_stream_echo_matches_one_shot_reply():
    model = FakeLanguageModel()
    messages = history("something new")
    streamed = "".join(model.stream_reply(messages)).strip()
    assert streamed == model.reply(messages)


def test_factory_builds_a_fake_model_by_default():
    assert isinstance(create_language_model(), FakeLanguageModel)


def test_factory_rejects_unknown_kinds():
    with pytest.raises(ValueError):
        create_language_model("gpt-5")


def test_conversation_starts_with_the_system_prompt():
    convo = Conversation()
    assert convo.messages[0].role == "system"
    assert "voice assistant" in convo.messages[0].content
    assert convo.turn_count == 0


def test_one_turn_records_user_and_reply():
    convo = Conversation()
    reply = convo.turn("hello")
    assert reply == "Hello! How can I help you today?"
    assert convo.turn_count == 1
    roles = [m.role for m in convo.messages]
    assert roles == ["system", "user", "assistant"]
    assert convo.messages[1].content == "hello"
    assert convo.messages[2].content == reply


def test_turns_build_on_each_other():
    convo = Conversation()
    convo.turn("hello")
    reply = convo.turn("tell me a joke")
    assert "programmer" in reply
    assert convo.turn_count == 2
    assert convo.turns[1] == Turn(
        user_text="tell me a joke", reply_text=reply
    )


def test_empty_turn_is_skipped_entirely():
    convo = Conversation()
    assert convo.turn("   ") == ""
    assert convo.turn_count == 0
    assert len(convo.messages) == 1


def test_history_is_trimmed_to_max_turns():
    convo = Conversation(max_turns=2)
    for i in range(4):
        convo.turn(f"message {i}")
    assert convo.turn_count == 4
    assert convo.messages[0].role == "system"
    user_texts = [m.content for m in convo.messages if m.role == "user"]
    assert user_texts == ["message 2", "message 3"]


def test_reset_keeps_only_the_system_prompt():
    convo = Conversation()
    convo.turn("hello")
    convo.reset()
    assert convo.turn_count == 0
    assert len(convo.messages) == 1
    assert convo.messages[0].role == "system"


def test_custom_system_prompt_is_used():
    convo = Conversation(system_prompt="Be brief.")
    assert convo.messages[0].content == "Be brief."


def test_max_turns_below_one_raises():
    with pytest.raises(ValueError):
        Conversation(max_turns=0)


def test_stream_turn_records_the_full_reply():
    convo = Conversation()
    chunks = list(convo.stream_turn("hello"))
    assert "".join(chunks).strip() == "Hello! How can I help you today?"
    assert convo.turn_count == 1
    assert convo.messages[-1].content == "Hello! How can I help you today?"


def test_stream_turn_skips_empty_input():
    convo = Conversation()
    assert list(convo.stream_turn("  ")) == []
    assert convo.turn_count == 0


def test_run_conversation_returns_each_turn():
    turns = run_conversation(["hello", "tell me a joke"])
    assert len(turns) == 2
    assert turns[0].user_text == "hello"
    assert turns[0].reply_text == "Hello! How can I help you today?"
    assert "programmer" in turns[1].reply_text


def test_messages_property_returns_a_copy():
    convo = Conversation()
    messages = convo.messages
    messages.append(ChatMessage("user", "sneaky"))
    assert len(convo.messages) == 1
