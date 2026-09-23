import pytest

from game.gamestate.interaction.models import Interaction
from game.gamestate.interaction.processor import InputInteractionProcessor


@pytest.fixture
def processor():
    return InputInteractionProcessor()


def test_process_action(processor):
    interaction = processor.process(
        "Open the chest",
        "action",
        "player",
        "wizard",
    )

    assert interaction == Interaction(
        input="Open the chest",
        input_type="action",
        input_character="player",
        responder_character="wizard",
    )


def test_process_speech(processor):
    interaction = processor.process(
        "I was a policeman",
        "speech",
        "police",
        "a tribesman",
    )

    assert interaction == Interaction(
        input="I was a policeman",
        input_type="speech",
        input_character="police",
        responder_character="a tribesman",
    )


def test_normalizes_whitespace(processor):
    interaction = processor.process(
        "  Open    the   chest  ",
        " action ",
        " player ",
        " wizard ",
    )

    assert interaction.input == "Open the chest"
    assert interaction.input_type == "action"
    assert interaction.input_character == "player"
    assert interaction.responder_character == "wizard"


def test_input_type_is_case_insensitive(processor):
    interaction = processor.process(
        "Hello there",
        "SPEECH",
        "player",
        "wizard",
    )

    assert interaction.input_type == "speech"


def test_empty_input_is_rejected(processor):
    with pytest.raises(ValueError, match="input_text cannot be empty"):
        processor.process(
            "   ",
            "speech",
            "player",
            "wizard",
        )


def test_unsupported_input_type_is_rejected(processor):
    with pytest.raises(ValueError, match="Unsupported input_type"):
        processor.process(
            "Do something",
            "unknown",
            "player",
            "wizard",
        )


def test_empty_input_character_is_rejected(processor):
    with pytest.raises(ValueError, match="input_character cannot be empty"):
        processor.process(
            "Open the chest",
            "action",
            "   ",
            "wizard",
        )


def test_empty_responder_character_is_rejected(processor):
    with pytest.raises(ValueError, match="responder_character cannot be empty"):
        processor.process(
            "Open the chest",
            "action",
            "player",
            "   ",
        )


def test_character_response_is_unchanged(processor):
    interaction = processor.process(
        "Where is the key?",
        "speech",
        "Player",
        "Guard",
    )

    assert interaction == Interaction(
        input="Where is the key?",
        input_type="speech",
        input_character="Player",
        responder_character="Guard",
    )


def test_world_action_without_responder(processor):
    interaction = processor.process(
        "I hit the chest with my sword.",
        "action",
        "Player",
        None,
    )

    assert interaction == Interaction(
        input="I hit the chest with my sword.",
        input_type="action",
        input_character="Player",
        responder_character=None,
    )


def test_responder_defaults_to_none(processor):
    interaction = processor.process(
        "I hit the chest with my sword.",
        "action",
        "Player",
    )

    assert interaction.responder_character is None


def test_none_responder_is_not_converted_to_string(processor):
    interaction = processor.process(
        "Look around",
        "action",
        " Player ",
        None,
    )

    assert interaction.responder_character is None
    assert interaction.responder_character != "None"
    assert interaction.input_character == "Player"


def test_non_string_responder_is_rejected(processor):
    with pytest.raises(TypeError, match="responder_character must be a string or None"):
        processor.process(
            "Open the chest",
            "action",
            "player",
            123,
        )


def test_validation_still_applies_without_responder(processor):
    with pytest.raises(ValueError, match="input_text cannot be empty"):
        processor.process("   ", "action", "player", None)

    with pytest.raises(ValueError, match="Unsupported input_type"):
        processor.process("Do something", "unknown", "player", None)

    with pytest.raises(ValueError, match="input_character cannot be empty"):
        processor.process("Open the chest", "action", "   ", None)
