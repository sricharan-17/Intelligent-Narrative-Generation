import gradio as gr

from game.world.fake_generator import PrototypeFakeGenerator
from game.world.prototype import (
    PLAYER_CHARACTER,
    create_prototype_manager,
    create_prototype_pipeline,
)
from ui.app import (
    NO_RESPONDER,
    create_app,
    play_turn,
    render_history,
    render_location,
    render_world_state,
    responder_dropdown,
)


class FailingGenerator:
    def generate(self, context):
        raise ConnectionError("Space is unavailable")


class EchoGenerator:
    """Returns the player input, which ResponseValidator rejects."""

    def generate(self, context):
        return context.interaction.input


def fake_pipeline():
    return create_prototype_pipeline(PrototypeFakeGenerator())


def test_speech_turn_is_recorded_in_history():
    manager = create_prototype_manager()

    status, details, success = play_turn(
        fake_pipeline(), manager, "Any rumors tonight?", "speech", "Mira"
    )

    assert success
    assert status.startswith("✅ Turn 1 complete.")
    assert "Retrieved knowledge (RAG)" in details

    messages = render_history(manager.get_state())
    assert messages[0] == {
        "role": "user",
        "content": "_[speech]_ Any rumors tonight?",
    }
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"].startswith("**Mira:**")


def test_action_turn_applies_prototype_state_change():
    manager = create_prototype_manager()

    status, _, success = play_turn(
        fake_pipeline(), manager, "take key", "action", NO_RESPONDER
    )

    state = manager.get_state()
    assert success
    assert "Ancient Key" in state.inventory
    assert "inventory +Ancient Key" in status
    assert render_history(state)[1]["content"].startswith("**Narrator:**")


def test_travel_updates_location_and_responders():
    manager = create_prototype_manager()

    status, _, success = play_turn(
        fake_pipeline(), manager, "travel to Ravenmoor", "action", None
    )

    state = manager.get_state()
    assert success
    assert "travelled to **Ravenmoor**" in status
    assert "Ravenmoor" in render_location(state)
    assert responder_dropdown(state, "speech").value == "Captain Rowan"


def test_speech_without_responder_is_refused():
    manager = create_prototype_manager()

    _, _, success = play_turn(
        fake_pipeline(), manager, "Hello?", "speech", NO_RESPONDER
    )

    assert not success
    assert manager.get_state().history == []


def test_empty_input_is_refused():
    manager = create_prototype_manager()

    status, _, success = play_turn(
        fake_pipeline(), manager, "   ", "speech", "Mira"
    )

    assert not success
    assert "type something" in status


def test_generator_error_is_reported_and_state_unchanged():
    manager = create_prototype_manager()
    pipeline = create_prototype_pipeline(FailingGenerator())

    status, _, success = play_turn(
        pipeline, manager, "Hello", "speech", "Mira"
    )

    assert not success
    assert "ConnectionError: Space is unavailable" in status
    assert manager.get_state().turn == 1
    assert manager.get_state().history == []


def test_rejected_response_is_reported():
    manager = create_prototype_manager()
    pipeline = create_prototype_pipeline(EchoGenerator())

    status, details, success = play_turn(
        pipeline, manager, "Hello there", "speech", "Mira"
    )

    assert not success
    assert "rejected by the response validator" in status
    assert "input_echo" in details
    assert manager.get_state().history == []


def test_impossible_action_is_rejected():
    manager = create_prototype_manager()
    play_turn(fake_pipeline(), manager, "travel to Blackstone Castle",
              "action", None)

    # Kael does not carry the Ancient Key yet.
    status, _, success = play_turn(
        fake_pipeline(), manager, "unlock gate", "action", None
    )

    assert not success
    assert "That could not happen" in status


def test_responder_dropdown_defaults():
    state = create_prototype_manager().get_state()

    speech = responder_dropdown(state, "speech")
    action = responder_dropdown(state, "action")

    assert speech.value == "Mira"
    assert action.value == NO_RESPONDER
    assert PLAYER_CHARACTER not in [choice[0] for choice in speech.choices]
    assert responder_dropdown(state, "speech", NO_RESPONDER).value == "Mira"
    assert responder_dropdown(state, "action", "Elara").value == "Elara"


def test_world_state_lists_present_characters_and_objects():
    text = render_world_state(create_prototype_manager().get_state())

    assert "Kael (you), Mira, Elara" in text
    assert "**Sealed Letter**" in text
    assert "Royal Amulet" not in text
    assert "Rusty Sword" in text


def test_create_app_builds_blocks():
    assert isinstance(create_app(fake_pipeline()), gr.Blocks)
