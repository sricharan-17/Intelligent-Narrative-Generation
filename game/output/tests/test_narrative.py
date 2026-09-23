import io
import pytest

from game.gamestate.state.models import Character, GameObject, GameState, Location
from game.output.narrative import NarrativeOutput


def test_valid_narrative_response():
    output = NarrativeOutput()
    raw = "  The guard slowly lowers his sword and steps aside.  "
    result = output.format(raw)
    assert result == "The guard slowly lowers his sword and steps aside."


def test_empty_response():
    output = NarrativeOutput()
    with pytest.raises(ValueError, match="empty or whitespace-only"):
        output.format("")


def test_whitespace_only_response():
    output = NarrativeOutput()
    with pytest.raises(ValueError, match="empty or whitespace-only"):
        output.format("   \n\t  ")


@pytest.mark.parametrize("invalid_input", [123, None, ["narrative"], {"text": "hello"}, 4.56, True])
def test_non_string_response(invalid_input):
    output = NarrativeOutput()
    with pytest.raises(TypeError, match="response must be a string"):
        output.format(invalid_input)


def test_preservation_of_valid_response_content():
    output = NarrativeOutput()
    raw = (
        '"Hold your ground!" shouted the captain.\n\n'
        'The dragon breathed a column of fire that illuminated the dark cavern.'
    )
    result = output.format(raw)
    assert result == raw
    assert '"Hold your ground!" shouted the captain.' in result
    assert "dark cavern." in result


def test_deterministic_behavior():
    output = NarrativeOutput()
    raw = "The mysterious merchant offers a glowing blue potion."
    res1 = output.format(raw)
    res2 = output.format(raw)
    assert res1 == res2 == "The mysterious merchant offers a glowing blue potion."


def test_multiple_independent_responses():
    output = NarrativeOutput()
    res1 = output.format("  First turn: You open the chest.  ")
    res2 = output.format("  Second turn: You find a silver key.  ")
    assert res1 == "First turn: You open the chest."
    assert res2 == "Second turn: You find a silver key."


def test_no_unintended_gamestate_mutation():
    output = NarrativeOutput()
    state = GameState(
        location=Location(name="Courtyard", description="A cobblestone yard"),
        inventory=["rusty_key"],
        characters=[Character(name="Guard", present=True)],
        objects=[GameObject(name="iron_gate", state={"locked": True})],
        turn=5,
    )

    initial_turn = state.turn
    initial_location = state.location.name
    initial_inventory = list(state.inventory)
    initial_char_presence = state.characters[0].present
    initial_obj_state = dict(state.objects[0].state)

    # Call NarrativeOutput
    res = output.format("The guard eyes you suspiciously from across the courtyard.")
    assert res == "The guard eyes you suspiciously from across the courtyard."

    # Verify GameState remains 100% unchanged
    assert state.turn == initial_turn
    assert state.location.name == initial_location
    assert state.inventory == initial_inventory
    assert state.characters[0].present == initial_char_presence
    assert state.objects[0].state == initial_obj_state


def test_display_cli_stream_output():
    output = NarrativeOutput()
    stream = io.StringIO()
    raw = "  The wizard casts a illumination spell.  "
    res = output.display(raw, stream=stream)

    assert res == "The wizard casts a illumination spell."
    assert stream.getvalue() == "The wizard casts a illumination spell.\n"


def test_display_invalid_input():
    output = NarrativeOutput()
    stream = io.StringIO()
    with pytest.raises(TypeError):
        output.display(None, stream=stream)

    with pytest.raises(ValueError):
        output.display("   ", stream=stream)

    assert stream.getvalue() == ""
