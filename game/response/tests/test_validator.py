from copy import deepcopy

import pytest

from game.context.models import NarrativeContext
from game.gamestate.interaction.models import Interaction
from game.gamestate.state.models import (
    Character,
    GameObject,
    GameState,
    HistoryEntry,
    Location,
)
from game.response.validator import (
    MAX_RESPONSE_LENGTH,
    ResponseValidationResult,
    ResponseValidator,
)
from rag.knowledge.models import KnowledgeResult, RetrievedKnowledge


def build_context(
    input_text: str = "Where is the key?",
    input_type: str = "speech",
    responder_character: str | None = "Guard",
) -> NarrativeContext:
    state = GameState(
        location=Location("Gatehouse"),
        characters=[
            Character("Player"),
            Character("Guard"),
        ],
        objects=[
            GameObject(name="Chest", state={"locked": True}),
        ],
        inventory=["Sword"],
        history=[
            HistoryEntry(1, "Player", "Hello there.", "speech"),
        ],
        turn=1,
    )

    return NarrativeContext(
        interaction=Interaction(
            input=input_text,
            input_type=input_type,
            input_character="Player",
            responder_character=responder_character,
        ),
        game_state=state,
        history=tuple(state.history),
        retrieved_knowledge=RetrievedKnowledge(
            results=[
                KnowledgeResult(
                    document_id="doc-1",
                    content="The gatehouse guards the castle entrance.",
                    source_type="location",
                    entity_id="gatehouse",
                    title="Gatehouse",
                    score=0.9,
                )
            ]
        ),
    )


def error_codes(result: ResponseValidationResult) -> list[str]:
    return [error.split(":", 1)[0] for error in result.errors]


@pytest.fixture
def validator():
    return ResponseValidator()


def test_valid_response(validator):
    result = validator.validate(
        "The key is hidden beneath the loose stone.",
        build_context(),
    )

    assert result == ResponseValidationResult(
        is_valid=True,
        response="The key is hidden beneath the loose stone.",
        errors=(),
    )


def test_valid_result_preserves_original_response(validator):
    raw = "  The guard shrugs.\n\n  \"Ask the wizard.\"  \n"

    result = validator.validate(raw, build_context())

    assert result.is_valid
    assert result.response is raw


def test_empty_response_is_rejected(validator):
    result = validator.validate("", build_context())

    assert not result.is_valid
    assert result.response == ""
    assert error_codes(result) == ["empty_response"]


@pytest.mark.parametrize("response", ["   ", "\n", "\t", " \n\t "])
def test_whitespace_only_response_is_rejected(validator, response):
    result = validator.validate(response, build_context())

    assert not result.is_valid
    assert result.response == response
    assert error_codes(result) == ["empty_response"]


@pytest.mark.parametrize("response", [None, 123, 4.5, True, ["text"], {"text": "hi"}])
def test_non_string_response_is_rejected(validator, response):
    result = validator.validate(response, build_context())

    assert not result.is_valid
    assert result.response == ""
    assert error_codes(result) == ["invalid_type"]
    assert type(response).__name__ in result.errors[0]


def test_response_header_leakage_is_rejected(validator):
    result = validator.validate(
        "The key is gone.\n### RESPONSE\nThe key is gone.",
        build_context(),
    )

    assert not result.is_valid
    assert error_codes(result) == ["prompt_leakage"]


def test_history_header_leakage_is_rejected(validator):
    result = validator.validate(
        "### HISTORY\nPlayer [speech]: Where is the key?",
        build_context(),
    )

    assert not result.is_valid
    assert error_codes(result) == ["prompt_leakage"]


@pytest.mark.parametrize(
    "response",
    [
        "### WORLD STATE\nRoom objects: chest",
        "I cannot say. ### PLAYER INPUT",
        "Go away.\n  ### Setting",
    ],
)
def test_other_prompt_header_leakage_is_rejected(validator, response):
    result = validator.validate(response, build_context())

    assert not result.is_valid
    assert error_codes(result) == ["prompt_leakage"]


def test_exact_input_echo_is_rejected(validator):
    result = validator.validate(
        "Open the chest",
        build_context(input_text="Open the chest", input_type="action"),
    )

    assert not result.is_valid
    assert error_codes(result) == ["input_echo"]


def test_case_insensitive_input_echo_is_rejected(validator):
    result = validator.validate(
        "OPEN THE CHEST",
        build_context(input_text="Open the chest", input_type="action"),
    )

    assert not result.is_valid
    assert error_codes(result) == ["input_echo"]


def test_whitespace_normalized_input_echo_is_rejected(validator):
    result = validator.validate(
        "  open   the\tchest \n",
        build_context(input_text="Open the chest", input_type="action"),
    )

    assert not result.is_valid
    assert error_codes(result) == ["input_echo"]


def test_response_sharing_input_words_is_not_an_echo(validator):
    result = validator.validate(
        "You try to open the chest, but the lid will not budge.",
        build_context(input_text="Open the chest", input_type="action"),
    )

    assert result.is_valid
    assert result.errors == ()


def test_responder_speaker_prefix_is_rejected(validator):
    result = validator.validate("Guard: Hello.", build_context())

    assert not result.is_valid
    assert error_codes(result) == ["speaker_prefix"]


def test_responder_history_style_prefix_is_rejected(validator):
    result = validator.validate("Guard [speech]: Hello.", build_context())

    assert not result.is_valid
    assert error_codes(result) == ["speaker_prefix"]


@pytest.mark.parametrize(
    "response",
    ["guard: Hello.", "  GUARD :Hello.", "Guard[action]: draws his sword"],
)
def test_speaker_prefix_variants_are_rejected(validator, response):
    result = validator.validate(response, build_context())

    assert not result.is_valid
    assert error_codes(result) == ["speaker_prefix"]


@pytest.mark.parametrize(
    "response",
    [
        "The Guard frowns: \"Move along.\"",
        "Ask the other Guard: he keeps the keys.",
        "Guard duty is dull tonight.",
        "Guardians of old sealed this gate: none may pass.",
    ],
)
def test_responder_name_elsewhere_is_allowed(validator, response):
    result = validator.validate(response, build_context())

    assert result.is_valid
    assert result.errors == ()


def test_no_responder_character_is_supported(validator):
    context = build_context(
        input_text="I hit the chest with my sword.",
        input_type="action",
        responder_character=None,
    )

    result = validator.validate(
        "The sword clangs against the chest. The chest is unbreakable.",
        context,
    )

    assert result.is_valid
    assert result.errors == ()


def test_no_responder_skips_speaker_prefix_check(validator):
    context = build_context(
        input_text="I hit the chest with my sword.",
        input_type="action",
        responder_character=None,
    )

    result = validator.validate("Guard: Hello.", context)

    assert result.is_valid


def test_response_at_maximum_length_is_accepted(validator):
    response = "a" * MAX_RESPONSE_LENGTH

    result = validator.validate(response, build_context())

    assert result.is_valid
    assert result.response == response


def test_response_over_maximum_length_is_rejected(validator):
    response = "a" * (MAX_RESPONSE_LENGTH + 1)

    result = validator.validate(response, build_context())

    assert not result.is_valid
    assert result.response == response
    assert error_codes(result) == ["too_long"]


def test_multiple_failures_are_all_reported(validator):
    response = "Guard: ### RESPONSE " + "a" * MAX_RESPONSE_LENGTH

    result = validator.validate(response, build_context())

    assert not result.is_valid
    assert result.response == response
    assert error_codes(result) == [
        "prompt_leakage",
        "speaker_prefix",
        "too_long",
    ]


def test_invalid_result_does_not_mutate_context(validator):
    context = build_context()
    before = deepcopy(context)

    result = validator.validate(
        "Guard [speech]: ### HISTORY Where is the key?",
        context,
    )

    assert not result.is_valid
    assert context == before


def test_validator_is_deterministic(validator):
    context = build_context()

    first = validator.validate("Guard: Hello.", context)
    second = validator.validate("Guard: Hello.", context)

    assert first == second
