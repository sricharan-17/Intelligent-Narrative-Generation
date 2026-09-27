import json
import subprocess
import sys
from pathlib import Path

import pytest

from game.gamestate.interaction.actions import StateChangeAction
from game.gamestate.state.models import GameState, HistoryEntry
from game.response.validator import ResponseValidator
from game.world.fake_generator import PrototypeFakeGenerator
from game.world.prototype import (
    CHARACTER_LOCATIONS,
    KNOWLEDGE_FILE,
    LOCATIONS,
    PLAYER_CHARACTER,
    PROTOTYPE_ACTIONS,
    create_prototype_manager,
    create_prototype_pipeline,
    create_prototype_retriever,
    create_prototype_state,
    run_prototype_turn,
    state_changes_for,
    travel_destination,
)
from rag.knowledge.loader import load_knowledge_entries_from_json
from rag.knowledge.models import RAGQuery


PROJECT_ROOT = Path(__file__).resolve().parents[3]

EXPECTED_OBJECTS = {
    "Rusty Sword": {"condition": "worn"},
    "Sealed Letter": {"opened": False},
    "Ancient Key": {"discovered": False},
    "Royal Amulet": {"discovered": False},
    "Ashen Crown": {"discovered": False},
    "Tavern Ledger": {"examined": False},
    "Castle Gate": {"locked": True},
    "Hidden Chest": {"opened": False},
}

EXPECTED_KNOWLEDGE_ENTITIES = {
    "Ravenmoor",
    "Old Tavern",
    "Mira",
    "Elara",
    "Blackstone Castle",
    "Ashen Crown",
    "Whispering Forest",
    "Ancient Key",
    "Royal Amulet",
    "Sealed Letter",
    "Castle Gate",
    "Royal Disappearance",
    "Tavern Rumors",
    "Captain Rowan",
    "The Warden",
}


class RecordingGenerator(PrototypeFakeGenerator):
    """PrototypeFakeGenerator that also records every context it receives."""

    def __init__(self):
        self.contexts = []

    def generate(self, context):
        self.contexts.append(context)
        return super().generate(context)


class FixedGenerator:
    def __init__(self, response):
        self.response = response

    def generate(self, context):
        return self.response


def find_object(state: GameState, name: str):
    return next(obj for obj in state.objects if obj.name == name)


def presence(state: GameState) -> dict[str, bool]:
    return {c.name: c.present for c in state.characters}


@pytest.fixture(scope="module")
def retriever():
    return create_prototype_retriever()


@pytest.fixture
def manager():
    return create_prototype_manager()


@pytest.fixture
def generator():
    return RecordingGenerator()


@pytest.fixture
def pipeline(generator, retriever):
    return create_prototype_pipeline(generator, retriever)


# --------------------------------------------------
# Initial world
# --------------------------------------------------


def test_initial_setting_and_turn():
    state = create_prototype_state()

    assert state.setting["name"] == "Kingdom of Eldoria"
    assert state.setting["story"] == "The Ashen Crown"
    assert state.turn == 1
    assert state.history == []
    assert state.recent_events == []


def test_initial_location_is_old_tavern():
    state = create_prototype_state()

    assert state.location is not None
    assert state.location.name == "Old Tavern"
    assert "tavern" in state.location.description


def test_initial_character_presence():
    state = create_prototype_state()

    assert presence(state) == {
        "Kael": True,
        "Mira": True,
        "Elara": True,
        "Captain Rowan": False,
        "The Warden": False,
    }
    assert all(c.description for c in state.characters)


def test_rusty_sword_is_the_only_inventory_item():
    assert create_prototype_state().inventory == ["Rusty Sword"]


def test_expected_objects_exist_with_initial_state():
    state = create_prototype_state()

    assert {obj.name: obj.state for obj in state.objects} == EXPECTED_OBJECTS
    assert all(obj.description for obj in state.objects)


def test_expected_available_actions():
    assert create_prototype_state().available_actions == [
        "look",
        "talk",
        "inspect",
        "search",
        "leave",
        "travel",
        "take",
        "open",
    ]


def test_expected_relationships():
    relationships = {
        (r["source"], r["target"]): r["status"]
        for r in create_prototype_state().relationships
    }

    assert relationships == {
        ("Kael", "Mira"): "neutral",
        ("Kael", "Elara"): "unknown",
        ("Kael", "Captain Rowan"): "neutral",
        ("Kael", "The Warden"): "unknown",
        ("Mira", "Elara"): "suspicious",
        ("Captain Rowan", "Kael"): "cautious",
    }


def test_each_call_returns_an_independent_world():
    first = create_prototype_manager()
    second = create_prototype_manager()

    first.add_inventory_item("Torch")

    assert "Torch" not in second.get_state().inventory
    assert "Torch" not in create_prototype_state().inventory


# --------------------------------------------------
# RAG knowledge
# --------------------------------------------------


def test_knowledge_file_is_a_tracked_project_file():
    assert KNOWLEDGE_FILE.exists()
    assert KNOWLEDGE_FILE.is_relative_to(PROJECT_ROOT / "game" / "world")

    with open(KNOWLEDGE_FILE, encoding="utf-8") as f:
        assert isinstance(json.load(f), list)


def test_knowledge_loads_with_existing_loader():
    entries = load_knowledge_entries_from_json(str(KNOWLEDGE_FILE))

    assert len(entries) == 15
    assert {entry.entity_id for entry in entries} == EXPECTED_KNOWLEDGE_ENTITIES
    assert len({entry.document_id for entry in entries}) == 15
    assert all(entry.title and entry.content for entry in entries)
    assert {entry.source_type for entry in entries} <= {
        "location",
        "character",
        "item",
        "history",
        "world_lore",
    }


def test_index_contains_every_knowledge_entry(retriever):
    assert retriever.index.size() == 15


def test_knowledge_entities_match_game_state_names():
    state = create_prototype_state()
    entries = load_knowledge_entries_from_json(str(KNOWLEDGE_FILE))
    by_type: dict[str, set[str]] = {}

    for entry in entries:
        by_type.setdefault(entry.source_type, set()).add(entry.entity_id)

    npc_names = {c.name for c in state.characters} - {PLAYER_CHARACTER}
    item_names = {obj.name for obj in state.objects}

    assert by_type["character"] == npc_names
    assert by_type["location"] == set(LOCATIONS)
    assert by_type["item"] <= item_names


@pytest.mark.parametrize(
    "query, responder, expected_document",
    [
        ("Tell me about this place.", "Mira", "loc_old_tavern"),
        ("Tell me about this place.", "Mira", "char_mira"),
        ("What do you know about Elara?", "Mira", "char_elara"),
        ("Tell me about Ravenmoor.", "Mira", "loc_ravenmoor"),
        ("What happened to the Ashen Crown?", "Elara", "item_ashen_crown"),
        ("Where is the sealed letter?", "Mira", "item_sealed_letter"),
        ("Who guards Blackstone Castle?", "Mira", "loc_blackstone_castle"),
    ],
)
def test_retrieval_returns_relevant_knowledge(
    retriever,
    query,
    responder,
    expected_document,
):
    # Built the same way ContextBuilder builds its RAGQuery.
    rag_query = RAGQuery(
        query=query,
        input_type="speech",
        location="Old Tavern",
        input_character=PLAYER_CHARACTER,
        responder_character=responder,
        relevant_entities=[PLAYER_CHARACTER, responder],
    )

    results = retriever.retrieve(rag_query).results

    assert expected_document in [result.document_id for result in results]


def test_unrelated_query_does_not_return_topic_lore(retriever):
    results = retriever.retrieve(RAGQuery(query="xylophone quantum")).results

    assert results == []


# --------------------------------------------------
# Explicit prototype state changes
# --------------------------------------------------


def test_state_changes_for_known_command():
    changes = state_changes_for("Open the letter", create_prototype_state())
    assert changes == ()

    changes = state_changes_for("  OPEN   Letter. ", create_prototype_state())
    assert changes == (
        StateChangeAction(
            action_type="object_state_change",
            target="Sealed Letter",
            changes={"opened": True},
        ),
    )


def test_state_changes_for_alias():
    state = create_prototype_state()

    assert state_changes_for("take ancient key", state) == (
        state_changes_for("take key", state)
    )
    assert state_changes_for("take key", state) != ()


def test_state_changes_for_unknown_command_is_empty():
    assert state_changes_for("dance on the table", create_prototype_state()) == ()


def test_state_changes_for_wrong_location_is_empty():
    state = create_prototype_state()

    assert state_changes_for("take amulet", state) == ()
    assert state_changes_for("unlock gate", state) == ()


def test_prototype_actions_only_use_supported_action_types():
    action_types = {
        change.action_type
        for _, changes in PROTOTYPE_ACTIONS.values()
        for change in changes
    }

    assert action_types <= {"object_state_change", "inventory_change"}


def test_prototype_action_targets_exist_in_the_world():
    object_names = {obj.name for obj in create_prototype_state().objects}

    for location, changes in PROTOTYPE_ACTIONS.values():
        assert location in LOCATIONS
        for change in changes:
            if change.target is not None:
                assert change.target in object_names


@pytest.mark.parametrize(
    "command, expected",
    [
        ("travel to Whispering Forest", "Whispering Forest"),
        ("Go to  blackstone castle.", "Blackstone Castle"),
        ("travel to Ravenmoor", "Ravenmoor"),
        ("travel to the moon", None),
        ("whispering forest", None),
    ],
)
def test_travel_destination(command, expected):
    destination = travel_destination(command)

    if expected is None:
        assert destination is None
    else:
        assert destination.name == expected


# --------------------------------------------------
# Fake generator
# --------------------------------------------------


def test_fake_generator_does_not_need_the_llm_package():
    # Importing the prototype and fake generator must not load llm or
    # torch, so none of these tests need the fine-tuned model.
    code = (
        "import sys, game.world.prototype, game.world.fake_generator; "
        "print(any(name == 'llm' or name.startswith(('llm.', 'torch')) "
        "for name in sys.modules))"
    )

    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert completed.stdout.strip() == "False"


def test_fake_generator_is_deterministic_and_passes_validation(
    pipeline,
    generator,
    manager,
):
    run_prototype_turn(
        pipeline,
        manager,
        "What do you know about Elara?",
        "speech",
        "Mira",
    )
    context = generator.contexts[0]

    first = PrototypeFakeGenerator().generate(context)
    second = PrototypeFakeGenerator().generate(context)

    assert first == second
    assert first.startswith("Mira studies Kael before answering.")
    assert ResponseValidator().validate(first, context).is_valid


# --------------------------------------------------
# Full pipeline against the prototype world
# --------------------------------------------------


def test_prototype_pipeline_speech_turn(pipeline, generator, manager):
    result = run_prototype_turn(
        pipeline,
        manager,
        "What do you know about Elara?",
        "speech",
        "Mira",
    )

    assert result.success
    assert result.validation.errors == ()
    assert result.state_update.applied_changes == ()

    # ContextBuilder received the prototype world and its RAG knowledge.
    context = generator.contexts[0]
    document_ids = [r.document_id for r in context.retrieved_knowledge.results]

    assert context.game_state.location.name == "Old Tavern"
    assert context.interaction.responder_character == "Mira"
    assert "char_elara" in document_ids
    assert "loc_old_tavern" in document_ids

    # The generated response uses the retrieved Elara lore.
    assert "Elara is a calm, secretive traveler" in result.output

    state = manager.get_state()

    assert state.turn == 2
    assert state.history == [
        HistoryEntry(1, "Kael", "What do you know about Elara?", "speech"),
        HistoryEntry(1, "Mira", result.raw_response, "speech"),
    ]


def test_prototype_pipeline_world_action_uses_narrator(pipeline, manager):
    result = run_prototype_turn(pipeline, manager, "look around")

    assert result.success
    assert result.output.startswith("Kael acts.")

    state = manager.get_state()

    assert [entry.speaker for entry in state.history] == ["Kael", "Narrator"]
    assert state.turn == 2


def test_explicit_state_change_is_applied(pipeline, generator, manager):
    result = run_prototype_turn(pipeline, manager, "open letter")

    assert result.success
    assert result.state_update.applied_changes == (
        state_changes_for("open letter", create_prototype_state())
    )

    state = manager.get_state()

    assert find_object(state, "Sealed Letter").state["opened"] is True
    assert state.turn == 2
    assert state.recent_events[0].turn == 1
    assert state.recent_events[0].event_type == "object_state_change"

    # The generator saw the state before the change was applied.
    before = generator.contexts[0].game_state
    assert find_object(before, "Sealed Letter").state["opened"] is False


def test_take_key_updates_object_and_inventory(pipeline, manager):
    result = run_prototype_turn(pipeline, manager, "take the key")
    assert result.state_update.applied_changes == ()

    result = run_prototype_turn(pipeline, manager, "take key")

    assert result.success
    assert len(result.state_update.applied_changes) == 2

    state = manager.get_state()

    assert find_object(state, "Ancient Key").state["discovered"] is True
    assert state.inventory == ["Rusty Sword", "Ancient Key"]
    assert state.turn == 3


def test_invalid_state_change_does_not_partially_mutate(pipeline, manager):
    manager.set_location(LOCATIONS["Blackstone Castle"])
    before = manager.get_state()

    # Kael does not carry the Ancient Key, so unlocking fails atomically.
    result = run_prototype_turn(pipeline, manager, "unlock gate")

    assert result.validation.is_valid
    assert not result.success
    assert result.state_update.applied_changes == ()
    assert len(result.state_update.rejected_changes) == 2
    assert result.output is None

    after = manager.get_state()

    assert after == before
    assert find_object(after, "Castle Gate").state["locked"] is True
    assert after.turn == before.turn
    assert after.history == []


def test_rejected_response_does_not_change_state(retriever, manager):
    before = manager.get_state()
    pipeline = create_prototype_pipeline(
        FixedGenerator("### RESPONSE\nThe letter opens."),
        retriever,
    )

    result = run_prototype_turn(pipeline, manager, "open letter")

    assert not result.success
    assert result.state_update is None
    assert manager.get_state() == before


def test_travel_moves_player_and_updates_presence(pipeline, manager):
    result = run_prototype_turn(
        pipeline,
        manager,
        "travel to Blackstone Castle",
    )

    assert result.success

    state = manager.get_state()

    assert state.location.name == "Blackstone Castle"
    assert state.turn == 2
    assert presence(state) == {
        "Kael": True,
        "Mira": False,
        "Elara": False,
        "Captain Rowan": False,
        "The Warden": True,
    }


def test_failed_turn_does_not_travel(retriever, manager):
    pipeline = create_prototype_pipeline(FixedGenerator(""), retriever)

    result = run_prototype_turn(pipeline, manager, "travel to Ravenmoor")

    assert not result.success
    assert manager.get_state().location.name == "Old Tavern"


def test_prototype_progression(pipeline, generator, manager):
    commands = [
        "open letter",
        "open chest",
        "take key",
        "travel to Whispering Forest",
        "take amulet",
        "travel to Blackstone Castle",
        "unlock gate",
        "take crown",
    ]

    for command in commands:
        assert run_prototype_turn(pipeline, manager, command).success, command

    state = manager.get_state()

    assert state.location.name == "Blackstone Castle"
    assert state.turn == 1 + len(commands)
    assert state.inventory == ["Rusty Sword", "Royal Amulet", "Ashen Crown"]
    assert find_object(state, "Sealed Letter").state["opened"] is True
    assert find_object(state, "Hidden Chest").state["opened"] is True
    assert find_object(state, "Castle Gate").state["locked"] is False
    assert find_object(state, "Ashen Crown").state["discovered"] is True
    assert len(state.history) == 2 * len(commands)

    # RAG followed the player: the forest turn retrieved forest lore.
    forest_context = generator.contexts[4]
    assert forest_context.game_state.location.name == "Whispering Forest"
    assert "loc_whispering_forest" in [
        r.document_id for r in forest_context.retrieved_knowledge.results
    ]


def test_character_locations_refer_to_known_places_and_characters():
    names = {c.name for c in create_prototype_state().characters}

    assert set(CHARACTER_LOCATIONS) == names - {PLAYER_CHARACTER}
    assert set(CHARACTER_LOCATIONS.values()) <= set(LOCATIONS)
