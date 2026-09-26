import subprocess
import sys
from pathlib import Path

import pytest

from game.context.builder import ContextBuilder
from game.context.models import NarrativeContext
from game.gamestate.interaction.actions import StateChangeAction
from game.gamestate.state.manager import GameStateManager
from game.gamestate.state.models import (
    Character,
    GameObject,
    GameState,
    HistoryEntry,
    Location,
)
from game.output.narrative import NarrativeOutput
from game.pipeline.pipeline import NarrativePipeline, PipelineResult
from game.response.validator import ResponseValidator
from rag.indexing.index import KnowledgeIndex
from rag.knowledge.models import KnowledgeEntry
from rag.retrieval.retriever import KnowledgeRetriever


PROJECT_ROOT = Path(__file__).resolve().parents[3]

START_TURN = 3

OPEN_CHEST = StateChangeAction(
    action_type="object_state_change",
    target="Old Chest",
    changes={"open": True},
)
USE_KEY = StateChangeAction(
    action_type="inventory_change",
    inventory_remove=("Key",),
)
MISSING_ITEM = StateChangeAction(
    action_type="inventory_change",
    inventory_remove=("Shield",),
)


class FakeNarrativeGenerator:
    """Stands in for llm.NarrativeGenerator without loading a model."""

    def __init__(self, response):
        self.response = response
        self.contexts: list[NarrativeContext] = []

    def generate(self, context):
        self.contexts.append(context)
        return self.response


class FailingGenerator:
    def __init__(self):
        self.calls = 0

    def generate(self, context):
        self.calls += 1
        raise RuntimeError("model crashed")


class FailingRetriever:
    def retrieve(self, query):
        raise RuntimeError("index unavailable")


def build_test_manager() -> GameStateManager:
    state = GameState(
        setting={"name": "Fantasy"},
        location=Location("Gatehouse"),
        characters=[
            Character("Player"),
            Character("Guard", description="A tired castle guard."),
        ],
        objects=[
            GameObject(name="Old Chest", state={"open": False}),
        ],
        inventory=["Key"],
        history=[
            HistoryEntry(2, "Player", "Good evening.", "speech"),
            HistoryEntry(2, "Guard", "Evening. State your business.", "speech"),
        ],
        turn=START_TURN,
    )

    return GameStateManager(state)


def build_retriever() -> KnowledgeRetriever:
    index = KnowledgeIndex()
    index.add_documents(
        [
            KnowledgeEntry(
                document_id="char_guard",
                source_type="character",
                entity_id="Guard",
                title="Character Persona: Guard",
                content="The guard keeps the gatehouse key hidden under a loose stone.",
            ),
            KnowledgeEntry(
                document_id="loc_gatehouse",
                source_type="location",
                entity_id="Gatehouse",
                title="Location: Gatehouse",
                content="The gatehouse guards the only road into the castle.",
            ),
        ]
    )

    return KnowledgeRetriever(index, min_score=0.1, default_top_k=3)


def build_pipeline(generator, retriever=None) -> NarrativePipeline:
    return NarrativePipeline(
        context_builder=ContextBuilder(
            retriever if retriever is not None else build_retriever()
        ),
        generator=generator,
    )


# --------------------------------------------------
# Successful interactions
# --------------------------------------------------


def test_successful_speech_interaction():
    manager = build_test_manager()
    generator = FakeNarrativeGenerator("  Under the loose stone, stranger.  ")

    result = build_pipeline(generator).run(
        manager,
        "  Where is   the key? ",
        "speech",
        "Player",
        "Guard",
    )

    assert isinstance(result, PipelineResult)
    assert result.success
    assert result.interaction.input == "Where is the key?"
    assert result.context.interaction == result.interaction
    assert result.raw_response == "  Under the loose stone, stranger.  "
    assert result.validation.is_valid
    assert result.state_update.success
    assert result.state_update.applied_changes == ()
    assert result.output == "Under the loose stone, stranger."

    state = manager.get_state()

    assert state.turn == START_TURN + 1
    assert state.history[-2:] == [
        HistoryEntry(START_TURN, "Player", "Where is the key?", "speech"),
        HistoryEntry(
            START_TURN,
            "Guard",
            "  Under the loose stone, stranger.  ",
            "speech",
        ),
    ]


def test_successful_action_interaction():
    manager = build_test_manager()
    generator = FakeNarrativeGenerator("Hands off me, you fool!")

    result = build_pipeline(generator).run(
        manager,
        "hug guard",
        "action",
        "Player",
        "Guard",
    )

    assert result.success
    assert result.output == "Hands off me, you fool!"
    assert result.context.interaction.input_type == "action"

    state = manager.get_state()

    assert state.history[-2:] == [
        HistoryEntry(START_TURN, "Player", "hug guard", "action"),
        HistoryEntry(START_TURN, "Guard", "Hands off me, you fool!", "action"),
    ]
    assert state.turn == START_TURN + 1


def test_interaction_without_responder_uses_narrator():
    manager = build_test_manager()
    generator = FakeNarrativeGenerator(
        "The sword clangs against the chest. The chest is unbreakable."
    )

    result = build_pipeline(generator).run(
        manager,
        "I hit the chest with my sword.",
        "action",
        "Player",
    )

    assert result.success
    assert result.interaction.responder_character is None
    assert generator.contexts[0].interaction.responder_character is None

    speakers = [entry.speaker for entry in manager.get_state().history[-2:]]

    assert speakers == ["Player", "Narrator"]
    assert "LLM" not in speakers


def test_valid_explicit_state_changes_are_applied():
    manager = build_test_manager()
    generator = FakeNarrativeGenerator("The key turns and the chest creaks open.")

    result = build_pipeline(generator).run(
        manager,
        "Unlock the chest with the key.",
        "action",
        "Player",
        state_changes=(USE_KEY, OPEN_CHEST),
    )

    assert result.success
    assert result.state_update.applied_changes == (USE_KEY, OPEN_CHEST)
    assert result.state_update.rejected_changes == ()

    state = manager.get_state()

    assert state.inventory == []
    assert state.objects[0].state["open"] is True
    assert state.turn == START_TURN + 1
    assert [event.turn for event in state.recent_events] == [
        START_TURN,
        START_TURN,
    ]
    assert [entry.turn for entry in state.history[-2:]] == [
        START_TURN,
        START_TURN,
    ]


def test_consecutive_interactions_advance_one_turn_each():
    manager = build_test_manager()
    pipeline = build_pipeline(FakeNarrativeGenerator("Move along."))

    pipeline.run(manager, "Hello.", "speech", "Player", "Guard")
    pipeline.run(manager, "Can I pass?", "speech", "Player", "Guard")

    state = manager.get_state()

    assert state.turn == START_TURN + 2
    assert [entry.turn for entry in state.history[-4:]] == [
        START_TURN,
        START_TURN,
        START_TURN + 1,
        START_TURN + 1,
    ]


# --------------------------------------------------
# Context reaching the generator
# --------------------------------------------------


def test_rag_knowledge_reaches_generator_context():
    generator = FakeNarrativeGenerator("Under the loose stone.")

    build_pipeline(generator).run(
        build_test_manager(),
        "Where is the key?",
        "speech",
        "Player",
        "Guard",
    )

    knowledge = generator.contexts[0].retrieved_knowledge
    document_ids = [result.document_id for result in knowledge.results]

    assert "char_guard" in document_ids
    assert any("loose stone" in result.content for result in knowledge.results)


def test_conversation_history_reaches_generator_context():
    manager = build_test_manager()
    generator = FakeNarrativeGenerator("Move along.")
    pipeline = build_pipeline(generator)

    pipeline.run(manager, "Hello.", "speech", "Player", "Guard")
    pipeline.run(manager, "Can I pass?", "speech", "Player", "Guard")

    first_history = generator.contexts[0].history
    second_history = generator.contexts[1].history

    assert [entry.text for entry in first_history] == [
        "Good evening.",
        "Evening. State your business.",
    ]
    assert [entry.text for entry in second_history][-2:] == [
        "Hello.",
        "Move along.",
    ]


def test_generator_sees_state_before_the_update():
    manager = build_test_manager()
    generator = FakeNarrativeGenerator("The chest creaks open.")

    build_pipeline(generator).run(
        manager,
        "Open the chest.",
        "action",
        "Player",
        state_changes=(OPEN_CHEST,),
    )

    context = generator.contexts[0]

    assert context.game_state.turn == START_TURN
    assert context.game_state.objects[0].state["open"] is False
    assert all(entry.text != "Open the chest." for entry in context.history)


# --------------------------------------------------
# Failures
# --------------------------------------------------


@pytest.mark.parametrize(
    "response",
    [
        "",
        "   ",
        "### RESPONSE\nMove along.",
        "Guard: Move along.",
        "where is the key?",
        "a" * 1001,
    ],
    ids=[
        "empty",
        "whitespace",
        "prompt_leakage",
        "speaker_prefix",
        "input_echo",
        "too_long",
    ],
)
def test_invalid_generated_response_leaves_state_unchanged(response):
    manager = build_test_manager()
    before = manager.get_state()

    result = build_pipeline(FakeNarrativeGenerator(response)).run(
        manager,
        "Where is the key?",
        "speech",
        "Player",
        "Guard",
        state_changes=(OPEN_CHEST,),
    )

    assert not result.success
    assert not result.validation.is_valid
    assert result.validation.errors
    assert result.raw_response == response
    assert result.state_update is None
    assert result.output is None
    assert manager.get_state() == before


def test_non_string_generated_response_is_rejected():
    manager = build_test_manager()
    before = manager.get_state()

    result = build_pipeline(FakeNarrativeGenerator(None)).run(
        manager,
        "Where is the key?",
        "speech",
        "Player",
        "Guard",
    )

    assert not result.success
    assert result.validation.response == ""
    assert result.output is None
    assert manager.get_state() == before


def test_invalid_explicit_state_change_leaves_state_unchanged():
    manager = build_test_manager()
    before = manager.get_state()

    result = build_pipeline(
        FakeNarrativeGenerator("You open the chest and hand over a shield.")
    ).run(
        manager,
        "Open the chest and give the guard my shield.",
        "action",
        "Player",
        "Guard",
        state_changes=(OPEN_CHEST, MISSING_ITEM),
    )

    assert not result.success
    assert result.validation.is_valid
    assert result.state_update.success is False
    assert result.state_update.applied_changes == ()
    assert result.state_update.rejected_changes == (OPEN_CHEST, MISSING_ITEM)
    assert result.output is None

    after = manager.get_state()

    assert after == before
    assert after.objects[0].state["open"] is False


def test_generator_failure_leaves_state_unchanged():
    manager = build_test_manager()
    before = manager.get_state()
    generator = FailingGenerator()

    with pytest.raises(RuntimeError, match="model crashed"):
        build_pipeline(generator).run(
            manager,
            "Where is the key?",
            "speech",
            "Player",
            "Guard",
            state_changes=(OPEN_CHEST,),
        )

    assert generator.calls == 1
    assert manager.get_state() == before


def test_invalid_interaction_is_rejected_before_generation():
    manager = build_test_manager()
    before = manager.get_state()
    generator = FakeNarrativeGenerator("Move along.")
    pipeline = build_pipeline(generator)

    with pytest.raises(ValueError, match="input_text cannot be empty"):
        pipeline.run(manager, "   ", "speech", "Player", "Guard")

    with pytest.raises(ValueError, match="Unsupported input_type"):
        pipeline.run(manager, "Hello.", "shout", "Player", "Guard")

    with pytest.raises(ValueError, match="input_character cannot be empty"):
        pipeline.run(manager, "Hello.", "speech", "  ", "Guard")

    assert generator.contexts == []
    assert manager.get_state() == before


def test_context_building_failure_leaves_state_unchanged():
    manager = build_test_manager()
    before = manager.get_state()
    generator = FakeNarrativeGenerator("Move along.")

    with pytest.raises(RuntimeError, match="index unavailable"):
        build_pipeline(generator, retriever=FailingRetriever()).run(
            manager,
            "Where is the key?",
            "speech",
            "Player",
            "Guard",
        )

    assert generator.contexts == []
    assert manager.get_state() == before


# --------------------------------------------------
# Composition
# --------------------------------------------------


def test_injected_components_are_used():
    class StrictValidator(ResponseValidator):
        def validate(self, response, context):
            result = super().validate(response, context)
            return type(result)(False, result.response, ("strict: rejected",))

    class ShoutingOutput(NarrativeOutput):
        def format(self, response):
            return super().format(response).upper()

    manager = build_test_manager()
    before = manager.get_state()

    rejected = NarrativePipeline(
        context_builder=ContextBuilder(build_retriever()),
        generator=FakeNarrativeGenerator("Move along."),
        response_validator=StrictValidator(),
    ).run(manager, "Hello.", "speech", "Player", "Guard")

    assert rejected.validation.errors == ("strict: rejected",)
    assert manager.get_state() == before

    accepted = NarrativePipeline(
        context_builder=ContextBuilder(build_retriever()),
        generator=FakeNarrativeGenerator("Move along."),
        output=ShoutingOutput(),
    ).run(manager, "Hello.", "speech", "Player", "Guard")

    assert accepted.output == "MOVE ALONG."


def test_pipeline_does_not_import_the_llm_package():
    # Importing the pipeline must not load torch/transformers, so the
    # pipeline stays usable and testable without the fine-tuned model.
    code = (
        "import sys, game.pipeline; "
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
