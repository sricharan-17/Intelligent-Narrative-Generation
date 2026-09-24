from game.context.builder import ContextBuilder
from game.context.models import NarrativeContext
from game.gamestate.interaction.models import Interaction
from game.gamestate.state.manager import GameStateManager
from game.gamestate.state.models import (
    Character,
    GameObject,
    GameState,
    HistoryEntry,
    Location,
)
from rag.knowledge.models import KnowledgeResult, RetrievedKnowledge


class FakeRetriever:
    def __init__(self):
        self.queries = []

    def retrieve(self, query):
        self.queries.append(query)
        return RetrievedKnowledge(
            results=[
                KnowledgeResult(
                    document_id="doc-1",
                    content="The castle was built centuries ago.",
                    source_type="world_lore",
                    entity_id="castle",
                    title="Castle History",
                    score=0.9,
                )
            ]
        )


def test_builds_speech_context():
    state = GameState(
        location=Location("Castle"),
        characters=[
            Character("Alice"),
            Character("Bob"),
        ],
    )
    manager = GameStateManager(state)
    retriever = FakeRetriever()

    interaction = Interaction(
        input="Tell me about this place",
        input_type="speech",
        input_character="Alice",
        responder_character="Bob",
    )

    builder = ContextBuilder(retriever)
    context = builder.build(interaction, manager)

    assert isinstance(context, NarrativeContext)
    assert context.interaction == interaction
    assert context.game_state.location.name == "Castle"
    assert len(context.retrieved_knowledge.results) == 1

    query = retriever.queries[0]
    assert query.query == "Tell me about this place"
    assert query.input_type == "speech"
    assert query.location == "Castle"
    assert query.input_character == "Alice"
    assert query.responder_character == "Bob"


def test_builds_action_context():
    state = GameState(
        location=Location("Armory"),
        characters=[Character("Player")],
        objects=[GameObject("Sword")],
    )
    manager = GameStateManager(state)
    retriever = FakeRetriever()

    interaction = Interaction(
        input="take sword",
        input_type="action",
        input_character="Player",
        responder_character="Guard",
    )

    context = ContextBuilder(retriever).build(interaction, manager)

    query = retriever.queries[0]

    assert query.input_type == "action"
    assert query.location == "Armory"
    assert query.relevant_entities == ["Player", "Guard"]


def test_includes_recent_history_with_limit():
    state = GameState(
        history=[
            HistoryEntry(1, "Alice", "First", "speech"),
            HistoryEntry(2, "Bob", "Second", "speech"),
            HistoryEntry(3, "Alice", "Third", "speech"),
        ]
    )
    manager = GameStateManager(state)
    retriever = FakeRetriever()

    interaction = Interaction(
        input="Continue",
        input_type="speech",
        input_character="Alice",
        responder_character="Bob",
    )

    context = ContextBuilder(
        retriever,
        history_limit=2,
    ).build(interaction, manager)

    assert len(context.history) == 2
    assert context.history[0].text == "Second"
    assert context.history[1].text == "Third"


def test_history_limit_none_keeps_all_history():
    state = GameState(
        history=[
            HistoryEntry(1, "Alice", "First", "speech"),
            HistoryEntry(2, "Bob", "Second", "speech"),
        ]
    )
    manager = GameStateManager(state)
    retriever = FakeRetriever()

    interaction = Interaction(
        input="Continue",
        input_type="speech",
        input_character="Alice",
        responder_character="Bob",
    )

    context = ContextBuilder(
        retriever,
        history_limit=None,
    ).build(interaction, manager)

    assert len(context.history) == 2


def test_context_builder_does_not_modify_game_state():
    state = GameState(
        location=Location("Village"),
        characters=[Character("Alice")],
        objects=[GameObject("Key")],
        inventory=["map"],
        turn=5,
    )
    manager = GameStateManager(state)
    retriever = FakeRetriever()

    interaction = Interaction(
        input="look around",
        input_type="action",
        input_character="Alice",
        responder_character="Guard",
    )

    ContextBuilder(retriever).build(interaction, manager)

    after = manager.get_state()

    assert after.location.name == "Village"
    assert [c.name for c in after.characters] == ["Alice"]
    assert [o.name for o in after.objects] == ["Key"]
    assert after.inventory == ["map"]
    assert after.turn == 5


def test_duplicate_entities_are_removed_case_insensitively():
    state = GameState(
        characters=[
            Character("Alice"),
            Character("alice"),
        ],
        objects=[
            GameObject("Key"),
            GameObject("KEY"),
        ],
    )
    manager = GameStateManager(state)
    retriever = FakeRetriever()

    interaction = Interaction(
        input="look",
        input_type="action",
        input_character="ALICE",
        responder_character="Guard",
    )

    ContextBuilder(retriever).build(interaction, manager)

    entities = retriever.queries[0].relevant_entities

    assert entities == ["ALICE", "Guard"]


def test_empty_location_is_supported():
    state = GameState()
    manager = GameStateManager(state)
    retriever = FakeRetriever()

    interaction = Interaction(
        input="hello",
        input_type="speech",
        input_character="Player",
        responder_character="NPC",
    )

    context = ContextBuilder(retriever).build(interaction, manager)

    assert retriever.queries[0].location == ""
    assert context.game_state.location is None


def test_negative_history_limit_is_rejected():
    retriever = FakeRetriever()

    try:
        ContextBuilder(retriever, history_limit=-1)
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_relevant_entities_without_responder():
    state = GameState(
        location=Location("Treasury"),
        characters=[Character("Player")],
        objects=[GameObject("Chest")],
    )
    manager = GameStateManager(state)
    retriever = FakeRetriever()

    interaction = Interaction(
        input="I hit the chest with my sword.",
        input_type="action",
        input_character="Player",
        responder_character=None,
    )

    context = ContextBuilder(retriever).build(interaction, manager)

    query = retriever.queries[0]

    assert query.relevant_entities == ["Player"]
    assert None not in query.relevant_entities
    assert "None" not in query.relevant_entities
    assert query.input_character == "Player"
    assert query.responder_character == ""
    assert context.interaction.responder_character is None


def test_relevant_entities_with_responder():
    state = GameState(
        location=Location("Gatehouse"),
        characters=[Character("Player"), Character("Guard")],
    )
    manager = GameStateManager(state)
    retriever = FakeRetriever()

    interaction = Interaction(
        input="Where is the key?",
        input_type="speech",
        input_character="Player",
        responder_character="Guard",
    )

    context = ContextBuilder(retriever).build(interaction, manager)

    query = retriever.queries[0]

    assert query.relevant_entities == ["Player", "Guard"]
    assert query.responder_character == "Guard"
    assert context.interaction.responder_character == "Guard"
