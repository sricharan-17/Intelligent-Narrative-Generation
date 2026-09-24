from game.context.models import NarrativeContext
from game.gamestate.interaction.models import Interaction
from game.gamestate.state.models import (
    Character,
    GameState,
    HistoryEntry,
    Location,
)
from rag.knowledge.models import RetrievedKnowledge

from llm.prompt_builder import NarrativePromptBuilder


def test_prompt_builder_contains_full_narrative_context():
    interaction = Interaction(
        input_character="King",
        responder_character="Spider",
        input="What's this? An itsy bitsy spider!",
        input_type="speech",
    )

    state = GameState(
        location=Location(
            name="Secret Tunnel",
            description="A dark tunnel inside the castle.",
        ),
        characters=[
            Character(
                name="King",
                description="I am a brave and fearless king.",
                present=True,
            ),
            Character(
                name="Spider",
                description="I am a large black spider that lives in dark places.",
                present=True,
            ),
        ],
        inventory=["crown", "scepter"],
        available_actions=["hit spider", "hug spider"],
        history=[
            HistoryEntry(
                turn=1,
                speaker="King",
                text="How did you get in here?",
                entry_type="speech",
            )
        ],
        turn=1,
    )

    context = NarrativeContext(
        interaction=interaction,
        game_state=state,
        history=tuple(state.history),
        retrieved_knowledge=RetrievedKnowledge(),
    )

    messages = NarrativePromptBuilder().build_messages(context)

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"

    prompt = messages[1]["content"]

    assert "RESPONDER CHARACTER" in prompt
    assert "Spider" in prompt
    assert "large black spider" in prompt

    assert "INPUT CHARACTER" in prompt
    assert "King" in prompt

    assert "Secret Tunnel" in prompt
    assert "crown" in prompt
    assert "scepter" in prompt
    assert "hit spider" in prompt

    assert "CONVERSATION HISTORY" in prompt
    assert "How did you get in here?" in prompt

    assert "CURRENT INPUT" in prompt
    assert "What's this? An itsy bitsy spider!" in prompt

    assert "INPUT TYPE" in prompt
    assert "speech" in prompt


def test_prompt_builder_handles_empty_history_and_knowledge():
    interaction = Interaction(
        input_character="Hero",
        responder_character="Guard",
        input="Open the gate.",
        input_type="action",
    )

    context = NarrativeContext(
        interaction=interaction,
        game_state=GameState(),
        history=(),
        retrieved_knowledge=RetrievedKnowledge(),
    )

    messages = NarrativePromptBuilder().build_messages(context)

    prompt = messages[1]["content"]

    assert "(none)" in prompt
    assert "Open the gate." in prompt
    assert "action" in prompt
