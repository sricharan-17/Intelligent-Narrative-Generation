from game.context.models import NarrativeContext
from game.gamestate.interaction.models import Interaction
from game.gamestate.state.models import (
    Character,
    GameObject,
    GameState,
    HistoryEntry,
    Location,
)
from rag.knowledge.models import RetrievedKnowledge

from llm.prompt_builder import (
    NARRATOR_INPUT_TYPE_INSTRUCTIONS,
    NARRATOR_SYSTEM_PROMPT,
    NARRATOR_TASK,
    SYSTEM_PROMPT,
    NarrativePromptBuilder,
)


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


def build_world_context(responder_character):
    interaction = Interaction(
        input_character="Player",
        responder_character=responder_character,
        input="I hit the chest with my sword.",
        input_type="action",
    )

    state = GameState(
        location=Location(name="Treasury"),
        characters=[
            Character(name="Player", description="A curious adventurer."),
            Character(name="Guard", description="A tired castle guard."),
        ],
        objects=[
            GameObject(
                name="Chest",
                description="An iron-bound chest.",
                state={"locked": True},
            ),
        ],
    )

    return NarrativeContext(
        interaction=interaction,
        game_state=state,
        history=(),
        retrieved_knowledge=RetrievedKnowledge(),
    )


def test_prompt_without_responder_asks_for_narration():
    messages = NarrativePromptBuilder().build_messages(
        build_world_context(None)
    )

    system_prompt = messages[0]["content"]
    prompt = messages[1]["content"]

    assert system_prompt == NARRATOR_SYSTEM_PROMPT
    assert NARRATOR_TASK in prompt
    assert NARRATOR_INPUT_TYPE_INSTRUCTIONS in prompt

    # No nonexistent responder character to impersonate.
    assert "RESPONDER CHARACTER" not in prompt
    assert "Unknown" not in prompt
    assert "responder" not in system_prompt.lower()
    assert "responder" not in prompt.lower()
    assert "LLM" not in prompt

    # The rest of the context is still present.
    assert "INPUT CHARACTER\nPlayer" in prompt
    assert "Treasury" in prompt
    assert "Chest" in prompt
    assert "I hit the chest with my sword." in prompt


def test_prompt_with_responder_is_unchanged_by_narrator_support():
    messages = NarrativePromptBuilder().build_messages(
        build_world_context("Guard")
    )

    system_prompt = messages[0]["content"]
    prompt = messages[1]["content"]

    assert system_prompt == SYSTEM_PROMPT
    assert prompt.startswith(
        "TASK\n"
        "Generate exactly one natural next response from the RESPONDER CHARACTER.\n"
    )
    assert "RESPONDER CHARACTER\nGuard\nPersona/Description: A tired castle guard." in prompt
    assert NARRATOR_TASK not in prompt
