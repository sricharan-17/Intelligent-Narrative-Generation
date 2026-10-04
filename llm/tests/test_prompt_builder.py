from rag.knowledge.models import RetrievedKnowledge
from game.context.models import NarrativeContext
from game.gamestate.state.models import (
    Character,
    GameObject,
    GameState,
    HistoryEntry,
    Location,
)
from llm.prompt_builder import (
    NARRATOR_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    NarrativePromptBuilder,
)


def build_test_context(responder_character="Spider"):
    state = GameState(
        setting={
            "name": "Secret Tunnel",
            "category": "Fantasy",
            "description": "A dark underground tunnel.",
            "background": "An ancient passage beneath the castle.",
        },
        location=Location(
            name="Secret Tunnel",
            description="A dark underground tunnel.",
        ),
        characters=[
            Character(
                name="King",
                description="A powerful and cautious ruler.",
            ),
            Character(
                name="Spider",
                description="A mysterious creature living in the tunnel.",
            ),
        ],
        objects=[
            GameObject(
                name="Crown",
                description="A golden royal crown.",
            ),
            GameObject(
                name="Scepter",
                description="A jeweled royal scepter.",
            ),
        ],
        inventory=["Crown", "Scepter"],
        available_actions=[
            "inspect the tunnel",
            "talk to Spider",
        ],
        history=[
            HistoryEntry(
                turn=1,
                speaker="King",
                text="Who are you?",
                entry_type="speech",
            )
        ],
        turn=1,
    )

    return NarrativeContext(
    interaction=type(
        "Interaction",
        (),
        {
            "input": "Tell me about this place.",
            "input_type": "speech",
            "input_character": "King",
            "responder_character": responder_character,
        },
    )(),
    game_state=state,
    history=tuple(state.history),
    retrieved_knowledge=RetrievedKnowledge(),

    )


def test_prompt_builder_contains_full_narrative_context():
    context = build_test_context()

    builder = NarrativePromptBuilder()
    messages = builder.build_messages(context)

    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"

    prompt = messages[1]["content"]

    assert "### SETTING" in prompt
    assert "Secret Tunnel" in prompt
    assert "Fantasy" in prompt

    assert "### INPUT CHARACTER" in prompt
    assert "Name: King" in prompt

    assert "### RESPONDING CHARACTER" in prompt
    assert "Name: Spider" in prompt

    assert "### WORLD STATE" in prompt
    assert "Crown" in prompt
    assert "Scepter" in prompt
    assert "inspect the tunnel" in prompt

    assert "### HISTORY" in prompt
    assert "Who are you?" in prompt

    assert "### PLAYER INPUT" in prompt
    assert "Tell me about this place." in prompt

    assert "### INPUT TYPE" in prompt
    assert "speech" in prompt

    assert "### RESPONSE" in prompt


def test_empty_history():
    context = build_test_context()

    context = NarrativeContext(
        interaction=context.interaction,
        game_state=context.game_state,
        history=(),
        retrieved_knowledge=context.retrieved_knowledge,
    )

    builder = NarrativePromptBuilder()
    messages = builder.build_messages(context)

    prompt = messages[1]["content"]

    assert "### HISTORY" in prompt
    assert "None" in prompt
    assert "### PLAYER INPUT" in prompt
    assert "Tell me about this place." in prompt


def test_prompt_without_responder_asks_for_narration():
    context = build_test_context(responder_character=None)

    builder = NarrativePromptBuilder()
    messages = builder.build_messages(context)

    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == NARRATOR_SYSTEM_PROMPT

    prompt = messages[1]["content"]

    assert "### RESPONDING CHARACTER" not in prompt
    assert "### INPUT CHARACTER" in prompt
    assert "Name: King" in prompt
    assert "### WORLD STATE" in prompt
    assert "### PLAYER INPUT" in prompt
    assert "### RESPONSE" in prompt


def test_prompt_with_responder_uses_character_system_prompt():
    context = build_test_context()

    builder = NarrativePromptBuilder()
    messages = builder.build_messages(context)

    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == SYSTEM_PROMPT

    prompt = messages[1]["content"]

    assert "### RESPONDING CHARACTER" in prompt
    assert "Name: Spider" in prompt