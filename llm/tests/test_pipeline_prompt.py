from game.context.builder import ContextBuilder
from game.gamestate.state.manager import GameStateManager
from game.gamestate.state.models import (
    Character,
    GameObject,
    GameState,
    Location,
)
from game.pipeline.pipeline import NarrativePipeline
from rag.indexing.index import KnowledgeIndex
from rag.retrieval.retriever import KnowledgeRetriever

from llm.prompt_builder import (
    NARRATOR_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    NarrativePromptBuilder,
)


class PromptRecordingGenerator:
    """Builds the real LLM prompt, but returns a fixed response instead of
    running the model."""

    def __init__(self, response):
        self.response = response
        self.prompt_builder = NarrativePromptBuilder()
        self.messages = []

    def generate(self, context):
        self.messages.append(self.prompt_builder.build_messages(context))
        return self.response


def build_manager() -> GameStateManager:
    return GameStateManager(
        GameState(
            location=Location("Treasury"),
            characters=[
                Character("Player"),
                Character("Guard", description="A tired castle guard."),
            ],
            objects=[GameObject(name="Chest", state={"locked": True})],
        )
    )


def build_pipeline(generator) -> NarrativePipeline:
    return NarrativePipeline(
        context_builder=ContextBuilder(KnowledgeRetriever(KnowledgeIndex())),
        generator=generator,
    )


def test_world_interaction_prompts_for_narration_end_to_end():
    manager = build_manager()
    generator = PromptRecordingGenerator(
        "The sword clangs against the chest. The chest is unbreakable."
    )

    result = build_pipeline(generator).run(
        manager,
        "I hit the chest with my sword.",
        "action",
        "Player",
    )

    system_message, user_message = generator.messages[0]

    assert system_message["content"] == NARRATOR_SYSTEM_PROMPT
    assert "RESPONDER CHARACTER" not in user_message["content"]

    assert result.success
    assert manager.get_state().history[-1].speaker == "Narrator"


def test_character_interaction_prompts_for_responder_end_to_end():
    manager = build_manager()
    generator = PromptRecordingGenerator("Keep your voice down.")

    result = build_pipeline(generator).run(
        manager,
        "Where is the key?",
        "speech",
        "Player",
        "Guard",
    )

    system_message, user_message = generator.messages[0]

    assert system_message["content"] == SYSTEM_PROMPT
    assert "RESPONDER CHARACTER\nGuard" in user_message["content"]

    assert result.success
    assert result.output == "Keep your voice down."
    assert manager.get_state().history[-1].speaker == "Guard"
