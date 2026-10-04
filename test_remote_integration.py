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
from llm.generator import NarrativeGenerator


def build_manager() -> GameStateManager:
    return GameStateManager(
        GameState(
            location=Location("Dark Forest"),
            characters=[
                Character("Player"),
                Character(
                    "Forest Guardian",
                    description="An ancient guardian who protects the forest.",
                ),
            ],
            objects=[
                GameObject(
                    name="Abandoned Tower",
                    state={"condition": "old and mysterious"},
                ),
            ],
        )
    )


def main():
    manager = build_manager()

    generator = NarrativeGenerator(
        max_new_tokens=128,
        temperature=0.7,
    )
    generator.load()

    pipeline = NarrativePipeline(
        context_builder=ContextBuilder(
            KnowledgeRetriever(KnowledgeIndex())
        ),
        generator=generator,
    )

    result = pipeline.run(
        manager,
        "The player walks toward the abandoned tower.",
        "action",
        "Player",
    )

    print("\n=== PIPELINE RESULT ===")
    print("Success:", result.success)
    print("\n=== GENERATED NARRATIVE ===")
    print(result.output)

    print("\n=== FINAL GAME HISTORY ===")
    for event in manager.get_state().history:
        print(f"{event.speaker}: {event.content}")


if __name__ == "__main__":
    main()