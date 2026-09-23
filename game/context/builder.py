from typing import Optional

from game.gamestate.interaction.models import Interaction
from game.gamestate.state.manager import GameStateManager
from rag.knowledge.models import RAGQuery, RetrievedKnowledge
from rag.retrieval.retriever import KnowledgeRetriever

from .models import NarrativeContext


class ContextBuilder:
    """Builds structured narrative-generation context from game state and RAG."""

    def __init__(
        self,
        retriever: KnowledgeRetriever,
        history_limit: Optional[int] = 10,
    ):
        if history_limit is not None and history_limit < 0:
            raise ValueError("history_limit must be non-negative or None")

        self.retriever = retriever
        self.history_limit = history_limit

    def build(
        self,
        interaction: Interaction,
        state_manager: GameStateManager,
    ) -> NarrativeContext:
        """Build a read-only snapshot of context for narrative generation."""

        # GameStateManager returns a deep copy, so this component
        # cannot accidentally modify the authoritative game state.
        state = state_manager.get_state()

        location = ""
        if state.location is not None:
            location = state.location.name

        # Only include characters directly involved in the interaction.
        # The RAG index separately handles location and character boosts.
        relevant_entities = self._collect_relevant_entities(interaction)

        query = RAGQuery(
            query=interaction.input,
            input_type=interaction.input_type,
            location=location,
            input_character=interaction.input_character,
            responder_character=interaction.responder_character,
            relevant_entities=relevant_entities,
        )

        # Use the existing RAG retrieval interface.
        retrieved_knowledge = self.retriever.retrieve(query)

        # Preserve only the requested amount of recent conversation history.
        history = tuple(state.history)

        if self.history_limit is not None:
            history = history[-self.history_limit:]

        return NarrativeContext(
            interaction=interaction,
            game_state=state,
            history=history,
            retrieved_knowledge=retrieved_knowledge,
        )

    @staticmethod
    def _collect_relevant_entities(
        interaction: Interaction,
    ) -> list[str]:
        """Collect entities directly involved in the current interaction."""

        entities: list[str] = []
        seen: set[str] = set()

        for name in (
            interaction.input_character,
            interaction.responder_character,
        ):
            normalized = name.strip()
            key = normalized.lower()

            if normalized and key not in seen:
                seen.add(key)
                entities.append(normalized)

        return entities