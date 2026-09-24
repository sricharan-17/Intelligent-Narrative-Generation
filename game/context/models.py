from dataclasses import dataclass

from game.gamestate.interaction.models import Interaction
from game.gamestate.state.models import GameState, HistoryEntry
from rag.knowledge.models import RetrievedKnowledge


@dataclass(frozen=True)
class NarrativeContext:
    """Structured context prepared for the narrative generation stage."""

    interaction: Interaction
    game_state: GameState
    history: tuple[HistoryEntry, ...]
    retrieved_knowledge: RetrievedKnowledge
