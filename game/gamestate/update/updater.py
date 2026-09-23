from dataclasses import dataclass

from game.gamestate.interaction.actions import StateChangeAction
from game.gamestate.interaction.models import Interaction
from game.gamestate.state.manager import GameStateManager
from game.gamestate.validation.validator import ActionValidator


# History speaker for world/object responses, where no character responds.
# The LLM generates the text but is never recorded as a speaker.
NARRATOR_SPEAKER = "Narrator"


@dataclass(frozen=True)
class StateUpdateResult:
    """Outcome of committing one interaction to the game state."""

    applied_changes: tuple[StateChangeAction, ...]
    rejected_changes: tuple[StateChangeAction, ...]
    success: bool


class StateUpdater:
    """Commits one interaction and its explicit state changes as one turn.

    This component does not:
    - parse or infer state changes from the response text,
    - validate the response text (see ResponseValidator),
    - call RAG or the LLM.
    """

    def __init__(self, validator: ActionValidator | None = None):
        self.validator = (
            validator if validator is not None else ActionValidator()
        )

    def apply(
        self,
        interaction: Interaction,
        response: str,
        state_manager: GameStateManager,
        state_changes: tuple[StateChangeAction, ...] = (),
    ) -> StateUpdateResult:
        """Record the interaction and apply its state changes atomically.

        On failure nothing is recorded, no change is applied, and the
        turn is unchanged. On success the player input and response are
        recorded at the current turn T and the state advances to T + 1.
        """

        changes = tuple(state_changes)

        # The whole batch is rejected if any change cannot be applied.
        if not self._can_apply(changes, state_manager):
            return StateUpdateResult(
                applied_changes=(),
                rejected_changes=changes,
                success=False,
            )

        # Both history entries are stamped with the current turn T.
        state_manager.record_history(
            speaker=interaction.input_character,
            text=interaction.input,
            entry_type=interaction.input_type,
        )
        state_manager.record_history(
            speaker=self._response_speaker(interaction),
            text=response,
            entry_type=interaction.input_type,
        )

        # Applies every change and advances to turn T + 1 exactly once.
        if not state_manager.apply_state_changes(changes):
            # Unreachable: the same changes succeeded on an identical copy.
            raise RuntimeError(
                "state changes failed after passing validation"
            )

        return StateUpdateResult(
            applied_changes=changes,
            rejected_changes=(),
            success=True,
        )

    def _can_apply(
        self,
        changes: tuple[StateChangeAction, ...],
        state_manager: GameStateManager,
    ) -> bool:
        """Validate and apply the changes in order on a scratch copy."""

        # Each change is checked against the state left by the previous
        # ones, so e.g. removing the same item twice is rejected.
        scratch = GameStateManager(state_manager.get_state())

        for change in changes:
            if not self.validator.validate_state_change(
                change,
                scratch.get_state(),
            ):
                return False

            if not scratch.apply_state_change(change):
                return False

        return True

    @staticmethod
    def _response_speaker(interaction: Interaction) -> str:
        if interaction.responder_character is None:
            return NARRATOR_SPEAKER

        return interaction.responder_character
