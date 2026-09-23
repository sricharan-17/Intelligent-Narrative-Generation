from game.gamestate.interaction.actions import StateChangeAction
from game.gamestate.interaction.models import Interaction
from game.gamestate.state.models import GameState


class ActionValidator:
    """Validates interactions and structured state changes."""

    def validate(
        self,
        interaction: Interaction,
        state: GameState,
    ) -> bool:
        if interaction.input_type != "action":
            return False

        action = interaction.input.strip().lower()

        available_actions = {
            available.strip().lower()
            for available in state.available_actions
        }

        return action in available_actions

    def validate_state_change(
        self,
        action: StateChangeAction,
        state: GameState,
    ) -> bool:
        """Validate a structured state change without mutating state."""

        if action.action_type == "object_state_change":
            if action.target is None:
                return False

            target_object = next(
                (
                    obj
                    for obj in state.objects
                    if obj.name.lower() == action.target.lower()
                ),
                None,
            )

            return target_object is not None

        if action.action_type == "inventory_change":
            return all(
                item in state.inventory
                for item in action.inventory_remove
            )

        return False