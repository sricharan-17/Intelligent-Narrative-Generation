from copy import deepcopy

from game.gamestate.interaction.actions import StateChangeAction

from .models import (
    Character,
    GameObject,
    GameState,
    HistoryEntry,
    Location,
    RecentEvent,
)


class GameStateManager:
    """Controls all state mutations for a GameState."""

    def __init__(self, state: GameState | None = None):
        self._state = state if state is not None else GameState()

    def get_state(self) -> GameState:
        """Return a copy of the current state for safe reading."""
        return deepcopy(self._state)

    def add_character(self, character: Character) -> None:
        self._state.characters.append(character)

    def set_character_presence(
        self,
        character_name: str,
        present: bool,
    ) -> bool:
        """Update whether a character is currently present."""

        for character in self._state.characters:
            if character.name.lower() == character_name.lower():
                character.present = present
                return True

        return False

    def add_object(self, game_object: GameObject) -> None:
        self._state.objects.append(game_object)

    def add_inventory_item(self, item: str) -> None:
        if item not in self._state.inventory:
            self._state.inventory.append(item)

    def remove_inventory_item(self, item: str) -> bool:
        if item not in self._state.inventory:
            return False

        self._state.inventory.remove(item)
        return True

    def set_location(self, location: Location) -> None:
        self._state.location = location

    def update_object_state(
        self,
        object_name: str,
        changes: dict,
    ) -> bool:
        """Update an existing object's state."""

        for game_object in self._state.objects:
            if game_object.name == object_name:
                game_object.state.update(changes)
                return True

        return False

    def record_event(
        self,
        event_type: str,
        description: str,
    ) -> None:
        self._state.recent_events.append(
            RecentEvent(
                turn=self._state.turn,
                event_type=event_type,
                description=description,
            )
        )

    def record_history(
        self,
        speaker: str,
        text: str,
        entry_type: str,
    ) -> None:
        self._state.history.append(
            HistoryEntry(
                turn=self._state.turn,
                speaker=speaker,
                text=text,
                entry_type=entry_type,
            )
        )

    def advance_turn(self) -> None:
        self._state.turn += 1

    def apply_state_change(
        self,
        action: StateChangeAction,
    ) -> bool:
        """Apply a previously validated state change atomically."""

        return self.apply_state_changes((action,))

    def apply_state_changes(
        self,
        changes: tuple[StateChangeAction, ...],
    ) -> bool:
        """Apply several state changes atomically as a single turn.

        If any change fails, nothing is committed and the turn is unchanged.
        """

        # Work on a copy first.
        candidate = deepcopy(self._state)

        for action in changes:
            if not self._apply_to(candidate, action):
                return False

        candidate.turn += 1

        # Commit only after every operation succeeded.
        self._state = candidate

        return True

    @staticmethod
    def _apply_to(
        candidate: GameState,
        action: StateChangeAction,
    ) -> bool:
        """Apply one state change to a working copy of the state."""

        if action.action_type == "object_state_change":
            if action.target is None:
                return False

            target_object = next(
                (
                    obj
                    for obj in candidate.objects
                    if obj.name.lower() == action.target.lower()
                ),
                None,
            )

            if target_object is None:
                return False

            target_object.state.update(action.changes)

        elif action.action_type == "inventory_change":
            for item in action.inventory_remove:
                if item not in candidate.inventory:
                    return False

            for item in action.inventory_add:
                if item not in candidate.inventory:
                    candidate.inventory.append(item)

            for item in action.inventory_remove:
                candidate.inventory.remove(item)

        else:
            return False

        candidate.recent_events.append(
            RecentEvent(
                turn=candidate.turn,
                event_type=action.action_type,
                description=(
                    f"Applied state change: "
                    f"{action.target or action.action_type}"
                ),
            )
        )

        return True