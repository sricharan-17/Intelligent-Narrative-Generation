from game.gamestate.interaction.actions import StateChangeAction
from copy import deepcopy

from game.gamestate.interaction.models import Interaction
from game.gamestate.state.manager import GameStateManager
from game.gamestate.state.models import (
    Character,
    GameObject,
    GameState,
    Location,
)
from game.gamestate.validation.validator import ActionValidator


def build_test_manager() -> GameStateManager:
    state = GameState(
        setting={"name": "Fantasy"},
        location=Location("Old Castle"),
        characters=[
            Character("Wizard"),
            Character("Guard", present=False),
        ],
        objects=[
            GameObject(
                name="Old Chest",
                state={"open": False},
                location="Old Castle",
            ),
            GameObject(
                name="Sword",
                state={"visible": True},
                location="Old Castle",
            ),
        ],
        inventory=["Key"],
        available_actions=[
            "open chest",
            "pick up sword",
            "talk to wizard",
        ],
    )

    return GameStateManager(state)


def test_valid_action():
    manager = build_test_manager()
    validator = ActionValidator()

    interaction = Interaction(
        input="open chest",
        input_type="action",
        input_character="player",
        responder_character="wizard",
    )

    assert validator.validate(
        interaction,
        manager.get_state(),
    )


def test_invalid_action():
    manager = build_test_manager()
    validator = ActionValidator()

    interaction = Interaction(
        input="fly to the moon",
        input_type="action",
        input_character="player",
        responder_character="wizard",
    )

    assert not validator.validate(
        interaction,
        manager.get_state(),
    )


def test_inventory_change():
    manager = build_test_manager()

    manager.add_inventory_item("Torch")

    state = manager.get_state()

    assert "Torch" in state.inventory

    assert manager.remove_inventory_item("Torch")

    state = manager.get_state()

    assert "Torch" not in state.inventory


def test_object_state_change():
    manager = build_test_manager()

    changed = manager.update_object_state(
        "Old Chest",
        {"open": True},
    )

    assert changed

    state = manager.get_state()

    chest = next(
        obj for obj in state.objects
        if obj.name == "Old Chest"
    )

    assert chest.state["open"] is True


def test_location_change():
    manager = build_test_manager()

    manager.set_location(
        Location("Wizard's Tower")
    )

    state = manager.get_state()

    assert state.location is not None
    assert state.location.name == "Wizard's Tower"


def test_character_presence():
    manager = build_test_manager()

    state = manager.get_state()

    wizard = next(
        character
        for character in state.characters
        if character.name == "Wizard"
    )

    guard = next(
        character
        for character in state.characters
        if character.name == "Guard"
    )

    assert wizard.present is True
    assert guard.present is False


def test_history_recording():
    manager = build_test_manager()

    manager.record_history(
        speaker="player",
        text="Open the old chest.",
        entry_type="action",
    )

    state = manager.get_state()

    assert len(state.history) == 1
    assert state.history[0].speaker == "player"
    assert state.history[0].text == "Open the old chest."
    assert state.history[0].entry_type == "action"


def test_invalid_action_does_not_modify_state():
    manager = build_test_manager()
    validator = ActionValidator()

    before = manager.get_state()

    interaction = Interaction(
        input="fly to the moon",
        input_type="action",
        input_character="player",
        responder_character="wizard",
    )

    is_valid = validator.validate(
        interaction,
        before,
    )

    assert not is_valid

    after = manager.get_state()

    assert after == before

def test_valid_state_change_updates_game_state():
    manager = build_test_manager()
    validator = ActionValidator()

    action = StateChangeAction(
        action_type="object_state_change",
        target="Old Chest",
        changes={"open": True},
    )

    assert validator.validate_state_change(
        action,
        manager.get_state(),
    )

    assert manager.apply_state_change(action)

    state = manager.get_state()

    chest = next(
        obj for obj in state.objects
        if obj.name == "Old Chest"
    )

    assert chest.state["open"] is True
    assert state.turn == 1
    assert len(state.recent_events) == 1

def test_invalid_state_change_does_not_modify_game_state():
    manager = build_test_manager()
    validator = ActionValidator()

    before = manager.get_state()

    action = StateChangeAction(
        action_type="object_state_change",
        target="Missing Chest",
        changes={"open": True},
    )

    assert not validator.validate_state_change(
        action,
        before,
    )

    after = manager.get_state()

    assert after == before

def test_valid_inventory_state_change():
    manager = build_test_manager()
    validator = ActionValidator()

    action = StateChangeAction(
        action_type="inventory_change",
        inventory_add=("Torch",),
    )

    assert validator.validate_state_change(
        action,
        manager.get_state(),
    )

    assert manager.apply_state_change(action)

    state = manager.get_state()

    assert "Torch" in state.inventory
    assert state.turn == 1
    assert len(state.recent_events) == 1


def test_inventory_removal_state_change():
    manager = build_test_manager()
    validator = ActionValidator()

    action = StateChangeAction(
        action_type="inventory_change",
        inventory_remove=("Key",),
    )

    assert validator.validate_state_change(
        action,
        manager.get_state(),
    )

    assert manager.apply_state_change(action)

    state = manager.get_state()

    assert "Key" not in state.inventory
    assert state.turn == 1

def test_removing_nonexistent_inventory_item_is_rejected_and_state_unchanged():
    manager = build_test_manager()
    validator = ActionValidator()

    before = manager.get_state()

    action = StateChangeAction(
        action_type="inventory_change",
        inventory_remove=("Shield",),
    )

    assert not validator.validate_state_change(
        action,
        before,
    )

    assert not manager.apply_state_change(action)

    after = manager.get_state()

    assert after == before


def test_multi_item_inventory_change_is_atomic():
    manager = build_test_manager()
    validator = ActionValidator()

    before = manager.get_state()

    action = StateChangeAction(
        action_type="inventory_change",
        inventory_add=("Torch",),
        inventory_remove=("Key", "Shield"),
    )

    assert not validator.validate_state_change(
        action,
        before,
    )

    assert not manager.apply_state_change(action)

    after = manager.get_state()

    assert after == before
    assert "Torch" not in after.inventory
    assert "Key" in after.inventory

def test_object_state_change_is_case_insensitive():
    manager = build_test_manager()
    validator = ActionValidator()

    action = StateChangeAction(
        action_type="object_state_change",
        target="old chest",
        changes={"open": True},
    )

    assert validator.validate_state_change(
        action,
        manager.get_state(),
    )

    assert manager.apply_state_change(action)

    state = manager.get_state()

    chest = next(
        obj for obj in state.objects
        if obj.name == "Old Chest"
    )

    assert chest.state["open"] is True

def test_character_presence_can_be_updated():
    manager = build_test_manager()

    assert manager.set_character_presence(
        "Guard",
        True,
    )

    state = manager.get_state()

    guard = next(
        character
        for character in state.characters
        if character.name == "Guard"
    )

    assert guard.present is True


def test_character_presence_update_is_case_insensitive():
    manager = build_test_manager()

    assert manager.set_character_presence(
        "guard",
        True,
    )

    state = manager.get_state()

    guard = next(
        character
        for character in state.characters
        if character.name == "Guard"
    )

    assert guard.present is True