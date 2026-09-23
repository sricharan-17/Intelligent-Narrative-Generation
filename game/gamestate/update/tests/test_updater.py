from copy import deepcopy

import pytest

from game.gamestate.interaction.actions import StateChangeAction
from game.gamestate.interaction.models import Interaction
from game.gamestate.state.manager import GameStateManager
from game.gamestate.state.models import (
    Character,
    GameObject,
    GameState,
    HistoryEntry,
    Location,
)
from game.gamestate.update.updater import (
    NARRATOR_SPEAKER,
    StateUpdateResult,
    StateUpdater,
)
from game.gamestate.validation.validator import ActionValidator


START_TURN = 5

OPEN_CHEST = StateChangeAction(
    action_type="object_state_change",
    target="Old Chest",
    changes={"open": True},
)
TAKE_TORCH = StateChangeAction(
    action_type="inventory_change",
    inventory_add=("Torch",),
)
USE_KEY = StateChangeAction(
    action_type="inventory_change",
    inventory_remove=("Key",),
)
MISSING_OBJECT = StateChangeAction(
    action_type="object_state_change",
    target="Missing Chest",
    changes={"open": True},
)
MISSING_ITEM = StateChangeAction(
    action_type="inventory_change",
    inventory_remove=("Shield",),
)


def build_test_manager() -> GameStateManager:
    state = GameState(
        setting={"name": "Fantasy"},
        location=Location("Old Castle"),
        characters=[
            Character("Wizard"),
            Character("Guard"),
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
        ],
        turn=START_TURN,
    )

    return GameStateManager(state)


def speech_interaction() -> Interaction:
    return Interaction(
        input="Hello, guard.",
        input_type="speech",
        input_character="Player",
        responder_character="Guard",
    )


def world_interaction() -> Interaction:
    return Interaction(
        input="Open the chest.",
        input_type="action",
        input_character="Player",
        responder_character=None,
    )


def find_object(state: GameState, name: str) -> GameObject:
    return next(obj for obj in state.objects if obj.name == name)


@pytest.fixture
def updater():
    return StateUpdater()


# --------------------------------------------------
# Successful interactions
# --------------------------------------------------


def test_update_without_state_changes(updater):
    manager = build_test_manager()
    before = manager.get_state()

    result = updater.apply(
        speech_interaction(),
        "Who are you?",
        manager,
    )

    assert result == StateUpdateResult(
        applied_changes=(),
        rejected_changes=(),
        success=True,
    )

    after = manager.get_state()

    assert after.turn == START_TURN + 1
    assert after.objects == before.objects
    assert after.inventory == before.inventory
    assert after.recent_events == before.recent_events
    assert after.history == [
        HistoryEntry(START_TURN, "Player", "Hello, guard.", "speech"),
        HistoryEntry(START_TURN, "Guard", "Who are you?", "speech"),
    ]


def test_explicit_empty_state_change_tuple(updater):
    manager = build_test_manager()

    result = updater.apply(
        speech_interaction(),
        "Who are you?",
        manager,
        state_changes=(),
    )

    assert result.success
    assert result.applied_changes == ()
    assert result.rejected_changes == ()
    assert manager.get_state().turn == START_TURN + 1


def test_update_with_one_state_change(updater):
    manager = build_test_manager()

    result = updater.apply(
        world_interaction(),
        "The chest creaks open.",
        manager,
        state_changes=(OPEN_CHEST,),
    )

    assert result == StateUpdateResult(
        applied_changes=(OPEN_CHEST,),
        rejected_changes=(),
        success=True,
    )

    state = manager.get_state()

    assert find_object(state, "Old Chest").state["open"] is True
    assert state.turn == START_TURN + 1


def test_update_with_multiple_state_changes(updater):
    manager = build_test_manager()
    changes = (OPEN_CHEST, TAKE_TORCH, USE_KEY)

    result = updater.apply(
        world_interaction(),
        "You unlock the chest and take a torch.",
        manager,
        state_changes=changes,
    )

    assert result.success
    assert result.applied_changes == changes
    assert result.rejected_changes == ()

    state = manager.get_state()

    assert find_object(state, "Old Chest").state["open"] is True
    assert state.inventory == ["Torch"]


def test_multiple_changes_increment_turn_once(updater):
    manager = build_test_manager()

    updater.apply(
        world_interaction(),
        "You unlock the chest and take a torch.",
        manager,
        state_changes=(OPEN_CHEST, TAKE_TORCH, USE_KEY),
    )

    state = manager.get_state()

    assert state.turn == START_TURN + 1
    assert [event.turn for event in state.recent_events] == [
        START_TURN,
        START_TURN,
        START_TURN,
    ]


def test_successful_inventory_addition(updater):
    manager = build_test_manager()

    result = updater.apply(
        world_interaction(),
        "You pick up a torch.",
        manager,
        state_changes=(TAKE_TORCH,),
    )

    assert result.success
    assert manager.get_state().inventory == ["Key", "Torch"]


def test_successful_inventory_removal(updater):
    manager = build_test_manager()

    result = updater.apply(
        world_interaction(),
        "The key turns in the lock.",
        manager,
        state_changes=(USE_KEY,),
    )

    assert result.success
    assert manager.get_state().inventory == []


def test_successful_object_state_change(updater):
    manager = build_test_manager()

    result = updater.apply(
        world_interaction(),
        "The chest creaks open.",
        manager,
        state_changes=(OPEN_CHEST,),
    )

    assert result.success

    state = manager.get_state()

    assert find_object(state, "Old Chest").state == {"open": True}
    assert find_object(state, "Sword").state == {"visible": True}


def test_mixed_object_and_inventory_changes(updater):
    manager = build_test_manager()
    hide_sword = StateChangeAction(
        action_type="object_state_change",
        target="sword",
        changes={"visible": False},
    )

    result = updater.apply(
        world_interaction(),
        "You open the chest, hide the sword, and grab a torch.",
        manager,
        state_changes=(OPEN_CHEST, hide_sword, TAKE_TORCH),
    )

    assert result.success

    state = manager.get_state()

    assert find_object(state, "Old Chest").state["open"] is True
    assert find_object(state, "Sword").state["visible"] is False
    assert state.inventory == ["Key", "Torch"]
    assert len(state.recent_events) == 3


def test_changes_are_validated_in_order(updater):
    manager = build_test_manager()
    drop_torch = StateChangeAction(
        action_type="inventory_change",
        inventory_remove=("Torch",),
    )

    result = updater.apply(
        world_interaction(),
        "You light the torch, then drop it.",
        manager,
        state_changes=(TAKE_TORCH, drop_torch),
    )

    assert result.success
    assert manager.get_state().inventory == ["Key"]


def test_duplicate_object_changes_are_applied(updater):
    manager = build_test_manager()

    result = updater.apply(
        world_interaction(),
        "The chest swings open.",
        manager,
        state_changes=(OPEN_CHEST, OPEN_CHEST),
    )

    assert result.success
    assert result.applied_changes == (OPEN_CHEST, OPEN_CHEST)

    state = manager.get_state()

    assert find_object(state, "Old Chest").state["open"] is True
    assert len(state.recent_events) == 2
    assert state.turn == START_TURN + 1


# --------------------------------------------------
# History recording
# --------------------------------------------------


def test_player_history_is_recorded(updater):
    manager = build_test_manager()

    updater.apply(speech_interaction(), "Who are you?", manager)

    player_entry = manager.get_state().history[0]

    assert player_entry.speaker == "Player"
    assert player_entry.text == "Hello, guard."
    assert player_entry.entry_type == "speech"


def test_character_response_uses_responder_as_speaker(updater):
    manager = build_test_manager()

    updater.apply(speech_interaction(), "Who are you?", manager)

    response_entry = manager.get_state().history[1]

    assert response_entry.speaker == "Guard"
    assert response_entry.text == "Who are you?"
    assert response_entry.entry_type == "speech"


def test_world_response_uses_narrator_as_speaker(updater):
    manager = build_test_manager()

    updater.apply(
        world_interaction(),
        "The chest creaks open.",
        manager,
        state_changes=(OPEN_CHEST,),
    )

    response_entry = manager.get_state().history[1]

    assert NARRATOR_SPEAKER == "Narrator"
    assert response_entry.speaker == "Narrator"
    assert response_entry.entry_type == "action"


def test_no_responder_never_records_llm_speaker(updater):
    manager = build_test_manager()

    updater.apply(world_interaction(), "Nothing happens.", manager)

    speakers = [entry.speaker for entry in manager.get_state().history]

    assert speakers == ["Player", "Narrator"]
    assert all(speaker.lower() != "llm" for speaker in speakers)
    assert None not in speakers


def test_player_and_response_share_the_same_turn(updater):
    manager = build_test_manager()

    updater.apply(
        world_interaction(),
        "The chest creaks open.",
        manager,
        state_changes=(OPEN_CHEST,),
    )

    state = manager.get_state()

    assert [entry.turn for entry in state.history] == [
        START_TURN,
        START_TURN,
    ]
    assert state.recent_events[0].turn == START_TURN


def test_final_turn_is_previous_turn_plus_one(updater):
    manager = build_test_manager()
    previous_turn = manager.get_state().turn

    updater.apply(speech_interaction(), "Who are you?", manager)

    assert manager.get_state().turn == previous_turn + 1


def test_response_text_is_recorded_exactly(updater):
    manager = build_test_manager()
    response = '  "Halt!" the guard barks.\n\n  Who goes there?  '

    updater.apply(speech_interaction(), response, manager)

    assert manager.get_state().history[1].text == response


def test_input_text_is_recorded_exactly(updater):
    manager = build_test_manager()
    interaction = Interaction(
        input="Hello, GUARD... are you awake?",
        input_type="speech",
        input_character="Player",
        responder_character="Guard",
    )

    updater.apply(interaction, "I am now.", manager)

    assert (
        manager.get_state().history[0].text
        == "Hello, GUARD... are you awake?"
    )


# --------------------------------------------------
# Failed interactions
# --------------------------------------------------


def test_invalid_change_causes_no_partial_mutation(updater):
    manager = build_test_manager()
    before = manager.get_state()

    updater.apply(
        world_interaction(),
        "You open the chest and take a torch.",
        manager,
        state_changes=(OPEN_CHEST, TAKE_TORCH, MISSING_OBJECT),
    )

    after = manager.get_state()

    assert find_object(after, "Old Chest").state["open"] is False
    assert "Torch" not in after.inventory
    assert after.recent_events == []
    assert after.history == []
    assert after == before


def test_invalid_change_does_not_advance_turn(updater):
    manager = build_test_manager()

    updater.apply(
        world_interaction(),
        "You pry at the chest.",
        manager,
        state_changes=(MISSING_OBJECT,),
    )

    assert manager.get_state().turn == START_TURN


def test_final_change_failing_rejects_whole_batch(updater):
    manager = build_test_manager()
    changes = (OPEN_CHEST, TAKE_TORCH, MISSING_ITEM)

    result = updater.apply(
        world_interaction(),
        "You open the chest and take a torch.",
        manager,
        state_changes=changes,
    )

    assert result == StateUpdateResult(
        applied_changes=(),
        rejected_changes=changes,
        success=False,
    )


def test_invalid_first_change_rejects_whole_batch(updater):
    manager = build_test_manager()
    before = manager.get_state()
    changes = (MISSING_OBJECT, OPEN_CHEST)

    result = updater.apply(
        world_interaction(),
        "You open the chest.",
        manager,
        state_changes=changes,
    )

    assert not result.success
    assert result.applied_changes == ()
    assert result.rejected_changes == changes
    assert manager.get_state() == before


def test_duplicate_inventory_removal_is_rejected(updater):
    manager = build_test_manager()
    before = manager.get_state()

    result = updater.apply(
        world_interaction(),
        "You use the key twice.",
        manager,
        state_changes=(USE_KEY, USE_KEY),
    )

    assert not result.success
    assert result.rejected_changes == (USE_KEY, USE_KEY)
    assert manager.get_state() == before


def test_unsupported_action_type_is_rejected(updater):
    manager = build_test_manager()
    before = manager.get_state()
    teleport = StateChangeAction(action_type="teleport", target="Moon")

    result = updater.apply(
        world_interaction(),
        "You vanish.",
        manager,
        state_changes=(teleport,),
    )

    assert not result.success
    assert manager.get_state() == before


def test_state_unchanged_after_failed_batch(updater):
    manager = build_test_manager()
    before = manager.get_state()

    result = updater.apply(
        world_interaction(),
        "You open the chest and use the key.",
        manager,
        state_changes=(OPEN_CHEST, USE_KEY, MISSING_ITEM),
    )

    assert not result.success
    assert manager.get_state() == before


def test_updater_uses_supplied_action_validator():
    class RejectingValidator(ActionValidator):
        def validate_state_change(self, action, state):
            return False

    manager = build_test_manager()
    before = manager.get_state()

    result = StateUpdater(RejectingValidator()).apply(
        world_interaction(),
        "The chest creaks open.",
        manager,
        state_changes=(OPEN_CHEST,),
    )

    assert not result.success
    assert manager.get_state() == before


def test_failed_update_does_not_mutate_inputs(updater):
    manager = build_test_manager()
    interaction = world_interaction()
    changes = (OPEN_CHEST, MISSING_OBJECT)
    interaction_before = deepcopy(interaction)
    changes_before = deepcopy(changes)

    updater.apply(interaction, "Nothing happens.", manager, changes)

    assert interaction == interaction_before
    assert changes == changes_before


# --------------------------------------------------
# Determinism
# --------------------------------------------------


def test_repeated_calls_are_deterministic(updater):
    first_manager = build_test_manager()
    second_manager = build_test_manager()
    changes = (OPEN_CHEST, TAKE_TORCH)

    first = updater.apply(
        world_interaction(), "The chest opens.", first_manager, changes
    )
    second = updater.apply(
        world_interaction(), "The chest opens.", second_manager, changes
    )

    assert first == second
    assert first_manager.get_state() == second_manager.get_state()


def test_consecutive_interactions_advance_one_turn_each(updater):
    manager = build_test_manager()

    updater.apply(speech_interaction(), "Who are you?", manager)
    updater.apply(
        world_interaction(),
        "The chest creaks open.",
        manager,
        state_changes=(OPEN_CHEST,),
    )

    state = manager.get_state()

    assert state.turn == START_TURN + 2
    assert [entry.turn for entry in state.history] == [
        START_TURN,
        START_TURN,
        START_TURN + 1,
        START_TURN + 1,
    ]


# --------------------------------------------------
# GameStateManager batch support
# --------------------------------------------------


def test_apply_state_changes_commits_all_and_increments_turn_once():
    manager = build_test_manager()

    assert manager.apply_state_changes((OPEN_CHEST, TAKE_TORCH, USE_KEY))

    state = manager.get_state()

    assert find_object(state, "Old Chest").state["open"] is True
    assert state.inventory == ["Torch"]
    assert state.turn == START_TURN + 1
    assert len(state.recent_events) == 3


def test_apply_state_changes_commits_nothing_on_failure():
    manager = build_test_manager()
    before = manager.get_state()

    assert not manager.apply_state_changes(
        (OPEN_CHEST, TAKE_TORCH, MISSING_ITEM)
    )

    assert manager.get_state() == before


def test_apply_state_changes_with_no_changes_advances_turn():
    manager = build_test_manager()

    assert manager.apply_state_changes(())

    state = manager.get_state()

    assert state.turn == START_TURN + 1
    assert state.recent_events == []


def test_apply_state_change_matches_single_item_batch():
    single = build_test_manager()
    batch = build_test_manager()

    assert single.apply_state_change(OPEN_CHEST)
    assert batch.apply_state_changes((OPEN_CHEST,))

    assert single.get_state() == batch.get_state()

    state = single.get_state()

    assert state.turn == START_TURN + 1
    assert state.recent_events[0].turn == START_TURN
    assert state.recent_events[0].event_type == "object_state_change"
    assert (
        state.recent_events[0].description
        == "Applied state change: Old Chest"
    )


def test_apply_state_change_failure_is_unchanged():
    manager = build_test_manager()
    before = manager.get_state()

    assert not manager.apply_state_change(MISSING_OBJECT)
    assert not manager.apply_state_change(MISSING_ITEM)

    assert manager.get_state() == before
