"""Play The Ashen Crown prototype from the command line.

Usage (from the project root):

    python scripts/run_prototype.py --fake   # deterministic, no model
    python scripts/run_prototype.py          # fine-tuned SmolLM2 adapter

Commands:
    say <Character>: <text>   speak to a character (e.g. say Mira: Hello)
    <anything else>           an action, narrated (e.g. open letter,
                              take key, travel to Whispering Forest)
    status                    show the current state
    help                      show this help
    quit                      exit
"""

import argparse
import sys
from pathlib import Path

# Allow "python scripts/run_prototype.py" as well as "python -m".
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from game.gamestate.interaction.actions import StateChangeAction  # noqa: E402
from game.gamestate.state.manager import GameStateManager  # noqa: E402
from game.pipeline.pipeline import PipelineResult  # noqa: E402
from game.world.fake_generator import PrototypeFakeGenerator  # noqa: E402
from game.world.prototype import (  # noqa: E402
    PROTOTYPE_ACTIONS,
    PROTOTYPE_TITLE,
    create_prototype_manager,
    create_prototype_pipeline,
    run_prototype_turn,
)


def create_generator(fake: bool):
    if fake:
        return PrototypeFakeGenerator()

    # Only this runner imports llm; game/ stays independent of the model.
    from llm.generator import NarrativeGenerator

    generator = NarrativeGenerator()
    print("Loading fine-tuned model (this can take a while)...")
    generator.load()
    print(f"Model ready: {generator.status()}")

    return generator


def parse_command(
    line: str,
    manager: GameStateManager,
) -> tuple[str, str, str | None] | None:
    """Return (input_text, input_type, responder) or None if invalid."""

    if line.lower().startswith("say ") and ":" in line:
        name, text = line[4:].split(":", 1)

        character = next(
            (
                c for c in manager.get_state().characters
                if c.name.lower() == name.strip().lower()
            ),
            None,
        )

        if character is None:
            print(f"There is no character called {name.strip()!r}.")
            return None

        if not character.present:
            print(f"{character.name} is not here.")
            return None

        return text.strip(), "speech", character.name

    return line, "action", None


def print_status(manager: GameStateManager) -> None:
    state = manager.get_state()
    location = state.location.name if state.location else "Unknown"
    present = [c.name for c in state.characters if c.present]

    print(f"\n[Turn {state.turn}] Location: {location}")
    print(f"  Present:   {', '.join(present) or 'nobody'}")
    print(f"  Inventory: {', '.join(state.inventory) or 'empty'}")
    print(f"  Actions:   {', '.join(state.available_actions)}")


def describe_change(change: StateChangeAction) -> str:
    if change.action_type == "inventory_change":
        parts = [f"+{item}" for item in change.inventory_add]
        parts += [f"-{item}" for item in change.inventory_remove]
        return f"inventory {' '.join(parts)}"

    return f"{change.target} {change.changes}"


def print_result(result: PipelineResult) -> None:
    if not result.validation.is_valid:
        print("\nThe generated response was rejected:")
        for error in result.validation.errors:
            print(f"  - {error}")
        print(f"  Raw response: {result.raw_response!r}")
        return

    if not result.success:
        print("\nThat could not happen; the game state was not changed.")
        for change in result.state_update.rejected_changes:
            print(f"  (rejected: {describe_change(change)})")
        return

    print(f"\n{result.output}")

    for change in result.state_update.applied_changes:
        print(f"  (state: {describe_change(change)})")


def main() -> int:
    parser = argparse.ArgumentParser(description=f"Play {PROTOTYPE_TITLE}.")
    parser.add_argument(
        "--fake",
        action="store_true",
        help="use the deterministic fake generator instead of the model",
    )
    args = parser.parse_args()

    manager = create_prototype_manager()
    pipeline = create_prototype_pipeline(create_generator(args.fake))

    print(f"=== {PROTOTYPE_TITLE} ===")
    print(__doc__.split("Commands:", 1)[1].rstrip())
    print(f"\nPrototype actions: {', '.join(PROTOTYPE_ACTIONS)}")
    print_status(manager)

    while True:
        try:
            line = input("\n> ").strip()
        except EOFError:
            print()
            break

        if not line:
            continue

        command = line.lower()

        if command in ("quit", "exit"):
            break

        if command == "help":
            print(__doc__)
            continue

        if command == "status":
            print_status(manager)
            continue

        parsed = parse_command(line, manager)
        if parsed is None:
            continue

        input_text, input_type, responder = parsed

        try:
            result = run_prototype_turn(
                pipeline,
                manager,
                input_text,
                input_type,
                responder,
            )
        except (TypeError, ValueError) as error:
            # InputInteractionProcessor rejects invalid input this way.
            print(f"Error: {error}")
            continue

        print_result(result)
        print_status(manager)

    return 0


if __name__ == "__main__":
    sys.exit(main())
