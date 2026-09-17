import json
import os
from collections import Counter


PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "light_processed.jsonl"
)


def main():

    print("=" * 70)
    print("PROCESSED LIGHT DATASET VALIDATION")
    print("=" * 70)

    total = 0
    invalid = 0

    input_types = Counter()

    empty_history = 0
    action_without_available = 0
    same_input_target = 0

    missing_input_character = 0
    missing_responder_character = 0

    required_fields = [
        "record_id",
        "turn_id",
        "input_character",
        "responder_character",
        "setting",
        "world_state",
        "history",
        "input",
        "input_type",
        "target"
    ]

    with open(DATA_PATH, "r", encoding="utf-8") as f:

        for line in f:

            try:
                example = json.loads(line)
            except json.JSONDecodeError:
                invalid += 1
                continue

            total += 1

            # Required fields
            if not all(
                field in example
                for field in required_fields
            ):
                invalid += 1
                continue

            # Character validation
            if not example["input_character"].get("name"):
                missing_input_character += 1

            if not example["responder_character"].get("name"):
                missing_responder_character += 1

            # Input type
            input_types[
                example["input_type"]
            ] += 1

            # History
            if not example["history"]:
                empty_history += 1

            # Action availability
            available = example[
                "world_state"
            ].get(
                "available_actions",
                []
            )

            if (
                example["input_type"] == "action"
                and not available
            ):
                action_without_available += 1

            # Input == target
            if (
                example["input"].strip().lower()
                == example["target"].strip().lower()
            ):
                same_input_target += 1

    print(f"\nTotal examples: {total:,}")
    print(f"Invalid examples: {invalid:,}")

    print("\n" + "=" * 70)
    print("INPUT TYPES")
    print("=" * 70)

    for key, value in sorted(input_types.items()):

        percentage = value / total * 100

        print(
            f"{key:10}: "
            f"{value:7,} "
            f"({percentage:5.2f}%)"
        )

    print("\n" + "=" * 70)
    print("QUALITY CHECKS")
    print("=" * 70)

    print(
        f"Examples with empty history: "
        f"{empty_history:,}"
    )

    print(
        f"Action examples without available actions: "
        f"{action_without_available:,}"
    )

    print(
        f"Examples where input == target: "
        f"{same_input_target:,}"
    )

    print(
        f"Missing input character: "
        f"{missing_input_character:,}"
    )

    print(
        f"Missing responder character: "
        f"{missing_responder_character:,}"
    )

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()