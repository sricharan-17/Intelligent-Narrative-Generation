import json
import os

from training.config import (
    TRAIN_FILE,
    VALIDATION_FILE,
    TEST_FILE,
)


PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))


def build_prompt(example):
    """
    Build the canonical semantic prompt used for all models.
    Model-specific tokenization and chat templates are applied later.
    """

    setting = example["setting"]
    input_character = example["input_character"]
    responder_character = example["responder_character"]
    world = example["world_state"]
    history = example["history"]

    lines = []

    lines.append(
        "You are an interactive fantasy game narrator. "
        "Generate the response of the responder character based on "
        "the current world state, character personas, available actions, "
        "conversation history, and current player input."
    )

    # --------------------------------------------------
    # Setting
    # --------------------------------------------------

    lines.append("\n### SETTING")

    lines.append(
        f"Name: {setting.get('name', '')}"
    )

    lines.append(
        f"Category: {setting.get('category', '')}"
    )

    lines.append(
        f"Description: {setting.get('description', '')}"
    )

    lines.append(
        f"Background: {setting.get('background', '')}"
    )

    # --------------------------------------------------
    # Input character
    # --------------------------------------------------

    lines.append("\n### INPUT CHARACTER")

    lines.append(
        f"Name: {input_character.get('name', '')}"
    )

    lines.append(
        f"Persona:\n{input_character.get('persona', '')}"
    )

    # --------------------------------------------------
    # Responder character
    # --------------------------------------------------

    lines.append("\n### RESPONDING CHARACTER")

    lines.append(
        f"Name: {responder_character.get('name', '')}"
    )

    lines.append(
        f"Persona:\n{responder_character.get('persona', '')}"
    )

    # --------------------------------------------------
    # World state
    # --------------------------------------------------

    lines.append("\n### WORLD STATE")

    lines.append(
        f"Context:\n{world.get('context', '')}"
    )

    lines.append(
        "Room objects: "
        + (
            ", ".join(world.get("room_objects", []))
            or "None"
        )
    )

    lines.append(
        "Room agents: "
        + (
            ", ".join(world.get("room_agents", []))
            or "None"
        )
    )

    object_descriptions = world.get("object_descriptions", {})

    if object_descriptions:
        lines.append("Object descriptions:")
        for object_name, description in object_descriptions.items():
            lines.append(f"- {object_name}: {description}")
    else:
        lines.append("Object descriptions: None")

    lines.append(
        "Carrying: "
        + (
            ", ".join(world.get("carrying", []))
            or "None"
        )
    )

    lines.append(
        "Wearing: "
        + (
            ", ".join(world.get("wearing", []))
            or "None"
        )
    )

    lines.append(
        "Wielding: "
        + (
            ", ".join(world.get("wielding", []))
            or "None"
        )
    )

    available_actions = world.get("available_actions", [])

    if available_actions:
        lines.append("Available actions:")
        for action in available_actions:
            lines.append(f"- {action}")
    else:
        lines.append("Available actions: None")

    # --------------------------------------------------
    # History
    # --------------------------------------------------

    lines.append("\n### HISTORY")

    if history:
        for item in history:
            lines.append(
                f"{item['speaker']} "
                f"[{item['type']}]: "
                f"{item['text']}"
            )
    else:
        lines.append("None")

    # --------------------------------------------------
    # Current input
    # --------------------------------------------------

    lines.append("\n### PLAYER INPUT")
    lines.append(example["input"])

    # --------------------------------------------------
    # Input type
    # --------------------------------------------------

    lines.append("\n### INPUT TYPE")
    lines.append(example["input_type"])

    # --------------------------------------------------
    # Response
    # --------------------------------------------------

    lines.append("\n### RESPONSE")

    return "\n".join(lines)


def load_jsonl(path):
    data = []

    with open(
        os.path.join(PROJECT_ROOT, path),
        "r",
        encoding="utf-8"
    ) as f:
        for line in f:
            data.append(json.loads(line))

    return data


def main():

    print("=" * 70)
    print("TRAINING DATA PREPARATION")
    print("=" * 70)

    train_data = load_jsonl(TRAIN_FILE)
    validation_data = load_jsonl(VALIDATION_FILE)
    test_data = load_jsonl(TEST_FILE)

    print(f"\nTrain examples:      {len(train_data):,}")
    print(f"Validation examples: {len(validation_data):,}")
    print(f"Test examples:       {len(test_data):,}")

    # --------------------------------------------------
    # Preview
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING EXAMPLE PREVIEW")
    print("=" * 70)

    if train_data:
        example = train_data[0]

        print("\nPROMPT:")
        print("-" * 70)
        print(build_prompt(example))

        print("\nTARGET:")
        print("-" * 70)
        print(example["target"])

    print("\n" + "=" * 70)
    print("VALIDATION EXAMPLE PREVIEW")
    print("=" * 70)

    if validation_data:
        example = validation_data[0]

        print("\nPROMPT:")
        print("-" * 70)
        print(build_prompt(example))

        print("\nTARGET:")
        print("-" * 70)
        print(example["target"])

    print("\n" + "=" * 70)
    print("TEST EXAMPLE PREVIEW")
    print("=" * 70)

    if test_data:
        example = test_data[0]

        print("\nPROMPT:")
        print("-" * 70)
        print(build_prompt(example))

        print("\nTARGET:")
        print("-" * 70)
        print(example["target"])

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()