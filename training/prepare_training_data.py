import json
import os


PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

SPLITS = {
    "train": os.path.join(
        PROJECT_ROOT, "data", "splits", "train.jsonl"
    ),
    "validation": os.path.join(
        PROJECT_ROOT, "data", "splits", "validation.jsonl"
    ),
    "test": os.path.join(
        PROJECT_ROOT, "data", "splits", "test.jsonl"
    ),
}


def build_prompt(example):
    """
    Build the canonical prompt used for all three models.

    The prompt content is intentionally model-independent.
    Model-specific chat templates are applied later during
    tokenization.
    """

    setting = example["setting"]
    input_character = example["input_character"]
    responder_character = example["responder_character"]
    world = example["world_state"]

    lines = []

    lines.append("You are an interactive fantasy game narrator.")
    lines.append(
        "Generate the response of the responding character "
        "to the current player interaction while remaining "
        "consistent with the game world, character personas, "
        "available actions, and conversation history."
    )

    # --------------------------------------------------
    # Setting
    # --------------------------------------------------

    lines.append("\n### SETTING")
    lines.append(f"Name: {setting.get('name', '')}")
    lines.append(f"Category: {setting.get('category', '')}")
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
        f"Persona: {input_character.get('persona', '')}"
    )

    # --------------------------------------------------
    # Responding character
    # --------------------------------------------------

    lines.append("\n### RESPONDING CHARACTER")
    lines.append(
        f"Name: {responder_character.get('name', '')}"
    )
    lines.append(
        f"Persona: {responder_character.get('persona', '')}"
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
    # Conversation history
    # --------------------------------------------------

    lines.append("\n### HISTORY")

    history = example.get("history", [])

    if history:
        for item in history:
            speaker = item.get("speaker", "unknown")
            item_type = item.get("type", "unknown")
            text = item.get("text", "")

            lines.append(
                f"{speaker} [{item_type}]: {text}"
            )
    else:
        lines.append("No previous interaction.")

    # --------------------------------------------------
    # Current input
    # --------------------------------------------------

    lines.append("\n### CURRENT INPUT")

    lines.append(
        f"Character: {input_character.get('name', '')}"
    )
    lines.append(
        f"Input: {example['input']}"
    )
    lines.append(
        f"Input type: {example['input_type']}"
    )

    # --------------------------------------------------
    # Response marker
    # --------------------------------------------------

    lines.append("\n### RESPONSE")

    return "\n".join(lines)


def load_split(split_name):
    """
    Load one JSONL split.
    """

    path = SPLITS[split_name]

    examples = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    return examples


def prepare_example(example):
    """
    Convert one processed example into the canonical
    prompt/target representation.

    Tokenization is intentionally NOT performed here.
    """

    prompt = build_prompt(example)
    target = example["target"]

    return {
        "record_id": example["record_id"],
        "turn_id": example["turn_id"],
        "input_type": example["input_type"],
        "prompt": prompt,
        "target": target,
    }


def main():

    print("=" * 70)
    print("TRAINING DATA PREPARATION")
    print("=" * 70)

    for split_name in SPLITS:

        print(f"\nLoading {split_name} split...")

        examples = load_split(split_name)

        print(f"Examples loaded: {len(examples):,}")

        if not examples:
            print("WARNING: split is empty.")
            continue

        prepared = prepare_example(examples[0])

        print("\nFirst example:")
        print("-" * 70)

        print(prepared["prompt"])

        print("\n### TARGET")
        print(prepared["target"])

        print("-" * 70)

    print("\n" + "=" * 70)
    print("PREPARATION CHECK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()