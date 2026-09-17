import json
import os


PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

INPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "splits",
    "train.jsonl"
)


def build_prompt(example):

    setting = example["setting"]
    character = example["character"]
    world = example["world_state"]

    lines = []

    lines.append("You are an interactive fantasy game narrator.")
    lines.append(
        "Generate a response that is consistent with the "
        "current game world and the ongoing interaction."
    )

    lines.append("\n### SETTING")
    lines.append(f"Name: {setting.get('name', '')}")
    lines.append(f"Category: {setting.get('category', '')}")
    lines.append(
        f"Description: {setting.get('description', '')}"
    )
    lines.append(
        f"Background: {setting.get('background', '')}"
    )

    lines.append("\n### CHARACTER")
    lines.append(
        f"Name: {character.get('name', '')}"
    )
    lines.append(
        f"Persona: {character.get('persona', '')}"
    )

    lines.append("\n### WORLD STATE")

    lines.append(
        f"Context:\n{world.get('context', '')}"
    )

    lines.append(
        f"Room objects: "
        f"{', '.join(world.get('room_objects', [])) or 'None'}"
    )

    lines.append(
        f"Room agents: "
        f"{', '.join(world.get('room_agents', [])) or 'None'}"
    )

    lines.append(
        f"Carrying: "
        f"{', '.join(world.get('carrying', [])) or 'None'}"
    )

    lines.append(
        f"Wearing: "
        f"{', '.join(world.get('wearing', [])) or 'None'}"
    )

    lines.append(
        f"Wielding: "
        f"{', '.join(world.get('wielding', [])) or 'None'}"
    )

    available = world.get("available_actions", [])

    if available:
        lines.append("Available actions:")
        for action in available:
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
    # Player input
    # --------------------------------------------------

    lines.append("\n### PLAYER INPUT")
    lines.append(example["player_input"])

    lines.append("\n### INPUT TYPE")
    lines.append(example["input_type"])

    lines.append("\n### RESPONSE")

    return "\n".join(lines)


def main():

    print("=" * 70)
    print("TRAINING FORMAT PREVIEW")
    print("=" * 70)

    examples = []

    with open(INPUT_PATH, "r", encoding="utf-8") as f:

        for line in f:

            examples.append(json.loads(line))

            if len(examples) >= 5:
                break

    print(f"\nLoaded {len(examples)} examples for preview.")

    for i, example in enumerate(examples):

        prompt = build_prompt(example)

        print("\n" + "=" * 70)
        print(f"EXAMPLE {i + 1}")
        print("=" * 70)

        print(prompt)

        print("\n### TARGET")
        print(example["target"])

    print("\n" + "=" * 70)
    print("PREVIEW COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()