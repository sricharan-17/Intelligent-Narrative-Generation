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
        f"Room objects: "
        f"{', '.join(world.get('room_objects', [])) or 'None'}"
    )

    lines.append(
        f"Room agents: "
        f"{', '.join(world.get('room_agents', [])) or 'None'}"
    )

    object_descriptions = world.get("object_descriptions", {})

    if object_descriptions:
        lines.append("Object descriptions:")
        for object_name, description in object_descriptions.items():
            lines.append(f"- {object_name}: {description}")
    else:
        lines.append("Object descriptions: None")

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

    lines.append(
        example["input"]
    )

    # --------------------------------------------------
    # Input type
    # --------------------------------------------------

    lines.append("\n### INPUT TYPE")

    lines.append(
        example["input_type"]
    )

    # --------------------------------------------------
    # Response
    # --------------------------------------------------

    lines.append("\n### RESPONSE")

    return "\n".join(lines)


def main():

    print("=" * 70)
    print("TRAINING DATA FORMAT PREVIEW")
    print("=" * 70)

    count = 0

    with open(
        INPUT_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            example = json.loads(line)

            prompt = build_prompt(example)

            print("\n" + "=" * 70)
            print(f"EXAMPLE {count + 1}")
            print("=" * 70)

            print("\nPROMPT:")
            print("-" * 70)
            print(prompt)

            print("\nTARGET:")
            print("-" * 70)
            print(example["target"])

            count += 1

            if count >= 5:
                break

    print("\n" + "=" * 70)
    print(f"Previewed {count} examples")
    print("=" * 70)


if __name__ == "__main__":
    main()