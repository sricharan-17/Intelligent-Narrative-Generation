import json
import os


PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

INPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "light_processed.jsonl"
)


def main():

    print("=" * 70)
    print("LIGHT ACTION EXAMPLE INSPECTION")
    print("=" * 70)

    count = 0

    with open(INPUT_PATH, "r", encoding="utf-8") as f:

        for line in f:

            example = json.loads(line)

            if example["input_type"] != "action":
                continue

            count += 1

            print("\n" + "=" * 70)
            print(f"ACTION EXAMPLE {count}")
            print("=" * 70)

            print(f"\nCharacter:")
            print(example["character"]["name"])

            print(f"\nPersona:")
            print(example["character"]["persona"])

            print(f"\nContext:")
            print(example["world_state"]["context"])

            print("\nAvailable actions:")

            for action in example["world_state"]["available_actions"]:
                print(f"  - {action}")

            print(f"\nHistory:")
            for item in example["history"][-6:]:
                print(
                    f"  {item['speaker']} "
                    f"[{item['type']}]: "
                    f"{item['text']}"
                )

            print(f"\nPLAYER ACTION:")
            print(example["player_input"])

            print(f"\nTARGET:")
            print(example["target"])

            if count >= 10:
                break

    print("\n" + "=" * 70)
    print(f"Inspected {count} action examples")
    print("=" * 70)


if __name__ == "__main__":
    main()