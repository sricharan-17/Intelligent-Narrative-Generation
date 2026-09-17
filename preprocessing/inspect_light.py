import os
import pickle
from collections import Counter

LIGHT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    ".venv",
    "Lib",
    "site-packages",
    "data",
    "light_dialogue",
    "light_data.pkl"
)

SEQUENCE_FIELDS = [
    "character",
    "context",
    "room_objects",
    "room_agents",
    "available_actions",
    "carrying",
    "wearing",
    "wielding",
    "speech",
    "emote",
    "action",
]


def main():
    print("=" * 70)
    print("LIGHT DATASET STATISTICS")
    print("=" * 70)

    print(f"\nFile: {LIGHT_PATH}")

    with open(LIGHT_PATH, "rb") as f:
        data = pickle.load(f)

    print(f"\nTotal records: {len(data):,}")

    total_turns = 0
    speech_turns = 0
    action_turns = 0
    emote_turns = 0

    turn_counts = []
    missing_counts = Counter()
    alignment_errors = []

    for record_idx, record in enumerate(data):

        # Check sequence lengths
        lengths = {}

        for field in SEQUENCE_FIELDS:
            value = record.get(field)

            if value is None:
                missing_counts[field] += 1
                continue

            try:
                lengths[field] = len(value)
            except TypeError:
                missing_counts[field] += 1

        if lengths:
            unique_lengths = set(lengths.values())

            if len(unique_lengths) != 1:
                alignment_errors.append(
                    (record_idx, lengths)
                )

            turns = max(unique_lengths)
            total_turns += turns
            turn_counts.append(turns)

        # Count actual interaction types
        speech = record.get("speech", [])
        action = record.get("action", [])
        emote = record.get("emote", [])

        if speech:
            speech_turns += sum(
                1 for x in speech
                if x is not None and str(x).strip()
            )

        if action:
            action_turns += sum(
                1 for x in action
                if x is not None and str(x).strip()
            )

        if emote:
            emote_turns += sum(
                1 for x in emote
                if x is not None and str(x).strip()
            )

    print("\n" + "=" * 70)
    print("TURN STATISTICS")
    print("=" * 70)

    print(f"Total interaction turns: {total_turns:,}")
    print(f"Speech turns:           {speech_turns:,}")
    print(f"Action turns:           {action_turns:,}")
    print(f"Emote turns:            {emote_turns:,}")

    if turn_counts:
        print(f"\nAverage turns/record: {sum(turn_counts) / len(turn_counts):.2f}")
        print(f"Minimum turns/record: {min(turn_counts)}")
        print(f"Maximum turns/record: {max(turn_counts)}")

    print("\n" + "=" * 70)
    print("MISSING FIELDS")
    print("=" * 70)

    if missing_counts:
        for field, count in missing_counts.items():
            print(f"{field}: {count:,}")
    else:
        print("No missing fields found.")

    print("\n" + "=" * 70)
    print("SEQUENCE ALIGNMENT")
    print("=" * 70)

    print(f"Records with alignment errors: {len(alignment_errors):,}")

    if alignment_errors:
        print("\nFirst 5 alignment errors:")

        for record_idx, lengths in alignment_errors[:5]:
            print(f"Record {record_idx}: {lengths}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()