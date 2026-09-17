import json
import os
from collections import Counter


PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
SPLIT_DIR = os.path.join(PROJECT_ROOT, "data", "splits")

SPLITS = [
    "train",
    "validation",
    "test"
]


def inspect_split(name):

    path = os.path.join(SPLIT_DIR, f"{name}.jsonl")

    total = 0
    types = Counter()
    records = set()

    with open(path, "r", encoding="utf-8") as f:

        for line in f:

            example = json.loads(line)

            total += 1
            types[example["input_type"]] += 1
            records.add(example["record_id"])

    print(f"\n{name.upper()}")
    print("-" * 50)
    print(f"Examples: {total:,}")
    print(f"Records:  {len(records):,}")

    for input_type, count in sorted(types.items()):

        percentage = count / total * 100

        print(
            f"{input_type:10}: "
            f"{count:7,} "
            f"({percentage:5.2f}%)"
        )


def main():

    print("=" * 70)
    print("LIGHT SPLIT DISTRIBUTION CHECK")
    print("=" * 70)

    for split in SPLITS:
        inspect_split(split)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()