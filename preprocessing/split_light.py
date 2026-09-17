import json
import os
import random
from collections import defaultdict


PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

INPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "light_processed.jsonl"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "splits"
)

SEED = 42

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10


def main():

    print("=" * 70)
    print("LIGHT DATASET SPLITTING")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\nInput: {INPUT_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Random seed: {SEED}")

    # --------------------------------------------------
    # Load examples
    # --------------------------------------------------

    records = defaultdict(list)

    with open(INPUT_PATH, "r", encoding="utf-8") as f:

        for line in f:

            example = json.loads(line)

            # Keep all turns from the same original record together
            record_id = example["record_id"]

            records[record_id].append(example)

    record_ids = list(records.keys())

    print(f"\nUnique original records: {len(record_ids):,}")

    # --------------------------------------------------
    # Shuffle records reproducibly
    # --------------------------------------------------

    random.seed(SEED)
    random.shuffle(record_ids)

    total_records = len(record_ids)

    train_end = int(total_records * TRAIN_RATIO)
    val_end = train_end + int(total_records * VAL_RATIO)

    train_ids = record_ids[:train_end]
    val_ids = record_ids[train_end:val_end]
    test_ids = record_ids[val_end:]

    splits = {
        "train": train_ids,
        "validation": val_ids,
        "test": test_ids
    }

    # --------------------------------------------------
    # Write splits
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("SPLIT INFORMATION")
    print("=" * 70)

    for split_name, ids in splits.items():

        output_path = os.path.join(
            OUTPUT_DIR,
            f"{split_name}.jsonl"
        )

        example_count = 0

        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as f:

            for record_id in ids:

                for example in records[record_id]:

                    f.write(
                        json.dumps(
                            example,
                            ensure_ascii=False
                        ) + "\n"
                    )

                    example_count += 1

        print(
            f"{split_name.capitalize():12}"
            f" records={len(ids):6,}"
            f" examples={example_count:7,}"
        )

    # --------------------------------------------------
    # Verify no record leakage
    # --------------------------------------------------

    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)

    overlap = (
        train_set & val_set
        | train_set & test_set
        | val_set & test_set
    )

    print("\n" + "=" * 70)
    print("LEAKAGE CHECK")
    print("=" * 70)

    print(f"Overlapping record IDs: {len(overlap)}")

    if len(overlap) == 0:
        print("No record leakage detected. ✓")
    else:
        print("WARNING: Record leakage detected!")

    # --------------------------------------------------
    # Final summary
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("SPLITTING COMPLETE")
    print("=" * 70)

    print("\nCreated:")

    for split_name in splits:
        print(
            f" - data/splits/{split_name}.jsonl"
        )

    print("\n")


if __name__ == "__main__":
    main()