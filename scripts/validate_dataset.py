import json
from collections import Counter

FILES = {
    "processed": "data/processed/light_processed.jsonl",
    "train": "data/splits/train.jsonl",
    "validation": "data/splits/validation.jsonl",
    "test": "data/splits/test.jsonl",
}

REQUIRED_FIELDS = [
    "record_id",
    "turn_id",
    "input_character",
    "responder_character",
    "setting",
    "world_state",
    "history",
    "input",
    "input_type",
    "target",
]


def validate_file(name, path):
    print(f"\n{'=' * 60}")
    print(f"VALIDATING: {name}")
    print(f"{'=' * 60}")

    total = 0
    invalid_json = 0
    missing_fields = Counter()
    empty_input = 0
    empty_target = 0
    input_types = Counter()
    record_ids = set()
    turn_ids = set()
    duplicate_pairs = 0

    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                invalid_json += 1
                continue

            total += 1

            # Required fields
            for field in REQUIRED_FIELDS:
                if field not in data:
                    missing_fields[field] += 1

            # Empty input / target
            if not str(data.get("input", "")).strip():
                empty_input += 1

            if not str(data.get("target", "")).strip():
                empty_target += 1

            # Input types
            input_types[data.get("input_type")] += 1

            # Duplicate record/turn combinations
            pair = (data.get("record_id"), data.get("turn_id"))

            if pair in turn_ids:
                duplicate_pairs += 1

            turn_ids.add(pair)
            record_ids.add(data.get("record_id"))

    print(f"Total records:              {total:,}")
    print(f"Invalid JSON lines:         {invalid_json:,}")
    print(f"Records with missing fields:{sum(missing_fields.values()):,}")
    print(f"Empty inputs:                {empty_input:,}")
    print(f"Empty targets:               {empty_target:,}")
    print(f"Unique record IDs:           {len(record_ids):,}")
    print(f"Unique (record_id, turn_id): {len(turn_ids):,}")
    print(f"Duplicate (record_id, turn_id): {duplicate_pairs:,}")

    print("\nInput types:")
    for input_type, count in input_types.items():
        print(f"  {input_type}: {count:,}")

    print("\nMissing fields:")
    if missing_fields:
        for field, count in missing_fields.items():
            print(f"  {field}: {count:,}")
    else:
        print("  None")


print("\nLIGHT DATASET VALIDATION")
print("=" * 60)

for name, path in FILES.items():
    validate_file(name, path)

print("\n" + "=" * 60)
print("VALIDATION COMPLETE")
print("=" * 60)