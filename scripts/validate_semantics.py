import json
from collections import Counter, defaultdict

FILES = {
    "processed": "data/processed/light_processed.jsonl",
    "train": "data/splits/train.jsonl",
    "validation": "data/splits/validation.jsonl",
    "test": "data/splits/test.jsonl",
}


def normalize(text):
    """Normalize text for basic comparison."""
    return str(text).strip().lower()


def validate_file(name, path):
    print(f"\n{'=' * 70}")
    print(f"SEMANTIC VALIDATION: {name}")
    print(f"{'=' * 70}")

    total = 0

    # Counters
    missing_world_state = 0
    missing_history = 0
    missing_characters = 0
    invalid_input_type = 0

    character_not_in_room = 0
    invalid_action_references = 0

    empty_context = 0
    empty_persona = 0
    short_targets = 0

    turn_sequences = defaultdict(list)

    input_lengths = []
    target_lengths = []

    suspicious_actions = Counter()

    with open(path, "r", encoding="utf-8") as f:

        for line_number, line in enumerate(f, 1):

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            total += 1

            # ---------------------------------------------------------
            # BASIC STRUCTURE
            # ---------------------------------------------------------

            world_state = data.get("world_state")

            if not isinstance(world_state, dict):
                missing_world_state += 1
                continue

            history = data.get("history")

            if not isinstance(history, list):
                missing_history += 1

            input_character = data.get("input_character", {})
            responder_character = data.get("responder_character", {})

            if not isinstance(input_character, dict) or \
               not isinstance(responder_character, dict):
                missing_characters += 1

            # ---------------------------------------------------------
            # INPUT TYPE
            # ---------------------------------------------------------

            input_type = data.get("input_type")

            if input_type not in {"speech", "action"}:
                invalid_input_type += 1

            # ---------------------------------------------------------
            # CHARACTER CONSISTENCY
            # ---------------------------------------------------------

            room_agents = world_state.get("room_agents", [])

            room_agents_normalized = {
                normalize(x) for x in room_agents
            }

            input_name = normalize(
                input_character.get("name", "")
            )

            responder_name = normalize(
                responder_character.get("name", "")
            )

            # Check whether the characters are represented
            # somewhere in the room state.
            #
            # LIGHT sometimes uses article variations such as
            # "a soldier" vs "soldier", so compare loosely.

            def character_present(name):
                if not name:
                    return False

                name = name.replace("a ", "").replace("an ", "")

                return any(
                    name in agent.replace("a ", "").replace("an ", "")
                    for agent in room_agents_normalized
                )

            if not character_present(input_name):
                character_not_in_room += 1

            if not character_present(responder_name):
                character_not_in_room += 1

            # ---------------------------------------------------------
            # WORLD STATE CONTENT
            # ---------------------------------------------------------

            context = world_state.get("context", "")

            if not str(context).strip():
                empty_context += 1

            # ---------------------------------------------------------
            # PERSONAS
            # ---------------------------------------------------------

            if not str(input_character.get("persona", "")).strip():
                empty_persona += 1

            if not str(responder_character.get("persona", "")).strip():
                empty_persona += 1

            # ---------------------------------------------------------
            # ACTION REFERENCES
            # ---------------------------------------------------------

            available_actions = world_state.get(
                "available_actions", []
            )

            known_objects = set()

            for obj in world_state.get("room_objects", []):
                known_objects.add(normalize(obj))

            for obj in world_state.get("carrying", []):
                known_objects.add(normalize(obj))

            for obj in world_state.get("wearing", []):
                known_objects.add(normalize(obj))

            for obj in world_state.get("wielding", []):
                known_objects.add(normalize(obj))

            for obj in world_state.get(
                "object_descriptions", {}
            ).keys():
                known_objects.add(normalize(obj))

            known_characters = set(room_agents_normalized)

            for action in available_actions:

                action_normalized = normalize(action)

                # Look for obvious object references.
                words = action_normalized.split()

                references_known_entity = False

                for entity in known_objects | known_characters:

                    entity_clean = (
                        entity.replace("a ", "")
                              .replace("an ", "")
                    )

                    if entity_clean and entity_clean in action_normalized:
                        references_known_entity = True
                        break

                if not references_known_entity:
                    suspicious_actions[action_normalized] += 1
                    invalid_action_references += 1

            # ---------------------------------------------------------
            # HISTORY
            # ---------------------------------------------------------

            if isinstance(history, list):

                for item in history:

                    if not isinstance(item, dict):
                        missing_history += 1
                        continue

                    if not str(item.get("speaker", "")).strip():
                        missing_history += 1

                    if not str(item.get("text", "")).strip():
                        missing_history += 1

                    if item.get("type") not in {"speech", "action"}:
                        missing_history += 1

            # ---------------------------------------------------------
            # TURN SEQUENCES
            # ---------------------------------------------------------

            record_id = data.get("record_id")
            turn_id = data.get("turn_id")

            turn_sequences[record_id].append(turn_id)

            # ---------------------------------------------------------
            # TEXT LENGTH
            # ---------------------------------------------------------

            input_text = str(data.get("input", ""))
            target_text = str(data.get("target", ""))

            input_lengths.append(len(input_text))
            target_lengths.append(len(target_text))

            if len(target_text.strip()) < 3:
                short_targets += 1

    # =============================================================
    # TURN SEQUENCE VALIDATION
    # =============================================================

    broken_sequences = 0

    for record_id, turns in turn_sequences.items():

        turns_sorted = sorted(turns)

        expected = list(range(
            turns_sorted[0],
            turns_sorted[0] + len(turns_sorted)
        ))

        if turns_sorted != expected:
            broken_sequences += 1

    # =============================================================
    # RESULTS
    # =============================================================

    print(f"Total records:                  {total:,}")
    print(f"Characters not in room state:   {character_not_in_room:,}")
    print(f"Invalid input types:             {invalid_input_type:,}")
    print(f"Missing world state:              {missing_world_state:,}")
    print(f"Invalid/missing history entries: {missing_history:,}")
    print(f"Empty world contexts:              {empty_context:,}")
    print(f"Empty personas:                    {empty_persona:,}")
    print(f"Suspicious action references:      {invalid_action_references:,}")
    print(f"Very short targets (<3 chars):     {short_targets:,}")
    print(f"Broken turn sequences:              {broken_sequences:,}")

    if input_lengths:
        print("\nText lengths:")
        print(f"  Average input length:  {sum(input_lengths) / len(input_lengths):.1f} chars")
        print(f"  Average target length: {sum(target_lengths) / len(target_lengths):.1f} chars")
        print(f"  Maximum input length:  {max(input_lengths):,} chars")
        print(f"  Maximum target length: {max(target_lengths):,} chars")

    print("\nMost common suspicious actions:")

    if suspicious_actions:
        for action, count in suspicious_actions.most_common(15):
            print(f"  {count:6,} × {action}")
    else:
        print("  None detected")


print("\nLIGHT DATASET — SEMANTIC VALIDATION")
print("=" * 70)

for name, path in FILES.items():
    validate_file(name, path)

print("\n" + "=" * 70)
print("SEMANTIC VALIDATION COMPLETE")
print("=" * 70)