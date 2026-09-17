import os
import json
import pickle


LIGHT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    ".venv",
    "Lib",
    "site-packages",
    "data",
    "light_dialogue",
    "light_data.pkl"
)

OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data",
    "processed",
    "light_processed.jsonl"
)


def clean_text(value):
    if value is None:
        return ""

    return str(value).strip()


def get_character_info(character_name, persona_lookup):
    return {
        "name": character_name,
        "persona": persona_lookup.get(character_name, "")
    }


def main():

    print("=" * 70)
    print("LIGHT DATASET PREPROCESSING")
    print("=" * 70)

    print(f"\nInput : {LIGHT_PATH}")
    print(f"Output: {OUTPUT_PATH}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    print("\nLoading LIGHT dataset...")

    with open(LIGHT_PATH, "rb") as f:
        data = pickle.load(f)

    print(f"Loaded {len(data):,} records.")

    examples = []

    speech_examples = 0
    action_examples = 0
    skipped_no_response = 0

    for record_idx, record in enumerate(data):

        agents = record.get("agents", [])
        setting = record.get("setting", {})

        characters = record.get("character", [])
        contexts = record.get("context", [])

        room_objects = record.get("room_objects", [])
        room_agents = record.get("room_agents", [])
        available_actions = record.get("available_actions", [])

        carrying = record.get("carrying", [])
        wearing = record.get("wearing", [])
        wielding = record.get("wielding", [])

        speech = record.get("speech", [])
        actions = record.get("action", [])
        emotes = record.get("emote", [])

        all_descriptions = record.get(
            "all_descriptions",
            {}
        )

        num_turns = len(contexts)

        # --------------------------------------------------
        # Character persona lookup
        # --------------------------------------------------

        persona_lookup = {}

        for agent in agents:

            if isinstance(agent, dict):

                name = clean_text(
                    agent.get("name")
                )

                persona = clean_text(
                    agent.get("persona")
                )

                if name:
                    persona_lookup[name] = persona

        # --------------------------------------------------
        # Process each turn
        # --------------------------------------------------

        for i in range(num_turns):

            input_character = clean_text(
                characters[i]
            )

            context = clean_text(
                contexts[i]
            )

            current_speech = (
                clean_text(speech[i])
                if i < len(speech)
                else ""
            )

            current_action = (
                clean_text(actions[i])
                if i < len(actions)
                else ""
            )

            # --------------------------------------------------
            # Determine current input
            # --------------------------------------------------

            if current_action:

                input_type = "action"
                current_input = current_action

            elif current_speech:

                input_type = "speech"
                current_input = current_speech

            else:
                continue

            # --------------------------------------------------
            # We need the next turn as the response
            # --------------------------------------------------

            if i + 1 >= num_turns:

                skipped_no_response += 1
                continue

            responder_character = clean_text(
                characters[i + 1]
            )

            next_speech = (
                clean_text(speech[i + 1])
                if i + 1 < len(speech)
                else ""
            )

            next_action = (
                clean_text(actions[i + 1])
                if i + 1 < len(actions)
                else ""
            )

            next_emote = (
                clean_text(emotes[i + 1])
                if i + 1 < len(emotes)
                else ""
            )

            # Prefer spoken response.
            if next_speech:

                target = next_speech

            elif next_action:

                target = next_action

            elif next_emote:

                target = next_emote

            else:

                skipped_no_response += 1
                continue

            # --------------------------------------------------
            # Count interaction types
            # --------------------------------------------------

            if input_type == "speech":
                speech_examples += 1

            else:
                action_examples += 1

            # --------------------------------------------------
            # Conversation history
            # --------------------------------------------------

            history = []

            for j in range(i):

                previous_character = clean_text(
                    characters[j]
                )

                previous_speech = (
                    clean_text(speech[j])
                    if j < len(speech)
                    else ""
                )

                previous_action = (
                    clean_text(actions[j])
                    if j < len(actions)
                    else ""
                )

                previous_emote = (
                    clean_text(emotes[j])
                    if j < len(emotes)
                    else ""
                )

                if previous_speech:

                    history.append({
                        "speaker": previous_character,
                        "type": "speech",
                        "text": previous_speech
                    })

                if previous_action:

                    history.append({
                        "speaker": previous_character,
                        "type": "action",
                        "text": previous_action
                    })

                if previous_emote:

                    history.append({
                        "speaker": previous_character,
                        "type": "emote",
                        "text": previous_emote
                    })

            # --------------------------------------------------
            # Current world state
            # --------------------------------------------------

            current_objects = (
                room_objects[i]
                if i < len(room_objects)
                else []
            )

            current_agents = (
                room_agents[i]
                if i < len(room_agents)
                else []
            )

            current_available_actions = (
                available_actions[i]
                if i < len(available_actions)
                else []
            )

            current_carrying = (
                carrying[i]
                if i < len(carrying)
                else []
            )

            current_wearing = (
                wearing[i]
                if i < len(wearing)
                else []
            )

            current_wielding = (
                wielding[i]
                if i < len(wielding)
                else []
            )

            # --------------------------------------------------
            # Build final processed example
            # --------------------------------------------------

            example = {

                "record_id": record_idx,

                "turn_id": i,

                "input_character": get_character_info(
                    input_character,
                    persona_lookup
                ),

                "responder_character": get_character_info(
                    responder_character,
                    persona_lookup
                ),

                "setting": setting,

                "world_state": {

                    "context": context,

                    "room_objects": current_objects,

                    "room_agents": current_agents,

                    "object_descriptions": all_descriptions,

                    "carrying": current_carrying,

                    "wearing": current_wearing,

                    "wielding": current_wielding,

                    "available_actions":
                        current_available_actions
                },

                "history": history,

                "input": current_input,

                "input_type": input_type,

                "target": target
            }

            examples.append(example)

    # ----------------------------------------------------------
    # Write JSONL
    # ----------------------------------------------------------

    print("\nWriting processed dataset...")

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        for example in examples:

            f.write(
                json.dumps(
                    example,
                    ensure_ascii=False
                ) + "\n"
            )

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------

    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)

    print(
        f"\nProcessed examples: "
        f"{len(examples):,}"
    )

    print(
        f"Speech examples:    "
        f"{speech_examples:,}"
    )

    print(
        f"Action examples:    "
        f"{action_examples:,}"
    )

    print(
        f"Skipped examples:   "
        f"{skipped_no_response:,}"
    )

    print("\nSaved to:")
    print(OUTPUT_PATH)

    # ----------------------------------------------------------
    # Show first example
    # ----------------------------------------------------------

    print("\n" + "=" * 70)
    print("FIRST PROCESSED EXAMPLE")
    print("=" * 70)

    if examples:

        print(
            json.dumps(
                examples[0],
                indent=2,
                ensure_ascii=False
            )
        )

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()