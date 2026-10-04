from game.context.models import NarrativeContext


# Keep this aligned with the system instruction used during SmolLM2 training.
SYSTEM_PROMPT = (
    "You are an interactive fantasy game narrator. "
    "Generate the response of the responder character based on "
    "the current world state, character personas, available actions, "
    "conversation history, and current player input."
)

NARRATOR_SYSTEM_PROMPT = (
    "You are an interactive fantasy game narrator. "
    "Describe what happens in the game world based on "
    "the current world state, available actions, conversation history, "
    "and current player input."
)


class NarrativePromptBuilder:

    def build_messages(
        self,
        context: NarrativeContext,
    ) -> list[dict[str, str]]:

        interaction = context.interaction
        state = context.game_state
        knowledge = context.retrieved_knowledge

        has_responder = interaction.responder_character is not None

        sections = []

        # --------------------------------------------------
        # Setting
        # --------------------------------------------------

        sections.append("### SETTING")

        location = state.location

        sections.append(
            f"Name: {location.name if location else ''}"
        )

        category = ""
        background = ""

        if isinstance(state.setting, dict):
            category = state.setting.get("category", "")
            background = state.setting.get("background", "")

        sections.append(f"Category: {category}")

        sections.append(
            f"Description: {location.description if location else ''}"
        )

        sections.append(f"Background: {background}")

        # --------------------------------------------------
        # Input character
        # --------------------------------------------------

        sections.append("\n### INPUT CHARACTER")

        input_character = self._find_character(
            state,
            interaction.input_character,
        )

        sections.append(
            f"Name: {interaction.input_character or ''}"
        )

        sections.append(
            f"Persona:\n"
            f"{input_character.description if input_character else ''}"
        )

        # --------------------------------------------------
        # Responder character
        # --------------------------------------------------

        if has_responder:

            sections.append("\n### RESPONDING CHARACTER")

            responder_character = self._find_character(
                state,
                interaction.responder_character,
            )

            sections.append(
                f"Name: {interaction.responder_character or ''}"
            )

            sections.append(
                f"Persona:\n"
                f"{responder_character.description if responder_character else ''}"
            )

        # --------------------------------------------------
        # World state
        # --------------------------------------------------

        sections.append("\n### WORLD STATE")

        # Context
        context_lines = []

        if location:
            context_lines.append(
                f"You are in the {location.name}."
            )

            if location.description:
                context_lines.append(
                    location.description
                )

        if context_lines:
            sections.append(
                "Context:\n" + "\n".join(context_lines)
            )
        else:
            sections.append("Context:")

        # Room objects
        present_objects = [
            obj
            for obj in state.objects
            if obj.location is None
            or (
                location is not None
                and obj.location == location.name
            )
        ]

        object_names = [
            obj.name
            for obj in present_objects
        ]

        sections.append(
            "Room objects: "
            + (", ".join(object_names) if object_names else "None")
        )

        # Room agents
        room_agents = [
            character.name
            for character in state.characters
            if character.present
        ]

        sections.append(
            "Room agents: "
            + (", ".join(room_agents) if room_agents else "None")
        )

        # Object descriptions
        if present_objects:

            sections.append("Object descriptions:")

            for obj in present_objects:

                description = (
                    obj.description
                    if obj.description
                    else "No description"
                )

                sections.append(
                    f"- {obj.name}: {description}"
                )

                if obj.state:
                    sections.append(
                        f"  State: {obj.state}"
                    )

        else:
            sections.append(
                "Object descriptions: None"
            )

        # Inventory / carrying
        sections.append(
            "Carrying: "
            + (
                ", ".join(state.inventory)
                if state.inventory
                else "None"
            )
        )

        # These fields are not currently represented by GameState.
        sections.append("Wearing: None")
        sections.append("Wielding: None")

        # Available actions
        if state.available_actions:

            sections.append("Available actions:")

            for action in state.available_actions:
                sections.append(
                    f"- {action}"
                )

        else:
            sections.append(
                "Available actions: None"
            )

        # --------------------------------------------------
        # History
        # --------------------------------------------------

        sections.append("\n### HISTORY")

        if context.history:

            for entry in context.history:

                sections.append(
                    f"{entry.speaker} "
                    f"[{entry.entry_type}]: "
                    f"{entry.text}"
                )

        else:
            sections.append("None")

        # --------------------------------------------------
        # Retrieved knowledge
        # --------------------------------------------------

        sections.append("\n### RETRIEVED KNOWLEDGE")

        if knowledge.results:

            for result in knowledge.results:

                title = (
                    result.title
                    or result.document_id
                )

                sections.append(
                    f"- {title} "
                    f"[score={result.score:.3f}]"
                )

                sections.append(
                    f"  {result.content}"
                )

        else:
            sections.append("None")

        # --------------------------------------------------
        # Current player input
        # --------------------------------------------------

        sections.append("\n### PLAYER INPUT")

        sections.append(
            interaction.input
        )

        # --------------------------------------------------
        # Input type
        # --------------------------------------------------

        sections.append("\n### INPUT TYPE")

        sections.append(
            interaction.input_type
        )

        # --------------------------------------------------
        # Generation boundary
        # --------------------------------------------------

        sections.append("\n### RESPONSE")

        user_prompt = "\n".join(sections)

        if has_responder:
            system_prompt = SYSTEM_PROMPT
        else:
            system_prompt = NARRATOR_SYSTEM_PROMPT

        return [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

    # ------------------------------------------------------
    # Character lookup
    # ------------------------------------------------------

    @staticmethod
    def _find_character(state, name):

        if not name:
            return None

        name_lower = name.strip().lower()

        for character in state.characters:

            if character.name.strip().lower() == name_lower:
                return character

        return None