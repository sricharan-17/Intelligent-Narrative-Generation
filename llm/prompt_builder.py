from game.context.models import NarrativeContext


SYSTEM_PROMPT = (
    "You are the responder character in an interactive game narrative. "
    "Generate the next response while respecting the game state, "
    "character persona, conversation history, and current input."
)


class NarrativePromptBuilder:
    """Converts NarrativeContext into the prompt used by the fine-tuned LLM."""

    def build_messages(self, context: NarrativeContext) -> list[dict[str, str]]:
        interaction = context.interaction
        state = context.game_state
        knowledge = context.retrieved_knowledge

        sections: list[str] = []

        sections.append(
            "TASK\n"
            "Generate exactly one natural next response from the RESPONDER CHARACTER.\n"
            "Do not respond as the INPUT CHARACTER.\n"
            "Use the conversation history and current input to determine the next response.\n"
            "Stay consistent with the RESPONDER CHARACTER's persona, role, and situation.\n"
            "Return only the response itself."
        )

        sections.append(
            "INPUT TYPE INSTRUCTIONS\n"
            "If the input type is speech, generate what the responder should say next.\n"
            "If the input type is action, generate the responder's appropriate next response to that action."
        )

        sections.append(self._format_setting(state))
        sections.append(self._format_input_character(interaction, state))
        sections.append(self._format_responder(interaction, state))
        sections.append(self._format_world_state(state))
        sections.append(self._format_history(context))
        sections.append(self._format_knowledge(knowledge))

        sections.append(
            "CURRENT INPUT\n"
            f"{interaction.input}"
        )

        sections.append(
            "INPUT TYPE\n"
            f"{interaction.input_type}"
        )

        user_prompt = "\n\n".join(sections)

        return [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

    @staticmethod
    def _format_setting(state) -> str:
        if state.location is None:
            return "SETTING\nUnknown"

        lines = [f"Location: {state.location.name}"]

        if state.location.description:
            lines.append(f"Description: {state.location.description}")

        if state.setting:
            lines.append(f"Setting data: {state.setting}")

        return "SETTING\n" + "\n".join(lines)

    @staticmethod
    def _find_character(state, name):
        if not name:
            return None

        name_lower = name.strip().lower()

        for character in state.characters:
            if character.name.strip().lower() == name_lower:
                return character

        return None

    def _format_input_character(self, interaction, state) -> str:
        character = self._find_character(
            state,
            interaction.input_character,
        )

        lines = [
            "INPUT CHARACTER",
            interaction.input_character or "Unknown",
        ]

        if character is not None and character.description:
            lines.append(f"Persona/Description: {character.description}")

        return "\n".join(lines)

    def _format_responder(self, interaction, state) -> str:
        name = interaction.responder_character

        lines = [
            "RESPONDER CHARACTER",
            name or "Unknown",
        ]

        character = self._find_character(state, name)

        if character is not None and character.description:
            lines.append(f"Persona/Description: {character.description}")

        return "\n".join(lines)

    @staticmethod
    def _format_world_state(state) -> str:
        lines = ["WORLD STATE"]

        if state.characters:
            lines.append("Characters:")
            for character in state.characters:
                status = "present" if character.present else "absent"
                lines.append(f"- {character.name} ({status})")

        if state.objects:
            lines.append("Objects:")
            for obj in state.objects:
                description = obj.description or "No description"
                lines.append(f"- {obj.name}: {description}")

                if obj.state:
                    lines.append(f"  State: {obj.state}")

        if state.inventory:
            lines.append(f"Inventory: {state.inventory}")

        if state.relationships:
            lines.append(f"Relationships: {state.relationships}")

        if state.available_actions:
            lines.append(
                "Available actions: "
                + ", ".join(state.available_actions)
            )

        if state.recent_events:
            lines.append("Recent events:")
            for event in state.recent_events:
                lines.append(
                    f"- Turn {event.turn}: "
                    f"{event.event_type}: {event.description}"
                )

        lines.append(f"Current turn: {state.turn}")

        return "\n".join(lines)

    @staticmethod
    def _format_history(context: NarrativeContext) -> str:
        lines = ["CONVERSATION HISTORY"]

        if not context.history:
            lines.append("(none)")
            return "\n".join(lines)

        for entry in context.history:
            lines.append(
                f"{entry.speaker} [{entry.entry_type}]: {entry.text}"
            )

        return "\n".join(lines)

    @staticmethod
    def _format_knowledge(retrieved_knowledge) -> str:
        lines = ["RETRIEVED KNOWLEDGE"]

        if not retrieved_knowledge.results:
            lines.append("(none)")
            return "\n".join(lines)

        for result in retrieved_knowledge.results:
            lines.append(
                f"- {result.title or result.document_id}"
                f" [score={result.score:.3f}]"
            )
            lines.append(f"  {result.content}")

        return "\n".join(lines)
