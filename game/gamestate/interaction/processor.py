from game.gamestate.interaction.models import Interaction


class InputInteractionProcessor:
    """Converts player input into the project's Interaction model."""

    SUPPORTED_INPUT_TYPES = frozenset({"action", "speech"})

    def process(
        self,
        input_text: str,
        input_type: str,
        input_character: str,
        responder_character: str | None = None,
    ) -> Interaction:
        """Create an Interaction from player input.

        This component does not:
        - validate whether an action is available,
        - modify GameState,
        - call RAG,
        - call the LLM,
        - create StateChangeAction objects.
        """

        if not isinstance(input_text, str):
            raise TypeError("input_text must be a string")

        if not isinstance(input_type, str):
            raise TypeError("input_type must be a string")

        if not isinstance(input_character, str):
            raise TypeError("input_character must be a string")

        # None means no character responds (e.g. a world/object action).
        if responder_character is not None and not isinstance(
            responder_character, str
        ):
            raise TypeError("responder_character must be a string or None")

        normalized_input = " ".join(input_text.split())
        normalized_type = input_type.strip().lower()
        normalized_input_character = input_character.strip()
        normalized_responder_character = (
            responder_character.strip()
            if responder_character is not None
            else None
        )

        if not normalized_input:
            raise ValueError("input_text cannot be empty")

        if normalized_type not in self.SUPPORTED_INPUT_TYPES:
            raise ValueError(
                f"Unsupported input_type: {input_type!r}. "
                "Expected 'action' or 'speech'."
            )

        if not normalized_input_character:
            raise ValueError("input_character cannot be empty")

        # A supplied responder must be a real name; use None for no responder.
        if (
            normalized_responder_character is not None
            and not normalized_responder_character
        ):
            raise ValueError("responder_character cannot be empty")

        return Interaction(
            input=normalized_input,
            input_type=normalized_type,
            input_character=normalized_input_character,
            responder_character=normalized_responder_character,
        )