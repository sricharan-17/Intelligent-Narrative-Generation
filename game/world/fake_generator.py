from game.context.models import NarrativeContext


class PrototypeFakeGenerator:
    """Deterministic stand-in for llm.NarrativeGenerator.

    Implements the pipeline's generate(context) -> str interface without
    loading a model. The response quotes the first sentence of the most
    relevant retrieved knowledge, so it also shows what RAG returned.
    """

    def generate(self, context: NarrativeContext) -> str:
        interaction = context.interaction
        lore = self._lore(context)

        if interaction.responder_character is None:
            location = (
                context.game_state.location.name
                if context.game_state.location is not None
                else "the world"
            )
            response = (
                f"{interaction.input_character} acts. "
                f"Around them, {location} grows quiet."
            )
            return f"{response} {lore}" if lore else response

        response = (
            f"{interaction.responder_character} studies "
            f"{interaction.input_character} before answering."
        )
        return f'{response} "{lore}"' if lore else response

    @staticmethod
    def _lore(context: NarrativeContext) -> str:
        """First sentence of the top result about something other than the
        current location or the characters taking part."""

        results = context.retrieved_knowledge.results
        if not results:
            return ""

        location = context.game_state.location
        interaction = context.interaction
        skipped = {
            interaction.input_character,
            interaction.responder_character,
            location.name if location is not None else None,
        }

        chosen = next(
            (result for result in results if result.entity_id not in skipped),
            results[0],
        )

        sentence = chosen.content.split(". ", 1)[0].strip()
        return sentence if sentence.endswith(".") else f"{sentence}."
