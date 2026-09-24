import sys
from typing import TextIO


class NarrativeOutput:
    """Narrative Output component responsible for presenting the final generated narrative.

    This component sits at the output end of the narrative generation pipeline.
    It performs basic interface-level validation, formats narrative responses for presentation,
    and displays them in a CLI-compatible environment or returns them for integration with a GUI/engine.

    This component strictly does NOT:
    - modify GameState or call GameStateManager,
    - invoke RAG or KnowledgeRetriever,
    - invoke LLM models or perform inference,
    - evaluate state change validity or game rules.
    """

    def format(self, response: str) -> str:
        """Validate and format a raw narrative response string.

        Args:
            response: The narrative string to format.

        Returns:
            The formatted narrative string.

        Raises:
            TypeError: If response is not a string.
            ValueError: If response is empty or contains only whitespace.
        """
        if not isinstance(response, str):
            raise TypeError(
                f"response must be a string, got {type(response).__name__}"
            )

        formatted = response.strip()
        if not formatted:
            raise ValueError("response cannot be empty or whitespace-only")

        return formatted

    def display(
        self,
        response: str,
        stream: TextIO | None = None,
    ) -> str:
        """Format and display a narrative response to a stream (CLI or file).

        Args:
            response: The narrative string to display.
            stream: Target text stream for output. Defaults to sys.stdout if None.

        Returns:
            The formatted narrative string.

        Raises:
            TypeError: If response is not a string.
            ValueError: If response is empty or contains only whitespace.
        """
        formatted = self.format(response)

        target_stream = stream if stream is not None else sys.stdout
        target_stream.write(formatted + "\n")
        target_stream.flush()

        return formatted
