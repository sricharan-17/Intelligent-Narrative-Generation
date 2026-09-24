import re
from dataclasses import dataclass

from game.context.models import NarrativeContext


# Generation is capped at 128 new tokens (EVAL_GENERATION_MAX_NEW_TOKENS),
# roughly 500-600 characters of English text. The longest reference
# response in the test split is 516 characters. 1000 characters leaves
# generous headroom and only rejects runaway output.
MAX_RESPONSE_LENGTH = 1000

# Machine-readable error codes. Each error is reported as "<code>: <message>".
ERROR_INVALID_TYPE = "invalid_type"
ERROR_EMPTY_RESPONSE = "empty_response"
ERROR_PROMPT_LEAKAGE = "prompt_leakage"
ERROR_INPUT_ECHO = "input_echo"
ERROR_SPEAKER_PREFIX = "speaker_prefix"
ERROR_TOO_LONG = "too_long"

# The training prompt marks sections with "### <NAME>" headers
# (e.g. "### HISTORY", "### RESPONSE"). Any "###" followed by
# header text indicates the model continued the prompt template.
_PROMPT_HEADER_PATTERN = re.compile(r"###[ \t]+\S")


@dataclass(frozen=True)
class ResponseValidationResult:
    """Outcome of validating a generated narrative response."""

    is_valid: bool
    response: str
    errors: tuple[str, ...]


class ResponseValidator:
    """Performs deterministic checks on plain-text generated responses.

    This component does not:
    - modify the response,
    - modify NarrativeContext or GameState,
    - check whether the response is consistent with GameState or RAG,
    - call the LLM or any other model.
    """

    def validate(
        self,
        response: str,
        context: NarrativeContext,
    ) -> ResponseValidationResult:
        """Validate a generated response against the current context."""

        # A non-string response is reported as "" so that the result's
        # response field is always a string.
        if not isinstance(response, str):
            return ResponseValidationResult(
                is_valid=False,
                response="",
                errors=(
                    f"{ERROR_INVALID_TYPE}: response must be a string, "
                    f"got {type(response).__name__}",
                ),
            )

        if not response.strip():
            return ResponseValidationResult(
                is_valid=False,
                response=response,
                errors=(
                    f"{ERROR_EMPTY_RESPONSE}: response cannot be empty "
                    "or whitespace-only",
                ),
            )

        errors: list[str] = []
        interaction = context.interaction

        if _PROMPT_HEADER_PATTERN.search(response):
            errors.append(
                f"{ERROR_PROMPT_LEAKAGE}: response contains a "
                "'###' prompt section header"
            )

        if self._normalize(response) == self._normalize(interaction.input):
            errors.append(
                f"{ERROR_INPUT_ECHO}: response repeats the player input"
            )

        if self._has_speaker_prefix(
            response,
            interaction.responder_character,
        ):
            errors.append(
                f"{ERROR_SPEAKER_PREFIX}: response starts with the "
                "responder's name as a speaker prefix"
            )

        if len(response) > MAX_RESPONSE_LENGTH:
            errors.append(
                f"{ERROR_TOO_LONG}: response length {len(response)} "
                f"exceeds {MAX_RESPONSE_LENGTH} characters"
            )

        return ResponseValidationResult(
            is_valid=not errors,
            response=response,
            errors=tuple(errors),
        )

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(text.split()).casefold()

    @staticmethod
    def _has_speaker_prefix(
        response: str,
        responder_character: str | None,
    ) -> bool:
        """Detect a leading history-style prefix like 'Guard:' or 'Guard [speech]:'."""

        # World/object interactions have no responder character.
        if responder_character is None:
            return False

        name = responder_character.strip()
        if not name:
            return False

        pattern = re.compile(
            rf"^\s*{re.escape(name)}\s*(\[[^\]\n]*\])?\s*:",
            re.IGNORECASE,
        )

        return pattern.match(response) is not None
