from dataclasses import dataclass
from typing import Protocol

from game.context.builder import ContextBuilder
from game.context.models import NarrativeContext
from game.gamestate.interaction.actions import StateChangeAction
from game.gamestate.interaction.models import Interaction
from game.gamestate.interaction.processor import InputInteractionProcessor
from game.gamestate.state.manager import GameStateManager
from game.gamestate.update.updater import StateUpdater, StateUpdateResult
from game.output.narrative import NarrativeOutput
from game.response.validator import (
    ResponseValidationResult,
    ResponseValidator,
)


class ResponseGenerator(Protocol):
    """Anything that turns a NarrativeContext into plain response text.

    llm.NarrativeGenerator satisfies this; tests use a fake generator so
    the model never has to be loaded.
    """

    def generate(self, context: NarrativeContext) -> str:
        ...


@dataclass(frozen=True)
class PipelineResult:
    """Everything produced while handling one player interaction.

    state_update and output are None when the generated response was
    rejected, because the pipeline stops before updating GameState.
    output is also None when the state update itself was rejected.
    """

    interaction: Interaction
    context: NarrativeContext
    raw_response: str
    validation: ResponseValidationResult
    state_update: StateUpdateResult | None
    output: str | None

    @property
    def success(self) -> bool:
        return (
            self.validation.is_valid
            and self.state_update is not None
            and self.state_update.success
        )


class NarrativePipeline:
    """Runs one player interaction through every narrative stage.

    Input -> Interaction -> NarrativeContext -> generated response
    -> validation -> state update -> formatted output.

    GameState is only modified by StateUpdater, and only after the
    generated response has passed validation. Exceptions raised by any
    stage propagate unchanged; no stage before StateUpdater modifies
    GameState, so a failure leaves the state untouched.
    """

    def __init__(
        self,
        context_builder: ContextBuilder,
        generator: ResponseGenerator,
        interaction_processor: InputInteractionProcessor | None = None,
        response_validator: ResponseValidator | None = None,
        state_updater: StateUpdater | None = None,
        output: NarrativeOutput | None = None,
    ):
        self.context_builder = context_builder
        self.generator = generator
        self.interaction_processor = (
            interaction_processor
            if interaction_processor is not None
            else InputInteractionProcessor()
        )
        self.response_validator = (
            response_validator
            if response_validator is not None
            else ResponseValidator()
        )
        self.state_updater = (
            state_updater if state_updater is not None else StateUpdater()
        )
        self.output = output if output is not None else NarrativeOutput()

    def run(
        self,
        state_manager: GameStateManager,
        input_text: str,
        input_type: str,
        input_character: str,
        responder_character: str | None = None,
        state_changes: tuple[StateChangeAction, ...] = (),
    ) -> PipelineResult:
        """Handle one player interaction.

        state_changes are explicit StateChangeAction objects supplied by
        the caller. They are never inferred from the generated text.
        """

        # Raises TypeError/ValueError for invalid player input.
        interaction = self.interaction_processor.process(
            input_text,
            input_type,
            input_character,
            responder_character,
        )

        context = self.context_builder.build(interaction, state_manager)

        raw_response = self.generator.generate(context)

        validation = self.response_validator.validate(raw_response, context)

        if not validation.is_valid:
            return PipelineResult(
                interaction=interaction,
                context=context,
                raw_response=raw_response,
                validation=validation,
                state_update=None,
                output=None,
            )

        state_update = self.state_updater.apply(
            interaction,
            validation.response,
            state_manager,
            state_changes,
        )

        output = None
        if state_update.success:
            output = self.output.format(validation.response)

        return PipelineResult(
            interaction=interaction,
            context=context,
            raw_response=raw_response,
            validation=validation,
            state_update=state_update,
            output=output,
        )
