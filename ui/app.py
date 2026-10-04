"""Gradio web UI for The Ashen Crown prototype.

Usage (from the project root, using the project's .venv):

    python ui/app.py            # http://127.0.0.1:7860
    python ui/app.py --port 7861

The UI is only a front-end. Every turn goes through the existing backend:

    run_prototype_turn -> NarrativePipeline -> InputInteractionProcessor
    -> ContextBuilder (+ RAG) -> NarrativeGenerator (NarrativePromptBuilder
    + remote fine-tuned SmolLM2 Space) -> ResponseValidator -> StateUpdater

The conversation shown in the UI is read from GameState.history, so the
game state stays the single source of truth.
"""

import argparse
import sys
from pathlib import Path

# Allow "python ui/app.py" as well as "python -m ui.app".
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr  # noqa: E402

from game.gamestate.interaction.actions import StateChangeAction  # noqa: E402
from game.gamestate.state.manager import GameStateManager  # noqa: E402
from game.gamestate.state.models import GameState  # noqa: E402
from game.pipeline.pipeline import NarrativePipeline, PipelineResult  # noqa: E402
from game.world.prototype import (  # noqa: E402
    LOCATIONS,
    PLAYER_CHARACTER,
    PROTOTYPE_ACTIONS,
    PROTOTYPE_TITLE,
    create_prototype_manager,
    create_prototype_pipeline,
    run_prototype_turn,
)


INPUT_TYPES = ["speech", "action"]

# Dropdown label used when no character responds (world/object actions).
NO_RESPONDER = "None (narrator describes the result)"

NO_TURN_YET = "_No turn has been played yet._"

HOW_TO_PLAY = f"""
You play **{PLAYER_CHARACTER}**.

- **Speech**: say something to a character who is here. Pick them as
  the responder character.
- **Action**: describe what you do. The narrator describes the result,
  or pick a character who should react.

These actions also change the game state:
{", ".join(f"`{command}`" for command in PROTOTYPE_ACTIONS)}, and
`travel to <place>` ({", ".join(LOCATIONS)}).

Every response comes from the fine-tuned SmolLM2 model, so the
wording changes each time.
"""


# ----------------------------------------------------------------------
# Rendering helpers: read-only views of GameState
# ----------------------------------------------------------------------


def present_npcs(state: GameState) -> list[str]:
    """Names of the non-player characters at the current location."""

    return [
        character.name
        for character in state.characters
        if character.present and character.name != PLAYER_CHARACTER
    ]


def responder_dropdown(
    state: GameState,
    input_type: str,
    current: str | None = None,
) -> gr.Dropdown:
    """Responder choices for the current location.

    Keeps the current choice when it is still valid. Otherwise speech
    defaults to the first character here and actions default to the
    narrator.
    """

    npcs = present_npcs(state)
    choices = npcs + [NO_RESPONDER]

    if current in choices and not (
        input_type == "speech" and current == NO_RESPONDER
    ):
        value = current
    elif input_type == "speech" and npcs:
        value = npcs[0]
    else:
        value = NO_RESPONDER

    return gr.Dropdown(choices=choices, value=value)


def render_location(state: GameState) -> str:
    location = state.location
    name = location.name if location else "Unknown"
    description = location.description if location else ""
    world = state.setting.get("name", "")

    return (
        f"## 📍 {name}\n"
        f"{description}\n\n"
        f"_Turn {state.turn}"
        + (f" · {world}" if world else "")
        + "_"
    )


def _format_object_state(object_state: dict) -> str:
    parts = []

    for key, value in object_state.items():
        if isinstance(value, bool):
            value = "yes" if value else "no"
        parts.append(f"{key}: {value}")

    return ", ".join(parts)


def render_world_state(state: GameState) -> str:
    location_name = state.location.name if state.location else None

    characters = [f"{PLAYER_CHARACTER} (you)"] + present_npcs(state)

    # Same visibility rule the prompt builder uses: objects without a
    # location, or objects at the current location.
    objects = [
        obj
        for obj in state.objects
        if obj.location is None or obj.location == location_name
    ]

    lines = ["### World state", f"**Characters here:** {', '.join(characters)}"]

    lines.append("\n**Objects here:**")
    if objects:
        for obj in objects:
            line = f"- **{obj.name}**: {obj.description}"
            if obj.state:
                line += f" _({_format_object_state(obj.state)})_"
            lines.append(line)
    else:
        lines.append("- none")

    lines.append(
        f"\n**Inventory:** {', '.join(state.inventory) or 'empty'}"
    )
    lines.append(
        f"\n**Available actions:** {', '.join(state.available_actions)}"
    )

    other_places = [name for name in LOCATIONS if name != location_name]
    lines.append(f"\n**You can travel to:** {', '.join(other_places)}")

    return "\n".join(lines)


def render_history(state: GameState) -> list[dict[str, str]]:
    """Convert GameState.history into Gradio chat messages."""

    messages = []

    for entry in state.history:
        if entry.speaker == PLAYER_CHARACTER:
            messages.append(
                {
                    "role": "user",
                    "content": f"_[{entry.entry_type}]_ {entry.text}",
                }
            )
        else:
            messages.append(
                {
                    "role": "assistant",
                    "content": f"**{entry.speaker}:** {entry.text}",
                }
            )

    return messages


# ----------------------------------------------------------------------
# Turn handling
# ----------------------------------------------------------------------


def describe_change(change: StateChangeAction) -> str:
    if change.action_type == "inventory_change":
        parts = [f"+{item}" for item in change.inventory_add]
        parts += [f"-{item}" for item in change.inventory_remove]
        return f"inventory {' '.join(parts)}"

    return f"{change.target} {_format_object_state(change.changes)}"


def render_pipeline_details(
    result: PipelineResult,
    pipeline: NarrativePipeline,
) -> str:
    """Show what each pipeline stage produced for the last turn."""

    interaction = result.interaction
    lines = [
        "**Interaction**",
        f"- input type: `{interaction.input_type}`",
        f"- input character: `{interaction.input_character}`",
        f"- responder character: `{interaction.responder_character}`",
        "",
        "**Retrieved knowledge (RAG)**",
    ]

    knowledge = result.context.retrieved_knowledge.results
    if knowledge:
        for item in knowledge:
            title = item.title or item.document_id
            lines.append(f"- {title} (score {item.score:.3f})")
    else:
        lines.append("- none")

    lines += [
        "",
        "**Raw model response**",
        f"> {result.raw_response or '(empty)'}",
        "",
        "**Response validation:** "
        + (
            "passed"
            if result.validation.is_valid
            else "; ".join(result.validation.errors)
        ),
    ]

    # The real NarrativeGenerator can show the exact chat-formatted
    # prompt it sent. Rebuilding it here is deterministic and reuses the
    # generator's own NarrativePromptBuilder.
    build_prompt = getattr(pipeline.generator, "build_prompt", None)
    if build_prompt is not None:
        try:
            prompt = build_prompt(result.context)
        except Exception as error:  # diagnostics only, never fatal
            prompt = f"(could not rebuild the prompt: {error})"

        lines += ["", "**Prompt sent to the model**", "```text", prompt, "```"]

    return "\n".join(lines)


def play_turn(
    pipeline: NarrativePipeline,
    manager: GameStateManager,
    input_text: str,
    input_type: str,
    responder_choice: str | None,
) -> tuple[str, str, bool]:
    """Run one turn through the real backend.

    Returns (status markdown, pipeline details markdown, success).
    GameState is only changed by the pipeline itself.
    """

    if not input_text or not input_text.strip():
        return "⚠️ Please type something first.", NO_TURN_YET, False

    responder = None if responder_choice in (None, NO_RESPONDER) else (
        responder_choice
    )

    if input_type == "speech" and responder is None:
        return (
            "⚠️ Speech needs someone to talk to. Choose a responder "
            "character, or switch the input type to **action**.",
            NO_TURN_YET,
            False,
        )

    location_before = manager.get_state().location

    try:
        result = run_prototype_turn(
            pipeline,
            manager,
            input_text,
            input_type,
            responder,
        )
    except Exception as error:
        # Interaction errors (TypeError/ValueError) and remote API errors
        # both end up here. The pipeline raises before StateUpdater runs,
        # so the game state is unchanged.
        return (
            "❌ **The turn failed. The game state was not changed.**\n\n"
            f"`{type(error).__name__}: {error}`\n\n"
            "If this is a connection or API error, the Hugging Face Space "
            "may be asleep or restarting. Wait a minute and try again, "
            "and check that HF_SPACE_TOKEN is set in the .env file.",
            NO_TURN_YET,
            False,
        )

    details = render_pipeline_details(result, pipeline)

    if not result.validation.is_valid:
        errors = "\n".join(f"- {error}" for error in result.validation.errors)
        return (
            "⚠️ **The model's response was rejected by the response "
            "validator**, so it was not added to the story and the game "
            "state was not changed. Responses are sampled, so trying "
            f"again usually works.\n\n{errors}",
            details,
            False,
        )

    if not result.success:
        rejected = "\n".join(
            f"- {describe_change(change)}"
            for change in result.state_update.rejected_changes
        )
        return (
            "🚫 **That could not happen.** The game state was not "
            f"changed.\n\nRejected state changes:\n{rejected}",
            details,
            False,
        )

    state = manager.get_state()
    status = [f"✅ Turn {state.turn - 1} complete."]

    for change in result.state_update.applied_changes:
        status.append(f"- state: {describe_change(change)}")

    if state.location != location_before and state.location is not None:
        status.append(f"- travelled to **{state.location.name}**")

    return "\n".join(status), details, True


# ----------------------------------------------------------------------
# Gradio app
# ----------------------------------------------------------------------


def create_app(pipeline: NarrativePipeline) -> gr.Blocks:
    """Build the Gradio interface around an existing NarrativePipeline."""

    model_info = ""
    status_fn = getattr(pipeline.generator, "status", None)
    if status_fn is not None:
        info = status_fn()
        model_info = (
            f"Model: `{info.get('model')}` (fine-tuned) via Hugging Face "
            f"Space `{info.get('space')}`"
        )

    def views(manager, input_type, responder_choice=None):
        state = manager.get_state()
        return (
            render_history(state),
            render_location(state),
            render_world_state(state),
            responder_dropdown(state, input_type, responder_choice),
        )

    def start_session():
        manager = create_prototype_manager()
        return (manager, *views(manager, "speech"), "", NO_TURN_YET)

    def on_input_type_change(manager, input_type):
        if manager is None:
            manager = create_prototype_manager()
        # Reset the responder to the default for the new input type.
        return responder_dropdown(manager.get_state(), input_type)

    def lock_controls():
        return (
            gr.Button(value="Generating...", interactive=False),
            "⏳ Generating a response with the remote SmolLM2 model...",
        )

    def unlock_controls():
        return gr.Button(value="Generate Response", interactive=True)

    def on_generate(manager, input_text, input_type, responder_choice):
        if manager is None:
            manager = create_prototype_manager()

        status, details, success = play_turn(
            pipeline,
            manager,
            input_text,
            input_type,
            responder_choice,
        )

        return (
            manager,
            *views(manager, input_type, responder_choice),
            status,
            details,
            # Clear the input box only when the turn succeeded.
            gr.Textbox(value="") if success else gr.Textbox(),
        )

    with gr.Blocks(title=PROTOTYPE_TITLE) as app:
        # One GameStateManager per browser session.
        manager_state = gr.State(None)

        gr.Markdown(
            f"# {PROTOTYPE_TITLE}\n"
            "Intelligent narrative generation using context-aware "
            "retrieval and an adapted large language model.  \n"
            f"{model_info}"
        )

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="Story so far", height=460)
                status = gr.Markdown()

                with gr.Row():
                    input_type = gr.Radio(
                        INPUT_TYPES,
                        value="speech",
                        label="Input type",
                    )
                    gr.Textbox(
                        value=PLAYER_CHARACTER,
                        label="Input character (you)",
                        interactive=False,
                    )
                    responder = gr.Dropdown(
                        label="Responder character",
                        choices=[],
                        interactive=True,
                    )

                user_input = gr.Textbox(
                    label="Your input",
                    placeholder=(
                        "e.g. speech: 'Have you heard any rumors about "
                        "Blackstone Castle?'  action: 'open letter'"
                    ),
                    lines=2,
                    max_lines=4,
                )

                with gr.Row():
                    generate_button = gr.Button(
                        "Generate Response",
                        variant="primary",
                    )
                    restart_button = gr.Button("Restart story")

                with gr.Accordion("Pipeline details (last turn)", open=False):
                    details = gr.Markdown(NO_TURN_YET)

            with gr.Column(scale=2):
                location = gr.Markdown()
                world_state = gr.Markdown()

                with gr.Accordion("How to play", open=False):
                    gr.Markdown(HOW_TO_PLAY)

        session_outputs = [
            manager_state,
            chatbot,
            location,
            world_state,
            responder,
            status,
            details,
        ]

        app.load(start_session, None, session_outputs)
        restart_button.click(start_session, None, session_outputs)

        input_type.change(
            on_input_type_change,
            [manager_state, input_type],
            responder,
        )

        turn_inputs = [manager_state, user_input, input_type, responder]
        turn_outputs = session_outputs + [user_input]

        for trigger in (generate_button.click, user_input.submit):
            trigger(
                lock_controls,
                None,
                [generate_button, status],
                queue=False,
            ).then(
                on_generate,
                turn_inputs,
                turn_outputs,
                show_progress_on=[chatbot],
            ).then(
                unlock_controls,
                None,
                generate_button,
            )

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description=f"Play {PROTOTYPE_TITLE}.")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--share",
        action="store_true",
        help="create a temporary public Gradio link",
    )
    args = parser.parse_args()

    # Only the launcher imports llm; game/ stays independent of the model.
    from llm.generator import NarrativeGenerator

    generator = NarrativeGenerator()
    print("Connecting to the remote fine-tuned SmolLM2 model...")
    try:
        generator.load()
    except Exception as error:
        print(f"Could not connect to the remote model: {error}")
        return 1
    print(f"Model ready: {generator.status()}")

    app = create_app(create_prototype_pipeline(generator))
    app.launch(
        server_name="127.0.0.1",
        server_port=args.port,
        share=args.share,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
