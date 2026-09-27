"""The Ashen Crown: a small playable prototype world.

Builds the initial GameState, the prototype RAG retriever, and the
explicit state changes for a handful of prototype actions. State changes
are decided here from the player's command, never from generated text.
"""

from pathlib import Path

from game.context.builder import ContextBuilder
from game.gamestate.interaction.actions import StateChangeAction
from game.gamestate.state.manager import GameStateManager
from game.gamestate.state.models import (
    Character,
    GameObject,
    GameState,
    Location,
)
from game.pipeline.pipeline import (
    NarrativePipeline,
    PipelineResult,
    ResponseGenerator,
)
from rag.indexing.index import KnowledgeIndex
from rag.knowledge.loader import load_knowledge_entries_from_json
from rag.retrieval.retriever import KnowledgeRetriever


PROTOTYPE_TITLE = "The Ashen Crown"
PLAYER_CHARACTER = "Kael"

KNOWLEDGE_FILE = (
    Path(__file__).resolve().parent / "data" / "ashen_crown_knowledge.json"
)

# The retriever always boosts the current location's and the responder's
# documents, which take two result slots; allow one more than the
# retriever default (3) so topic-specific lore still fits.
KNOWLEDGE_TOP_K = 4

LOCATIONS: dict[str, Location] = {
    "Ravenmoor": Location(
        "Ravenmoor",
        "A small village surrounded by dense woodland.",
    ),
    "Old Tavern": Location(
        "Old Tavern",
        "A weathered tavern in Ravenmoor with scarred wooden tables "
        "and warm lanterns.",
    ),
    "Whispering Forest": Location(
        "Whispering Forest",
        "A dense forest surrounding Ravenmoor.",
    ),
    "Blackstone Castle": Location(
        "Blackstone Castle",
        "A ruined stone fortress overlooking the surrounding woodland.",
    ),
}

STARTING_LOCATION = "Old Tavern"

# Where each non-player character can be found. Travelling updates their
# presence through GameStateManager.set_character_presence.
CHARACTER_LOCATIONS: dict[str, str] = {
    "Mira": "Old Tavern",
    "Elara": "Old Tavern",
    "Captain Rowan": "Ravenmoor",
    "The Warden": "Blackstone Castle",
}


def _discover(target: str) -> StateChangeAction:
    return StateChangeAction(
        action_type="object_state_change",
        target=target,
        changes={"discovered": True},
    )


def _take(item: str) -> StateChangeAction:
    return StateChangeAction(
        action_type="inventory_change",
        inventory_add=(item,),
    )


# Prototype command -> (required location, explicit state changes).
# Only the existing object_state_change and inventory_change types are used.
PROTOTYPE_ACTIONS: dict[str, tuple[str, tuple[StateChangeAction, ...]]] = {
    "open letter": (
        "Old Tavern",
        (
            StateChangeAction(
                action_type="object_state_change",
                target="Sealed Letter",
                changes={"opened": True},
            ),
        ),
    ),
    "inspect ledger": (
        "Old Tavern",
        (
            StateChangeAction(
                action_type="object_state_change",
                target="Tavern Ledger",
                changes={"examined": True},
            ),
        ),
    ),
    "open chest": (
        "Old Tavern",
        (
            StateChangeAction(
                action_type="object_state_change",
                target="Hidden Chest",
                changes={"opened": True},
            ),
        ),
    ),
    "take key": (
        "Old Tavern",
        (_discover("Ancient Key"), _take("Ancient Key")),
    ),
    "take amulet": (
        "Whispering Forest",
        (_discover("Royal Amulet"), _take("Royal Amulet")),
    ),
    # The key is left in the lock. Removing it from the inventory also means
    # the whole change is rejected atomically if Kael does not carry it.
    "unlock gate": (
        "Blackstone Castle",
        (
            StateChangeAction(
                action_type="inventory_change",
                inventory_remove=("Ancient Key",),
            ),
            StateChangeAction(
                action_type="object_state_change",
                target="Castle Gate",
                changes={"locked": False},
            ),
        ),
    ),
    "take crown": (
        "Blackstone Castle",
        (_discover("Ashen Crown"), _take("Ashen Crown")),
    ),
}

# Alternative phrasings accepted for the commands above.
COMMAND_ALIASES: dict[str, str] = {
    "open sealed letter": "open letter",
    "read letter": "open letter",
    "inspect tavern ledger": "inspect ledger",
    "read ledger": "inspect ledger",
    "open hidden chest": "open chest",
    "take ancient key": "take key",
    "take royal amulet": "take amulet",
    "open gate": "unlock gate",
    "unlock castle gate": "unlock gate",
    "take ashen crown": "take crown",
}

TRAVEL_PREFIXES = ("travel to ", "go to ")


def create_prototype_state() -> GameState:
    """Return the initial Ashen Crown GameState."""

    return GameState(
        setting={
            "name": "Kingdom of Eldoria",
            "story": PROTOTYPE_TITLE,
            "premise": (
                "Kael, a wandering adventurer, arrives in Ravenmoor. "
                "The Ashen Crown of the old royal family vanished years ago "
                "and Blackstone Castle was abandoned after the final heir "
                "disappeared. Strange activity has recently been reported "
                "around the castle."
            ),
        },
        location=LOCATIONS[STARTING_LOCATION],
        characters=[
            Character(
                PLAYER_CHARACTER,
                "A wandering adventurer. Curious, cautious, and determined.",
            ),
            Character(
                "Mira",
                "The tavern keeper. Practical and observant; knows many "
                "local rumors but is cautious about dangerous subjects.",
            ),
            Character(
                "Elara",
                "A mysterious traveler. Calm, secretive, and knowledgeable; "
                "knows more about Blackstone Castle than she reveals.",
            ),
            Character(
                "Captain Rowan",
                "The village guard captain. Serious and protective; believes "
                "Blackstone Castle should remain sealed.",
                present=False,
            ),
            Character(
                "The Warden",
                "Guardian of Blackstone Castle. Ancient, formal, and cryptic; "
                "protects the secrets of the old royal family.",
                present=False,
            ),
        ],
        objects=[
            GameObject(
                "Rusty Sword",
                "A worn but usable sword.",
                state={"condition": "worn"},
            ),
            GameObject(
                "Sealed Letter",
                "An old letter bearing the royal crest.",
                state={"opened": False},
                location="Old Tavern",
            ),
            GameObject(
                "Ancient Key",
                "An old key bearing the royal crest.",
                state={"discovered": False},
                location="Old Tavern",
            ),
            GameObject(
                "Royal Amulet",
                "A silver amulet bearing the royal crown insignia.",
                state={"discovered": False},
                location="Whispering Forest",
            ),
            GameObject(
                "Ashen Crown",
                "The legendary artifact of the old royal family.",
                state={"discovered": False},
                location="Blackstone Castle",
            ),
            GameObject(
                "Tavern Ledger",
                "An old ledger containing records and notes from the tavern.",
                state={"examined": False},
                location="Old Tavern",
            ),
            GameObject(
                "Castle Gate",
                "An old iron gate leading into Blackstone Castle.",
                state={"locked": True},
                location="Blackstone Castle",
            ),
            GameObject(
                "Hidden Chest",
                "A wooden chest hidden beneath a loose stone.",
                state={"opened": False},
                location="Old Tavern",
            ),
        ],
        inventory=["Rusty Sword"],
        relationships=[
            {"source": "Kael", "target": "Mira", "status": "neutral"},
            {"source": "Kael", "target": "Elara", "status": "unknown"},
            {"source": "Kael", "target": "Captain Rowan", "status": "neutral"},
            {"source": "Kael", "target": "The Warden", "status": "unknown"},
            {"source": "Mira", "target": "Elara", "status": "suspicious"},
            {"source": "Captain Rowan", "target": "Kael", "status": "cautious"},
        ],
        available_actions=[
            "look",
            "talk",
            "inspect",
            "search",
            "leave",
            "travel",
            "take",
            "open",
        ],
        turn=1,
    )


def create_prototype_manager() -> GameStateManager:
    """Return a GameStateManager holding a fresh Ashen Crown world."""

    return GameStateManager(create_prototype_state())


def create_prototype_retriever(
    knowledge_file: Path | str = KNOWLEDGE_FILE,
) -> KnowledgeRetriever:
    """Load the Ashen Crown knowledge into an index and return a retriever."""

    index = KnowledgeIndex()
    index.add_documents(load_knowledge_entries_from_json(str(knowledge_file)))

    return KnowledgeRetriever(index, default_top_k=KNOWLEDGE_TOP_K)


def create_prototype_pipeline(
    generator: ResponseGenerator,
    retriever: KnowledgeRetriever | None = None,
) -> NarrativePipeline:
    """Wire the prototype retriever into the existing NarrativePipeline."""

    if retriever is None:
        retriever = create_prototype_retriever()

    return NarrativePipeline(
        context_builder=ContextBuilder(retriever),
        generator=generator,
    )


def _normalize_command(command: str) -> str:
    return " ".join(command.lower().split()).rstrip(".!?")


def state_changes_for(
    command: str,
    state: GameState,
) -> tuple[StateChangeAction, ...]:
    """Return the explicit state changes for a prototype command.

    Unknown commands, or commands used in the wrong location, change
    nothing; the interaction is still narrated.
    """

    normalized = _normalize_command(command)
    normalized = COMMAND_ALIASES.get(normalized, normalized)

    entry = PROTOTYPE_ACTIONS.get(normalized)
    if entry is None:
        return ()

    required_location, changes = entry

    if state.location is None or state.location.name != required_location:
        return ()

    return changes


def travel_destination(command: str) -> Location | None:
    """Return the Location named by 'travel to X' / 'go to X', if any."""

    normalized = _normalize_command(command)

    for prefix in TRAVEL_PREFIXES:
        if normalized.startswith(prefix):
            name = normalized[len(prefix):]

            for location_name, location in LOCATIONS.items():
                if location_name.lower() == name:
                    return location

    return None


def move_to(manager: GameStateManager, location: Location) -> None:
    """Move the player and update which characters are present."""

    manager.set_location(location)

    for name, home in CHARACTER_LOCATIONS.items():
        manager.set_character_presence(name, home == location.name)


def run_prototype_turn(
    pipeline: NarrativePipeline,
    manager: GameStateManager,
    input_text: str,
    input_type: str = "action",
    responder_character: str | None = None,
) -> PipelineResult:
    """Run one player turn through the pipeline with prototype game logic.

    Explicit state changes are chosen from the command before generation.
    Travel is applied with set_location only after a successful turn.
    """

    state = manager.get_state()
    state_changes = (
        state_changes_for(input_text, state)
        if input_type == "action"
        else ()
    )

    result = pipeline.run(
        manager,
        input_text,
        input_type,
        PLAYER_CHARACTER,
        responder_character,
        state_changes=state_changes,
    )

    destination = (
        travel_destination(input_text) if input_type == "action" else None
    )

    if result.success and destination is not None:
        move_to(manager, destination)

    return result
