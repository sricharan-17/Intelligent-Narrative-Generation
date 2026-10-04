from game.context.builder import ContextBuilder
from game.gamestate.interaction.models import Interaction
from game.gamestate.state.manager import GameStateManager
from game.gamestate.state.models import (
    Character,
    GameObject,
    GameState,
    Location,
)
from rag.indexing.index import KnowledgeIndex
from rag.retrieval.retriever import KnowledgeRetriever
from llm.prompt_builder import NarrativePromptBuilder


state = GameState(
    setting={
        "category": "Inside Tower",
        "background": (
            "Slaves are forced to reside here while they work off their debt. "
            "It is an extremely depressing place that few ever seem to leave."
        ),
    },

    location=Location(
        "Slave quarters",
        description=(
            "The slave quarters are a dark and foul smelling stone room "
            "where men are chained shoulder to shoulder against one another "
            "until work needs to be done. There's a large bucket of slop "
            "that comes through and the slaves are allowed to feed themselves briefly."
        ),
    ),

    characters=[
        Character(
            "police",
            description=(
                "I am a policeman in charge of maintaining order in a bad neighborhood. "
                "I arrest others who challenge my authority. "
                "I make sure to demand a high salary for the dangerous work I do."
            ),
        ),
        Character(
            "a tribesman",
            description=(
                "My tribe is my life. We gather, hunt and sing. "
                "I live a free man, not hiding behind a wall."
            ),
        ),
    ],

    objects=[
        GameObject(
            "a A basket of grain",
            description=(
                "The basket has a leather-hinged lid to keep its contents secure. "
                "The inside is large and would hold a great deal."
            ),
        ),
        GameObject(
            "a Arrow",
            description=(
                "The arrow is sharpebed in the way that could pierce the any thing."
            ),
        ),
    ],

    inventory=[
        "a ceremonial hat",
        "a gun",
        "a hats",
    ],

    available_actions=[
        "steal A basket of grain from a tribesman",
        "hit a tribesman",
        "remove gun",
        "hug a tribesman",
        "remove hats",
        "remove ceremonial hat",
    ],
)


manager = GameStateManager(state)


interaction = Interaction(
    input_character="police",
    responder_character="a tribesman",
    input="I was a policeman",
    input_type="speech",
)


context = ContextBuilder(
    KnowledgeRetriever(KnowledgeIndex())
).build(
    interaction,
    manager,
)


messages = NarrativePromptBuilder().build_messages(context)


print("===== SYSTEM =====")
print(messages[0]["content"])

print("\n===== USER PROMPT =====")
print(messages[1]["content"])