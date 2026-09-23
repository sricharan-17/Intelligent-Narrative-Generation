from dataclasses import dataclass


@dataclass(frozen=True)
class Interaction:
    input: str
    input_type: str
    input_character: str
    # The game-world character whose response is generated, or None for
    # world/object interactions where no character responds.
    responder_character: str | None