from dataclasses import dataclass


@dataclass(frozen=True)
class Interaction:
    input: str
    input_type: str
    input_character: str
    responder_character: str