from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class StateChangeAction:
    """A validated description of a requested GameState change."""

    action_type: str
    target: str | None = None
    changes: dict[str, Any] = field(default_factory=dict)
    inventory_add: tuple[str, ...] = ()
    inventory_remove: tuple[str, ...] = ()