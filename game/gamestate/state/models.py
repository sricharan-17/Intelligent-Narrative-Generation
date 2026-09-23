from dataclasses import dataclass, field
from typing import Any


@dataclass
class Location:
    name: str
    description: str = ""


@dataclass
class Character:
    name: str
    description: str = ""
    present: bool = True


@dataclass
class GameObject:
    name: str
    description: str = ""
    state: dict[str, Any] = field(default_factory=dict)
    location: str | None = None


@dataclass
class RecentEvent:
    turn: int
    event_type: str
    description: str


@dataclass
class HistoryEntry:
    turn: int
    speaker: str
    text: str
    entry_type: str


@dataclass
class GameState:
    setting: dict[str, Any] = field(default_factory=dict)
    location: Location | None = None
    characters: list[Character] = field(default_factory=list)
    objects: list[GameObject] = field(default_factory=list)
    inventory: list[str] = field(default_factory=list)
    relationships: list[dict[str, Any]] = field(default_factory=list)
    available_actions: list[str] = field(default_factory=list)
    recent_events: list[RecentEvent] = field(default_factory=list)
    history: list[HistoryEntry] = field(default_factory=list)
    turn: int = 0