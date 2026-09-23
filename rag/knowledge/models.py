"""
Data models for the RAG / Knowledge Retrieval module.
Defines KnowledgeEntry, RAGQuery, KnowledgeResult, and RetrievedKnowledge.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional


@dataclass
class KnowledgeEntry:
    """
    Represents a single piece of stable/semi-stable world knowledge document.

    Attributes:
        document_id: Unique identifier for the document.
        source_type: Category of knowledge (e.g., world_lore, character, location, item, history, world_rule).
        entity_id: Specific entity identifier associated with this entry.
        title: Short title or heading for the document.
        content: Detailed text content of the knowledge entry.
        metadata: Additional key-value metadata preserved alongside the entry.
    """
    document_id: str
    source_type: str
    entity_id: str
    title: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert entry to dictionary representation."""
        return {
            "document_id": self.document_id,
            "source_type": self.source_type,
            "entity_id": self.entity_id,
            "title": self.title,
            "content": self.content,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "KnowledgeEntry":
        """Construct KnowledgeEntry from dictionary representation."""
        return cls(
            document_id=str(d.get("document_id", "")),
            source_type=str(d.get("source_type", "")),
            entity_id=str(d.get("entity_id", "")),
            title=str(d.get("title", "")),
            content=str(d.get("content", "")),
            metadata=dict(d.get("metadata", {})),
        )


@dataclass
class RAGQuery:
    """
    Query interface for knowledge retrieval.

    Attributes:
        query: Main query text or player input snippet.
        input_type: Type of player input ("speech" or "action").
        location: Current location/setting name.
        input_character: Name of the active player/character initiating input.
        responder_character: Name of the responding character/NPC.
        relevant_entities: List of entity names/IDs relevant to the query context.
    """
    query: str
    input_type: str = "speech"
    location: str = ""
    input_character: str = ""
    responder_character: str = ""
    relevant_entities: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert RAGQuery to dictionary representation matching frozen schema."""
        return {
            "query": self.query,
            "input_type": self.input_type,
            "location": self.location,
            "input_character": self.input_character,
            "responder_character": self.responder_character,
            "relevant_entities": list(self.relevant_entities),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RAGQuery":
        """Construct RAGQuery from dictionary representation."""
        return cls(
            query=str(d.get("query", "")),
            input_type=str(d.get("input_type", "speech")),
            location=str(d.get("location", "")),
            input_character=str(d.get("input_character", "")),
            responder_character=str(d.get("responder_character", "")),
            relevant_entities=list(d.get("relevant_entities", [])),
        )


@dataclass
class KnowledgeResult:
    """
    Individual retrieved document result with relevance score and metadata.

    Attributes:
        document_id: Unique identifier of the document.
        content: Text content of the document.
        source_type: Category of knowledge.
        entity_id: Entity identifier.
        title: Document title.
        metadata: Metadata dictionary.
        score: Computed relevance score.
    """
    document_id: str
    content: str
    source_type: str = ""
    entity_id: str = ""
    title: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert KnowledgeResult to dictionary representation."""
        return {
            "document_id": self.document_id,
            "content": self.content,
            "source_type": self.source_type,
            "entity_id": self.entity_id,
            "title": self.title,
            "metadata": dict(self.metadata),
            "score": round(float(self.score), 4),
        }


@dataclass
class RetrievedKnowledge:
    """
    Container for retrieved knowledge results matching frozen schema.

    Attributes:
        results: List of retrieved KnowledgeResult items.
    """
    results: List[KnowledgeResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary matching frozen conceptual interface."""
        return {
            "results": [r.to_dict() for r in self.results]
        }
