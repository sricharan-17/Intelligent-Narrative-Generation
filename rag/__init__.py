"""
RAG / Knowledge Retrieval Module for Interactive Narrative System.
Provides stable/semi-stable world knowledge retrieval independent of Game State and LLM.
"""

from rag.knowledge.models import (
    KnowledgeEntry,
    RAGQuery,
    KnowledgeResult,
    RetrievedKnowledge,
)
from rag.knowledge.loader import (
    load_knowledge_entries_from_dicts,
    load_knowledge_entries_from_json,
    extract_knowledge_from_light_record,
    load_knowledge_from_light_dataset,
)
from rag.indexing.index import KnowledgeIndex
from rag.retrieval.retriever import KnowledgeRetriever

__all__ = [
    "KnowledgeEntry",
    "RAGQuery",
    "KnowledgeResult",
    "RetrievedKnowledge",
    "load_knowledge_entries_from_dicts",
    "load_knowledge_entries_from_json",
    "extract_knowledge_from_light_record",
    "load_knowledge_from_light_dataset",
    "KnowledgeIndex",
    "KnowledgeRetriever",
]
