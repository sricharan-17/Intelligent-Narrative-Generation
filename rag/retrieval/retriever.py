"""
Retriever component for RAG module.
Accepts RAGQuery, searches KnowledgeIndex, applies relevance threshold filtering,
and returns RetrievedKnowledge matching the frozen conceptual interface.
"""

from typing import Optional, List, Dict, Any
from rag.knowledge.models import (
    RAGQuery,
    RetrievedKnowledge,
    KnowledgeResult,
    KnowledgeEntry,
)
from rag.indexing.index import KnowledgeIndex


class KnowledgeRetriever:
    """
    Independent knowledge retriever component.

    Operates strictly on stable world knowledge indexed in KnowledgeIndex.
    Applies relevance filtering to exclude non-matching documents.
    Does NOT modify or depend on dynamic Game State or LLM generation.
    """

    def __init__(
        self,
        index: KnowledgeIndex,
        min_score: float = 0.5,
        default_top_k: int = 3,
    ):
        """
        Initialize the retriever.

        Args:
            index: Built KnowledgeIndex instance.
            min_score: Minimum relevance score threshold. Results below this threshold are omitted.
            default_top_k: Default maximum number of knowledge documents to return.
        """
        self.index = index
        self.min_score = float(min_score)
        self.default_top_k = int(default_top_k)

    def retrieve(
        self,
        query: RAGQuery,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> RetrievedKnowledge:
        """
        Retrieve relevant knowledge documents for a given RAGQuery.

        Args:
            query: The RAGQuery object containing query text, location, characters, and entities.
            top_k: Optional override for max results count.
            min_score: Optional override for minimum relevance score threshold.

        Returns:
            RetrievedKnowledge containing a list of KnowledgeResult items.
            If no documents meet the minimum relevance threshold, returns RetrievedKnowledge(results=[]).
        """
        k = top_k if top_k is not None else self.default_top_k
        threshold = min_score if min_score is not None else self.min_score

        # Query the index
        scored_docs = self.index.search_scored(
            query_text=query.query,
            relevant_entities=query.relevant_entities,
            location=query.location,
            input_character=query.input_character,
            responder_character=query.responder_character,
        )

        filtered_results: List[KnowledgeResult] = []

        for doc, score in scored_docs:
            if score >= threshold:
                result_item = KnowledgeResult(
                    document_id=doc.document_id,
                    content=doc.content,
                    source_type=doc.source_type,
                    entity_id=doc.entity_id,
                    title=doc.title,
                    metadata=dict(doc.metadata),
                    score=score,
                )
                filtered_results.append(result_item)

            if len(filtered_results) >= k:
                break

        return RetrievedKnowledge(results=filtered_results)
