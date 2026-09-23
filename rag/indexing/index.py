"""
Inverted index and term statistics management for KnowledgeEntry documents.
Provides deterministic, multi-field weighted indexing over title, entity_id, content, and metadata.
"""

import re
import math
from collections import defaultdict, Counter
from typing import List, Dict, Any, Optional, Set, Tuple, Iterable
from rag.knowledge.models import KnowledgeEntry

# Common English stop words to filter out uninformative terms
STOP_WORDS: Set[str] = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
    "to", "was", "were", "will", "with", "this", "there", "they", "or",
    "you", "your", "my", "i", "me", "we", "our", "us", "his", "her",
    "she", "him", "them", "their", "what", "which", "who", "whom"
}


def tokenize(text: str) -> List[str]:
    """
    Tokenize text into lowercase alphanumeric words.
    Filters out single-char tokens and common stop words.
    """
    if not text:
        return []
    words = re.findall(r"\b\w+\b", text.lower())
    return [w for w in words if len(w) > 1 and w not in STOP_WORDS]


class KnowledgeIndex:
    """
    In-memory inverted index for fast multi-field lexical matching over KnowledgeEntry objects.

    Uses term frequency weighting across title, entity_id, content, and metadata.
    """

    def __init__(self):
        self._documents: Dict[str, KnowledgeEntry] = {}
        # term -> dict of doc_id -> score weight sum
        self._inverted_index: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        # doc_id -> word length (for length normalization)
        self._doc_lengths: Dict[str, int] = {}
        # entity_id (lowercase) -> list of doc_ids
        self._entity_map: Dict[str, List[str]] = defaultdict(list)
        # source_type -> list of doc_ids
        self._type_map: Dict[str, List[str]] = defaultdict(list)

    def size(self) -> int:
        """Return total number of indexed documents."""
        return len(self._documents)

    def clear(self) -> None:
        """Clear all indexed documents and structures."""
        self._documents.clear()
        self._inverted_index.clear()
        self._doc_lengths.clear()
        self._entity_map.clear()
        self._type_map.clear()

    def get_document(self, doc_id: str) -> Optional[KnowledgeEntry]:
        """Retrieve indexed KnowledgeEntry by document_id."""
        return self._documents.get(doc_id)

    def add_document(self, doc: KnowledgeEntry) -> None:
        """
        Add or update a KnowledgeEntry in the index.

        Indexes text from title (weight 3.0), entity_id (weight 4.0),
        content (weight 1.0), and metadata (weight 1.5).
        """
        doc_id = doc.document_id
        if doc_id in self._documents:
            # Re-indexing: remove existing entries first
            self.remove_document(doc_id)

        self._documents[doc_id] = doc

        # Record entity mapping
        if doc.entity_id:
            self._entity_map[doc.entity_id.strip().lower()].append(doc_id)

        # Record source_type mapping
        if doc.source_type:
            self._type_map[doc.source_type.strip().lower()].append(doc_id)

        # Extract tokens with field weights
        title_tokens = tokenize(doc.title)
        entity_tokens = tokenize(doc.entity_id)
        content_tokens = tokenize(doc.content)

        metadata_text = " ".join(
            str(v) for v in doc.metadata.values() if isinstance(v, (str, int, float))
        )
        metadata_tokens = tokenize(metadata_text)

        total_word_count = len(title_tokens) + len(entity_tokens) + len(content_tokens) + len(metadata_tokens)
        self._doc_lengths[doc_id] = max(total_word_count, 1)

        # Accumulate weighted frequencies for inverted index
        for t in title_tokens:
            self._inverted_index[t][doc_id] += 3.0

        for t in entity_tokens:
            self._inverted_index[t][doc_id] += 4.0

        for t in content_tokens:
            self._inverted_index[t][doc_id] += 1.0

        for t in metadata_tokens:
            self._inverted_index[t][doc_id] += 1.5

    def add_documents(self, docs: Iterable[KnowledgeEntry]) -> None:
        """Add multiple KnowledgeEntry documents to the index."""
        for d in docs:
            self.add_document(d)

    def remove_document(self, doc_id: str) -> None:
        """Remove a document from the index if present."""
        if doc_id not in self._documents:
            return

        doc = self._documents.pop(doc_id)
        self._doc_lengths.pop(doc_id, None)

        if doc.entity_id:
            ent_key = doc.entity_id.strip().lower()
            if doc_id in self._entity_map[ent_key]:
                self._entity_map[ent_key].remove(doc_id)

        if doc.source_type:
            st_key = doc.source_type.strip().lower()
            if doc_id in self._type_map[st_key]:
                self._type_map[st_key].remove(doc_id)

        # Remove from inverted index
        for term, posting in list(self._inverted_index.items()):
            if doc_id in posting:
                del posting[doc_id]
                if not posting:
                    del self._inverted_index[term]

    def search_scored(
        self,
        query_text: str,
        relevant_entities: Optional[List[str]] = None,
        location: str = "",
        input_character: str = "",
        responder_character: str = "",
    ) -> List[Tuple[KnowledgeEntry, float]]:
        """
        Search index and return scored (KnowledgeEntry, score) tuples ordered by relevance.

        Scoring combines term-based matching (TF-IDF inspired) and exact entity/location match boosts.
        """
        if not self._documents:
            return []

        doc_scores: Dict[str, float] = defaultdict(float)
        query_tokens = tokenize(query_text)

        # 1. Term matching score
        num_docs = len(self._documents)
        for token in set(query_tokens):
            if token in self._inverted_index:
                postings = self._inverted_index[token]
                df = len(postings)
                idf = math.log((num_docs + 1.0) / (df + 0.5)) + 1.0
                for doc_id, tf in postings.items():
                    norm_len = math.sqrt(self._doc_lengths.get(doc_id, 10))
                    doc_scores[doc_id] += (tf * idf) / max(norm_len, 1.0)

        # 2. Entity match boost
        entities_to_check = []
        if relevant_entities:
            entities_to_check.extend(relevant_entities)

        for ent in entities_to_check:
            ent_clean = str(ent).strip().lower()
            if not ent_clean:
                continue

            # Boost documents whose entity_id matches exactly or as substring
            for doc_id, doc in self._documents.items():
                doc_ent = doc.entity_id.lower()
                doc_title = doc.title.lower()
                if ent_clean == doc_ent or ent_clean in doc_ent or ent_clean in doc_title:
                    doc_scores[doc_id] += 5.0

            # Also tokenize entity for term lookup boost
            for ent_token in tokenize(ent_clean):
                if ent_token in self._inverted_index:
                    for doc_id, tf in self._inverted_index[ent_token].items():
                        doc_scores[doc_id] += 2.0

        # 3. Location match boost
        if location:
            loc_clean = location.strip().lower()
            for doc_id, doc in self._documents.items():
                if doc.source_type == "location":
                    if loc_clean in doc.entity_id.lower() or loc_clean in doc.title.lower():
                        doc_scores[doc_id] += 4.0

        # 4. Character match boost
        characters = [c for c in [input_character, responder_character] if c and c.strip()]
        for char in characters:
            char_clean = char.strip().lower()
            for doc_id, doc in self._documents.items():
                if doc.source_type == "character":
                    if char_clean in doc.entity_id.lower() or char_clean in doc.title.lower():
                        doc_scores[doc_id] += 4.0

        # Build sorted list of (KnowledgeEntry, score)
        results = []
        for doc_id, score in doc_scores.items():
            if doc_id in self._documents:
                results.append((self._documents[doc_id], score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results
