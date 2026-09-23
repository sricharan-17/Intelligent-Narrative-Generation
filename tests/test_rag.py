"""
Comprehensive standalone unit tests for the RAG / Knowledge Retrieval Module.
Validates loading, indexing, retrieval, relevance threshold filtering, metadata preservation,
and independence from Game State / LLM modules.
"""

import os
import json
import tempfile
import unittest

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
from rag.indexing.index import KnowledgeIndex, tokenize
from rag.retrieval.retriever import KnowledgeRetriever


class TestRAGModule(unittest.TestCase):
    """Test suite for the RAG knowledge retrieval module."""

    def setUp(self):
        """Set up fixture data for testing."""
        self.doc_castle = KnowledgeEntry(
            document_id="loc_royal_castle",
            source_type="location",
            entity_id="Royal Castle",
            title="Location: Royal Castle",
            content="The Royal Castle sits high atop the dragon ridge. It was constructed by ancient stone masons in 1200 AD.",
            metadata={"built_year": 1200, "region": "Highland"},
        )
        self.doc_wizard = KnowledgeEntry(
            document_id="char_court_wizard",
            source_type="character",
            entity_id="court wizard",
            title="Character Persona: Court Wizard",
            content="I am the master advisor of arcane arts. I sell spells and advise the king on ancient prophecies.",
            metadata={"role": "advisor", "element": "arcane"},
        )
        self.doc_sword = KnowledgeEntry(
            document_id="item_silver_blade",
            source_type="item",
            entity_id="silver blade",
            title="Item Description: Silver Blade",
            content="The Silver Blade was forged in royal dwarven furnaces. It shines with moonlight luminosity.",
            metadata={"material": "silver", "origin": "dwarven"},
        )
        self.doc_dragon = KnowledgeEntry(
            document_id="lore_dragon_war",
            source_type="world_lore",
            entity_id="Dragon War",
            title="World Lore: The Dragon War",
            content="Centuries ago, five dragons attacked the kingdom during the Great Dragon War until sealed by the wizard council.",
            metadata={"era": "ancient_history"},
        )

        self.sample_docs = [self.doc_castle, self.doc_wizard, self.doc_sword, self.doc_dragon]

    # -------------------------------------------------------------
    # 1. Knowledge Loading Tests
    # -------------------------------------------------------------

    def test_knowledge_entry_and_query_dict_conversion(self):
        """Test KnowledgeEntry and RAGQuery serialization to/from dictionary."""
        d = self.doc_castle.to_dict()
        self.assertEqual(d["document_id"], "loc_royal_castle")
        self.assertEqual(d["source_type"], "location")
        self.assertEqual(d["metadata"]["built_year"], 1200)

        reconstructed = KnowledgeEntry.from_dict(d)
        self.assertEqual(reconstructed.document_id, self.doc_castle.document_id)
        self.assertEqual(reconstructed.content, self.doc_castle.content)

        query = RAGQuery(
            query="tell me about dragons",
            input_type="speech",
            location="Royal Castle",
            input_character="court wizard",
            responder_character="soldier",
            relevant_entities=["Dragon War"],
        )
        q_dict = query.to_dict()
        self.assertEqual(q_dict["query"], "tell me about dragons")
        self.assertEqual(q_dict["location"], "Royal Castle")
        self.assertEqual(q_dict["relevant_entities"], ["Dragon War"])

    def test_loading_from_dicts_and_json(self):
        """Test loading knowledge entries from list of dicts and JSON files."""
        dicts = [self.doc_castle.to_dict(), self.doc_wizard.to_dict()]
        loaded = load_knowledge_entries_from_dicts(dicts)
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0].document_id, "loc_royal_castle")

        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".json", encoding="utf-8") as f:
            json.dump(dicts, f)
            temp_path = f.name

        try:
            loaded_json = load_knowledge_entries_from_json(temp_path)
            self.assertEqual(len(loaded_json), 2)
            self.assertEqual(loaded_json[1].entity_id, "court wizard")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_extract_knowledge_from_light_record(self):
        """Test extraction of stable knowledge entries from a LIGHT record."""
        record = {
            "setting": {
                "name": "Watchtower",
                "category": "Outside Tower",
                "description": "The tower is the largest section of the castle.",
                "background": "First line of defense for the castle.",
            },
            "input_character": {
                "name": "court wizard",
                "persona": "I am an advisor of magic.",
            },
            "responder_character": {
                "name": "soldier",
                "persona": "I came from the valley.",
            },
            "world_state": {
                "object_descriptions": {
                    "an alarm horn": "A bronze bugle, loud enough to sound the alarm."
                }
            },
        }

        extracted = extract_knowledge_from_light_record(record)
        self.assertEqual(len(extracted), 4)

        doc_ids = {e.document_id for e in extracted}
        self.assertIn("loc_watchtower", doc_ids)
        self.assertIn("char_court_wizard", doc_ids)
        self.assertIn("char_soldier", doc_ids)
        self.assertIn("item_an_alarm_horn", doc_ids)

    # -------------------------------------------------------------
    # 2. Indexing Tests
    # -------------------------------------------------------------

    def test_index_creation_and_management(self):
        """Test index creation, document indexing, size, and document lookup."""
        index = KnowledgeIndex()
        self.assertEqual(index.size(), 0)

        index.add_documents(self.sample_docs)
        self.assertEqual(index.size(), 4)

        retrieved_doc = index.get_document("loc_royal_castle")
        self.assertIsNotNone(retrieved_doc)
        self.assertEqual(retrieved_doc.title, "Location: Royal Castle")

        index.remove_document("loc_royal_castle")
        self.assertEqual(index.size(), 3)
        self.assertIsNone(index.get_document("loc_royal_castle"))

        index.clear()
        self.assertEqual(index.size(), 0)

    # -------------------------------------------------------------
    # 3. Retrieval & Relevant Query Tests
    # -------------------------------------------------------------

    def test_basic_retrieval(self):
        """Test basic retrieval using KnowledgeRetriever."""
        index = KnowledgeIndex()
        index.add_documents(self.sample_docs)
        retriever = KnowledgeRetriever(index, min_score=0.1, default_top_k=2)

        query = RAGQuery(query="Where is the dragon ridge castle?")
        res = retriever.retrieve(query)

        self.assertIsInstance(res, RetrievedKnowledge)
        self.assertGreater(len(res.results), 0)
        self.assertEqual(res.results[0].document_id, "loc_royal_castle")

    def test_relevant_query_retrieval_with_context(self):
        """Test query matching with location, character, and relevant entities."""
        index = KnowledgeIndex()
        index.add_documents(self.sample_docs)
        retriever = KnowledgeRetriever(index, min_score=0.1, default_top_k=3)

        query = RAGQuery(
            query="Who sells spells here?",
            location="Royal Castle",
            input_character="court wizard",
            relevant_entities=["silver blade"],
        )

        res = retriever.retrieve(query)
        retrieved_ids = [r.document_id for r in res.results]

        self.assertIn("char_court_wizard", retrieved_ids)
        self.assertIn("item_silver_blade", retrieved_ids)

    # -------------------------------------------------------------
    # 4. Irrelevant Query Filtering & Empty Results
    # -------------------------------------------------------------

    def test_irrelevant_query_filtering(self):
        """Test that irrelevant queries scoring below min_score return empty results."""
        index = KnowledgeIndex()
        index.add_documents(self.sample_docs)

        # Set high relevance threshold
        retriever = KnowledgeRetriever(index, min_score=10.0)

        query = RAGQuery(query="quantum physics space station satellite")
        res = retriever.retrieve(query)

        self.assertEqual(len(res.results), 0)
        self.assertEqual(res.to_dict(), {"results": []})

    def test_empty_index_handling(self):
        """Test retriever behavior with an empty index."""
        index = KnowledgeIndex()
        retriever = KnowledgeRetriever(index, min_score=0.1)

        query = RAGQuery(query="tell me about the castle")
        res = retriever.retrieve(query)

        self.assertEqual(len(res.results), 0)
        self.assertEqual(res.to_dict(), {"results": []})

    # -------------------------------------------------------------
    # 5. Metadata Preservation Tests
    # -------------------------------------------------------------

    def test_metadata_preservation(self):
        """Test that document_id, source_type, entity_id, title, and metadata are strictly preserved."""
        index = KnowledgeIndex()
        index.add_document(self.doc_castle)
        retriever = KnowledgeRetriever(index, min_score=0.1)

        query = RAGQuery(query="royal castle built year")
        res = retriever.retrieve(query)

        self.assertEqual(len(res.results), 1)
        item = res.results[0]

        self.assertEqual(item.document_id, "loc_royal_castle")
        self.assertEqual(item.source_type, "location")
        self.assertEqual(item.entity_id, "Royal Castle")
        self.assertEqual(item.title, "Location: Royal Castle")
        self.assertEqual(item.metadata.get("built_year"), 1200)
        self.assertEqual(item.metadata.get("region"), "Highland")

    # -------------------------------------------------------------
    # 6. Multiple Retrieved Documents
    # -------------------------------------------------------------

    def test_multiple_retrieved_documents_and_top_k(self):
        """Test retrieving multiple top-K documents ordered by relevance score."""
        index = KnowledgeIndex()
        index.add_documents(self.sample_docs)

        retriever = KnowledgeRetriever(index, min_score=0.01)

        query = RAGQuery(query="dragon wizard spell castle blade")
        res = retriever.retrieve(query, top_k=3)

        self.assertLessEqual(len(res.results), 3)
        self.assertGreater(len(res.results), 1)

        # Check score descending order
        scores = [r.score for r in res.results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    # -------------------------------------------------------------
    # 7. LLM & Game State Independence
    # -------------------------------------------------------------

    def test_llm_and_gamestate_independence(self):
        """Verify RAG operates deterministically without model or game state dependencies."""
        index = KnowledgeIndex()
        index.add_documents(self.sample_docs)
        retriever = KnowledgeRetriever(index, min_score=0.1)

        query = RAGQuery(query="Who is the wizard?")

        # 1. Determinism: identical input yields identical output
        res1 = retriever.retrieve(query)
        res2 = retriever.retrieve(query)
        self.assertEqual(res1.to_dict(), res2.to_dict())

        # 2. No Game State modification: RAG results do not mutate input query or index state
        self.assertEqual(index.size(), 4)
        self.assertEqual(query.query, "Who is the wizard?")

        # 3. Output follows strict schema: {"results": [...]}
        d = res1.to_dict()
        self.assertIn("results", d)
        for doc in d["results"]:
            self.assertIn("document_id", doc)
            self.assertIn("content", doc)

    def test_real_dataset_sample_extraction(self):
        """Test extraction and retrieval over sample records from project's processed dataset split."""
        dataset_file = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data",
            "splits",
            "test.jsonl"
        )
        if not os.path.exists(dataset_file):
            self.skipTest(f"Dataset file not found at {dataset_file}")

        entries = load_knowledge_from_light_dataset(dataset_file, max_records=50)
        self.assertGreater(len(entries), 0)

        index = KnowledgeIndex()
        index.add_documents(entries)
        self.assertEqual(index.size(), len(entries))

        first_entry = entries[0]
        retriever = KnowledgeRetriever(index, min_score=0.1)

        query = RAGQuery(
            query=first_entry.title,
            relevant_entities=[first_entry.entity_id]
        )
        res = retriever.retrieve(query)

        self.assertGreater(len(res.results), 0)
        self.assertEqual(res.results[0].document_id, first_entry.document_id)


if __name__ == "__main__":
    unittest.main()
