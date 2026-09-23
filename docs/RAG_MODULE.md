# RAG / Knowledge Retrieval Module Documentation

**Project:** Intelligent Narrative Generation for Interactive Gaming Using Fine-Tuned Large Language Models  
**Phase:** Phase I — Text-Based Interactive Narrative System  
**Module:** `rag`

---

## 1. Module Overview & Responsibilities

The `rag` module provides a standalone, lightweight knowledge retrieval system that supplies stable and semi-stable world lore to downstream narrative components.

### What RAG Stores:
- **World Lore & History:** Regional backgrounds, historical events, dragon wars, ancient kingdoms.
- **Character Personas & Backgrounds:** Character roles, personality traits, personal backstories.
- **Location Descriptions:** Setting names, environment categories, location descriptions, lore history.
- **Item & Object Descriptions:** Craftsmanship, lore origin, physical descriptions of legendary artifacts.
- **World Rules:** General rules of magic, factions, world laws.

### What RAG DOES NOT Store (Game State Scope):
- **Current Player Location** (autoritative in Game State)
- **Current Room Inventory & Object States** (e.g. open/closed chest, broken door)
- **Character HP / Status / Active Effects**
- **Available Actions in the turn**
- **Recent Interaction History**

> **Fundamental Principle:** RAG provides contextual world lore. RAG NEVER overrides dynamic Game State.

---

## 2. Architecture & Data Structures

```
Player Input ──> Game State ──> RAGQuery ──> KnowledgeRetriever ──> RetrievedKnowledge ──> Context Builder ──> LLM
                                                   │
                                                   ▼
                                            KnowledgeIndex
                                                   │
                                            KnowledgeEntry List
```

### Data Models (`rag/knowledge/models.py`)

#### `KnowledgeEntry`
```python
@dataclass
class KnowledgeEntry:
    document_id: str
    source_type: str        # "location", "character", "item", "world_lore", "world_rule", "history"
    entity_id: str          # Entity name (e.g., "Royal Castle", "court wizard")
    title: str              # Descriptive heading
    content: str            # Full text body
    metadata: Dict[str, Any]# Arbitrary key-value metadata dictionary
```

#### `RAGQuery` (Frozen Interface)
```python
@dataclass
class RAGQuery:
    query: str                       # Query text / player input string
    input_type: str = "speech"       # "speech" or "action"
    location: str = ""               # Current setting name
    input_character: str = ""        # Character taking action
    responder_character: str = ""    # Character responding
    relevant_entities: List[str] = field(default_factory=list)
```

#### `RetrievedKnowledge` & `KnowledgeResult` (Frozen Interface)
```python
@dataclass
class KnowledgeResult:
    document_id: str
    content: str
    source_type: str = ""
    entity_id: str = ""
    title: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0

@dataclass
class RetrievedKnowledge:
    results: List[KnowledgeResult]
```

Default JSON serialization format (`RetrievedKnowledge.to_dict()`):
```json
{
    "results": [
        {
            "document_id": "loc_watchtower",
            "content": "Description of Watchtower...",
            "source_type": "location",
            "entity_id": "Watchtower",
            "title": "Location: Watchtower",
            "metadata": {"category": "Outside Tower"},
            "score": 4.521
        }
    ]
}
```

---

## 3. Dataset Loading & Extraction (`rag/knowledge/loader.py`)

- **Dataset Safety:** Extractor utilities scan dataset records to harvest unique settings, character personas, and object descriptions. **Source dataset files (`data/processed/` and `data/splits/`) are NEVER modified.**
- **Functions:**
  - `load_knowledge_entries_from_dicts(data: list[dict]) -> list[KnowledgeEntry]`
  - `load_knowledge_entries_from_json(file_path: str) -> list[KnowledgeEntry]`
  - `extract_knowledge_from_light_record(record: dict) -> list[KnowledgeEntry]`
  - `load_knowledge_from_light_dataset(dataset_path: str, max_records=None) -> list[KnowledgeEntry]`

---

## 4. Indexing & Retrieval (`rag/indexing` & `rag/retrieval`)

### Inverted Index (`KnowledgeIndex`)
- In-memory multi-field inverted index.
- Field weights: `title` (3.0), `entity_id` (4.0), `content` (1.0), `metadata` (1.5).
- Length normalization and IDF scaling for fair relevance scoring.
- Exact entity, location, and character match boosts.

### Retriever (`KnowledgeRetriever`)
- Performs relevance-scored retrieval given a `RAGQuery`.
- **Relevance Filtering:** Filters out candidate documents whose relevance score falls below `min_score` (default `0.5`).
- **Empty Result Handling:** Returns `RetrievedKnowledge(results=[])` when no document passes `min_score`.
- Deterministic, offline execution with zero external dependencies.

---

## 5. How to Run Unit Tests

Execute the standalone unit test suite from project root:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

The test suite covers:
1. Knowledge loading & dictionary conversion
2. Index creation & document lookup
3. Basic retrieval with RAGQuery
4. Contextual retrieval (location, characters, entities)
5. Irrelevant query threshold filtering (`results: []`)
6. Metadata preservation
7. Empty-result handling
8. Multiple document retrieval and top-k ordering
9. Real dataset sample extraction & retrieval
10. Deterministic independence from LLM and network.

---

## 6. Integration Guide for Other Modules

### 1. Game State Module Integration
The Game State module should build a `RAGQuery` combining the player's latest input text with current context attributes:

```python
from rag import RAGQuery

# Game State constructs query without letting RAG dictate dynamic facts:
query = RAGQuery(
    query=player_action_text,
    input_type="action",
    location=current_game_state.location_name,
    input_character=current_game_state.player_name,
    responder_character=current_game_state.npc_name,
    relevant_entities=current_game_state.get_nearby_entity_ids(),
)
```

### 2. Context Builder / LLM Integration
The Context Builder receives `RetrievedKnowledge` and formats lore snippets into the LLM prompt:

```python
from rag import KnowledgeRetriever, KnowledgeIndex, load_knowledge_from_light_dataset

# Initialize once during game start:
knowledge_entries = load_knowledge_from_light_dataset("data/splits/train.jsonl")
index = KnowledgeIndex()
index.add_documents(knowledge_entries)
retriever = KnowledgeRetriever(index, min_score=0.5, default_top_k=3)

# Turn loop:
retrieved = retriever.retrieve(query)

# Format lore for LLM context window:
lore_context_str = "\n".join([
    f"[{r.title}]: {r.content}" for r in retrieved.results
])
```
