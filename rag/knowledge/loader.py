"""
Knowledge loading and dataset extraction utilities for RAG.
Extracts stable world lore, character backgrounds, location descriptions, and item details
from structured JSON/JSONL resources or LIGHT dataset records without modifying source datasets.
"""

import os
import json
from typing import List, Dict, Any, Optional, Set
from rag.knowledge.models import KnowledgeEntry


def load_knowledge_entries_from_dicts(data: List[Dict[str, Any]]) -> List[KnowledgeEntry]:
    """Load KnowledgeEntry instances from a list of dictionaries."""
    entries = []
    for d in data:
        if isinstance(d, dict):
            entries.append(KnowledgeEntry.from_dict(d))
    return entries


def load_knowledge_entries_from_json(file_path: str) -> List[KnowledgeEntry]:
    """
    Load KnowledgeEntry instances from a JSON or JSONL file.

    Supports either a JSON array of documents or a JSONL file (one JSON object per line).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Knowledge file not found: {file_path}")

    entries = []
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if not content:
            return []

        if content.startswith("["):
            data = json.loads(content)
            entries = load_knowledge_entries_from_dicts(data)
        else:
            for line in content.splitlines():
                line = line.strip()
                if line:
                    d = json.loads(line)
                    entries.append(KnowledgeEntry.from_dict(d))

    return entries


def extract_knowledge_from_light_record(record: Dict[str, Any]) -> List[KnowledgeEntry]:
    """
    Extract stable narrative knowledge entries (settings, character personas, object descriptions)
    from a single LIGHT dataset record.

    Does NOT extract dynamic runtime game state (available actions, inventory status, dialogue turns).
    """
    entries = []

    # 1. Location / Setting Knowledge
    setting = record.get("setting", {})
    if isinstance(setting, dict) and setting.get("name"):
        name = str(setting["name"]).strip()
        category = str(setting.get("category", "")).strip()
        description = str(setting.get("description", "")).strip()
        background = str(setting.get("background", "")).strip()

        content_parts = []
        if category:
            content_parts.append(f"Category: {category}")
        if description:
            content_parts.append(f"Description: {description}")
        if background:
            content_parts.append(f"Background Lore: {background}")

        if content_parts:
            doc_id = f"loc_{name.lower().replace(' ', '_')}"
            entries.append(
                KnowledgeEntry(
                    document_id=doc_id,
                    source_type="location",
                    entity_id=name,
                    title=f"Location: {name}",
                    content="\n".join(content_parts),
                    metadata={
                        "category": category,
                        "description": description,
                        "background": background,
                    },
                )
            )

    # 2. Character Persona Knowledge
    for char_key in ["input_character", "responder_character"]:
        char_info = record.get(char_key, {})
        if isinstance(char_info, dict) and char_info.get("name"):
            char_name = str(char_info["name"]).strip()
            persona = str(char_info.get("persona", "")).strip()
            if char_name and persona:
                doc_id = f"char_{char_name.lower().replace(' ', '_')}"
                entries.append(
                    KnowledgeEntry(
                        document_id=doc_id,
                        source_type="character",
                        entity_id=char_name,
                        title=f"Character Persona: {char_name}",
                        content=f"Persona and Background of {char_name}:\n{persona}",
                        metadata={
                            "character_name": char_name,
                            "persona": persona,
                        },
                    )
                )

    # 3. Object / Item Description Knowledge
    world_state = record.get("world_state", {})
    if isinstance(world_state, dict):
        obj_descs = world_state.get("object_descriptions", {})
        if isinstance(obj_descs, dict):
            for obj_name, obj_desc in obj_descs.items():
                obj_name_clean = str(obj_name).strip()
                obj_desc_clean = str(obj_desc).strip()
                if obj_name_clean and obj_desc_clean:
                    doc_id = f"item_{obj_name_clean.lower().replace(' ', '_')}"
                    entries.append(
                        KnowledgeEntry(
                            document_id=doc_id,
                            source_type="item",
                            entity_id=obj_name_clean,
                            title=f"Item Description: {obj_name_clean.capitalize()}",
                            content=f"Description of {obj_name_clean}:\n{obj_desc_clean}",
                            metadata={
                                "item_name": obj_name_clean,
                                "description": obj_desc_clean,
                            },
                        )
                    )

    return entries


def load_knowledge_from_light_dataset(
    dataset_path: str, max_records: Optional[int] = None
) -> List[KnowledgeEntry]:
    """
    Scans a LIGHT dataset JSONL file and extracts unique stable knowledge entries.

    Deduplicates entries based on document_id so that each unique setting, character,
    or item is represented once with complete content.

    IMPORTANT: Does not alter or rewrite the source dataset file.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"LIGHT dataset file not found: {dataset_path}")

    seen_ids: Set[str] = set()
    unique_entries: List[KnowledgeEntry] = []

    with open(dataset_path, "r", encoding="utf-8") as f:
        count = 0
        for line in f:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            extracted = extract_knowledge_from_light_record(record)

            for entry in extracted:
                if entry.document_id not in seen_ids:
                    seen_ids.add(entry.document_id)
                    unique_entries.append(entry)

            count += 1
            if max_records is not None and count >= max_records:
                break

    return unique_entries
