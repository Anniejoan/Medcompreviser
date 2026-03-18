from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
import json
import re

from medcompreviser.llm import VLLMChatClient


@dataclass
class DefinitionItem:
    term: str
    definition: str


@dataclass
class DefinitionResult:
    glossary: List[DefinitionItem]
    raw_response: str
    accepted: bool
    notes: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "glossary": [asdict(item) for item in self.glossary],
            "raw_response": self.raw_response,
            "accepted": self.accepted,
            "notes": self.notes,
        }


def normalize_term(term: str) -> str:
    return re.sub(r"\s+", " ", term.strip().lower())


def deduplicate_glossary(glossary: List[Dict[str, str]]) -> List[DefinitionItem]:
    seen = set()
    deduped: List[DefinitionItem] = []

    for item in glossary:
        term = item.get("term", "").strip()
        definition = item.get("definition", "").strip()

        if not term or not definition:
            continue

        key = normalize_term(term)
        if key in seen:
            continue

        seen.add(key)
        deduped.append(DefinitionItem(term=term, definition=definition))

    return deduped


def build_definition_prompt(
    rewritten_text: str,
    existing_glossary: Optional[List[Dict[str, str]]] = None,
    max_terms: int = 10,
) -> str:
    existing_glossary = existing_glossary or []

    return f"""
You are reviewing a patient education rewrite.

Task:
1. Find medical or difficult words in the rewritten text that may still need simple definitions.
2. Reuse or improve any existing glossary entries if helpful.
3. Only include terms that are actually present in the rewritten text.
4. Keep definitions short, plain, and easy for a patient with low health literacy to understand.
5. Do not define common everyday words.
6. Return at most {max_terms} terms.

Return ONLY valid JSON in this exact format:

{{
  "glossary": [
    {{"term": "example term", "definition": "simple explanation"}},
    {{"term": "example term", "definition": "simple explanation"}}
  ]
}}

Existing glossary:
{json.dumps(existing_glossary, indent=2)}

Rewritten text:
{rewritten_text}
""".strip()


def parse_definition_response(response_text: str) -> DefinitionResult:
    try:
        data = json.loads(response_text)
        glossary_raw = data.get("glossary", [])
        glossary = deduplicate_glossary(glossary_raw)
        return DefinitionResult(
            glossary=glossary,
            raw_response=response_text,
            accepted=True,
            notes="Parsed JSON successfully.",
        )
    except Exception as e:
        return DefinitionResult(
            glossary=[],
            raw_response=response_text,
            accepted=False,
            notes=f"Failed to parse JSON response: {e}",
        )


def merge_glossaries(
    existing_glossary: Optional[List[Dict[str, str]]],
    new_glossary: List[DefinitionItem],
) -> List[DefinitionItem]:
    merged: List[Dict[str, str]] = []
    existing_glossary = existing_glossary or []

    for item in existing_glossary:
        if isinstance(item, dict):
            merged.append(
                {
                    "term": item.get("term", "").strip(),
                    "definition": item.get("definition", "").strip(),
                }
            )

    for item in new_glossary:
        merged.append({"term": item.term, "definition": item.definition})

    return deduplicate_glossary(merged)


class DefinitionRefiner:
    def __init__(self, client: VLLMChatClient, max_terms: int = 10):
        self.client = client
        self.max_terms = max_terms

    def refine(
        self,
        rewritten_text: str,
        existing_glossary: Optional[List[Dict[str, str]]] = None,
    ) -> DefinitionResult:
        prompt = build_definition_prompt(
            rewritten_text=rewritten_text,
            existing_glossary=existing_glossary,
            max_terms=self.max_terms,
        )

        raw = self.client.chat(
            system_prompt="You are a careful medical plain-language assistant. Return only valid JSON.",
            user_prompt=prompt,
            temperature=0.1,
            max_tokens=800,
        )

        parsed = parse_definition_response(raw)

        if not parsed.accepted:
            return parsed

        merged = merge_glossaries(existing_glossary, parsed.glossary)
        return DefinitionResult(
            glossary=merged,
            raw_response=raw,
            accepted=True,
            notes="Definitions refined and merged successfully.",
        )