from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


@dataclass
class RewriteResult:
    rewritten_text: str
    glossary: List[Dict[str, str]]
    raw_response: str


class BaseRewriter:
    def rewrite(
        self,
        source_text: str,
        target_grade: int = 6,
        patient_profile: Optional[Dict[str, Any]] = None,
        personalization_plan: Optional[Dict[str, Any]] = None,
        track_definitions: bool = True,
    ) -> RewriteResult:
        raise NotImplementedError


def build_rewrite_prompt(
    source_text: str,
    target_grade: int = 6,
    patient_profile: Optional[Dict[str, Any]] = None,
    personalization_plan: Optional[Dict[str, Any]] = None,
    track_definitions: bool = True,
) -> str:
    """
    Build a structured prompt for rewriting patient education material.
    """

    profile_text = patient_profile if patient_profile is not None else {}
    plan_text = personalization_plan if personalization_plan is not None else {}

    definition_instruction = ""
    if track_definitions:
        definition_instruction = """
Also identify medical or difficult words that should be defined for the reader.
Return a glossary list with simple definitions.
"""

    prompt = f"""
You are rewriting patient education material for a patient-facing audience.

Your goals:
1. Rewrite the text to approximately grade {target_grade} reading level.
2. Preserve all medically important facts, instructions, warnings, timelines, and quantities.
3. Do not add new medical facts, diagnoses, treatments, or recommendations not supported by the source text.
4. Use short, clear sentences and common words where possible.
5. Keep the meaning faithful to the original.
6. Apply personalization only if supported by the provided personalization plan.

Patient profile:
{profile_text}

Personalization plan:
{plan_text}

{definition_instruction}

Return your answer in exactly this format:

REWRITTEN_TEXT:
<the rewritten patient-friendly text>

GLOSSARY:
- term: definition
- term: definition

Source text:
{source_text}
""".strip()

    return prompt


def parse_rewrite_response(response_text: str) -> RewriteResult:
    """
    Parse the model response into rewritten text + glossary.
    This is intentionally simple and can be hardened later.
    """
    rewritten_text = ""
    glossary: List[Dict[str, str]] = []

    if "REWRITTEN_TEXT:" in response_text:
        after_rewrite = response_text.split("REWRITTEN_TEXT:", 1)[1]
    else:
        after_rewrite = response_text

    if "GLOSSARY:" in after_rewrite:
        rewritten_part, glossary_part = after_rewrite.split("GLOSSARY:", 1)
        rewritten_text = rewritten_part.strip()

        for line in glossary_part.splitlines():
            line = line.strip()
            if not line.startswith("-"):
                continue
            line = line[1:].strip()
            if ":" in line:
                term, definition = line.split(":", 1)
                glossary.append(
                    {"term": term.strip(), "definition": definition.strip()}
                )
    else:
        rewritten_text = after_rewrite.strip()

    return RewriteResult(
        rewritten_text=rewritten_text,
        glossary=glossary,
        raw_response=response_text,
    )