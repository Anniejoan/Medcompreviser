from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from medcompreviser.eval import readability_metrics
from medcompreviser.llm import VLLMChatClient


@dataclass
class RewriteResult:
    rewritten_text: str
    glossary: List[Dict[str, str]]
    raw_response: str
    attempts: int
    source_grade: float
    final_grade: float


def build_rewrite_prompt(
    source_text: str,
    target_grade: int = 6,
    patient_profile: Optional[Dict[str, Any]] = None,
    personalization_plan: Optional[Dict[str, Any]] = None,
    track_definitions: bool = True,
    prior_attempt_feedback: Optional[str] = None,
) -> str:
    profile_text = patient_profile or {}
    plan_text = personalization_plan or {}

    definition_instruction = ""
    if track_definitions:
        definition_instruction = """
Also identify medical or difficult words that should be defined for the reader.
Return a glossary list with simple definitions.
"""

    feedback_block = ""
    if prior_attempt_feedback:
        feedback_block = f"""
IMPORTANT RETRY FEEDBACK:
{prior_attempt_feedback}
"""

    return f"""
You are rewriting patient education material for a patient-facing audience.

Your goals:
1. Rewrite the text to approximately grade {target_grade} reading level.
2. Preserve all medically important facts, instructions, warnings, timelines, and quantities.
3. Do not add new medical facts, diagnoses, treatments, or recommendations not supported by the source text.
4. Do not add examples, foods, explanations, or clarifications unless they are explicitly supported by the source text.
5. Use short, clear sentences and common words where possible.
6. Keep the meaning faithful to the original.
7. Apply personalization only if supported by the provided personalization plan.

Patient profile:
{profile_text}

Personalization plan:
{plan_text}

{definition_instruction}

{feedback_block}

Return your answer in exactly this format:

REWRITTEN_TEXT:
<the rewritten patient-friendly text>

GLOSSARY:
- term: definition
- term: definition

Source text:
{source_text}
""".strip()


def parse_rewrite_response(response_text: str):
    rewritten_text = ""
    glossary = []

    if "REWRITTEN_TEXT:" in response_text:
        after_rewrite = response_text.split("REWRITTEN_TEXT:", 1)[1]
    else:
        after_rewrite = response_text

    if "GLOSSARY:" in after_rewrite:
        rewritten_part, glossary_part = after_rewrite.split("GLOSSARY:", 1)
        rewritten_text = rewritten_part.strip()

        for line in glossary_part.splitlines():
            line = line.strip()
            if line.startswith("-") and ":" in line:
                line = line[1:].strip()
                term, definition = line.split(":", 1)
                glossary.append(
                    {"term": term.strip(), "definition": definition.strip()}
                )
    else:
        rewritten_text = after_rewrite.strip()

    return rewritten_text, glossary


class QwenRewriter:
    def __init__(
        self,
        client: VLLMChatClient,
        target_grade: int = 6,
        max_attempts: int = 3,
        grade_tolerance: float = 0.5,
        min_grade_drop: float = 1.0,
    ):
        self.client = client
        self.target_grade = target_grade
        self.max_attempts = max_attempts
        self.grade_tolerance = grade_tolerance
        self.min_grade_drop = min_grade_drop

    def rewrite(
        self,
        source_text: str,
        patient_profile: Optional[Dict[str, Any]] = None,
        personalization_plan: Optional[Dict[str, Any]] = None,
        track_definitions: bool = True,
    ) -> RewriteResult:
        source_metrics = readability_metrics(source_text)
        source_grade = float(source_metrics["flesch_kincaid_grade"])

        feedback = None
        last_raw = ""
        last_text = ""
        last_glossary = []

        for attempt in range(1, self.max_attempts + 1):
            prompt = build_rewrite_prompt(
                source_text=source_text,
                target_grade=self.target_grade,
                patient_profile=patient_profile,
                personalization_plan=personalization_plan,
                track_definitions=track_definitions,
                prior_attempt_feedback=feedback,
            )

            raw = self.client.chat(
                system_prompt="You are a careful medical rewriting assistant.",
                user_prompt=prompt,
                temperature=0.1,
                max_tokens=1400,
            )

            rewritten_text, glossary = parse_rewrite_response(raw)
            rewritten_metrics = readability_metrics(rewritten_text)
            rewritten_grade = float(rewritten_metrics["flesch_kincaid_grade"])

            grade_drop = source_grade - rewritten_grade
            target_hit = rewritten_grade <= (self.target_grade + self.grade_tolerance)
            enough_drop = grade_drop >= self.min_grade_drop

            last_raw = raw
            last_text = rewritten_text
            last_glossary = glossary

            if target_hit and enough_drop:
                return RewriteResult(
                    rewritten_text=rewritten_text,
                    glossary=glossary,
                    raw_response=raw,
                    attempts=attempt,
                    source_grade=source_grade,
                    final_grade=rewritten_grade,
                )

            feedback = (
                f"The prior rewrite did not sufficiently reduce readability. "
                f"Source FK grade: {source_grade:.2f}. "
                f"Previous rewrite FK grade: {rewritten_grade:.2f}. "
                f"Required target: <= {self.target_grade + self.grade_tolerance:.2f}. "
                f"Required minimum drop: {self.min_grade_drop:.2f}. "
                f"Rewrite again using shorter sentences, simpler vocabulary, and clearer phrasing, "
                f"while preserving all medical facts and instructions exactly."
            )

        final_metrics = readability_metrics(last_text)
        return RewriteResult(
            rewritten_text=last_text,
            glossary=last_glossary,
            raw_response=last_raw,
            attempts=self.max_attempts,
            source_grade=source_grade,
            final_grade=float(final_metrics["flesch_kincaid_grade"]),
        )