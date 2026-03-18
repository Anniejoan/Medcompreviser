from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass
class EntailmentCheck:
    rewritten_index: int
    premise: str
    hypothesis: str
    entailment: float
    contradiction: float
    neutral: float
    accepted: bool
    notes: str


@dataclass
class SemanticVerificationResult:
    checks: List[EntailmentCheck]
    failed_indices: List[int]
    failed_sentences: List[str]
    accepted: bool
    summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "checks": [asdict(c) for c in self.checks],
            "failed_indices": self.failed_indices,
            "failed_sentences": self.failed_sentences,
            "accepted": self.accepted,
            "summary": self.summary,
        }


class EntailmentVerifier:
    def __init__(
        self,
        model_name: str = "facebook/bart-large-mnli",
        device: Optional[str] = None,
        entailment_threshold: float = 0.55,
        contradiction_threshold: float = 0.30,
    ):
        self.model_name = model_name
        self.entailment_threshold = entailment_threshold
        self.contradiction_threshold = contradiction_threshold

        if device is None:
            device = "cpu"
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # Common MNLI label mapping for BART:
        # 0 = contradiction, 1 = neutral, 2 = entailment
        self.contradiction_idx = 0
        self.neutral_idx = 1
        self.entailment_idx = 2

    @torch.inference_mode()
    def score_pair(self, premise: str, hypothesis: str) -> Dict[str, float]:
        encoded = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        logits = self.model(**encoded).logits
        probs = torch.softmax(logits, dim=-1)[0]

        return {
            "contradiction": float(probs[self.contradiction_idx].item()),
            "neutral": float(probs[self.neutral_idx].item()),
            "entailment": float(probs[self.entailment_idx].item()),
        }

    def verify_from_mapping(
        self,
        rewritten_sentences: List[str],
        matched_source_sentences: List[List[str]],
    ) -> SemanticVerificationResult:
        checks: List[EntailmentCheck] = []
        failed_indices: List[int] = []
        failed_sentences: List[str] = []

        for idx, rewritten_sentence in enumerate(rewritten_sentences):
            source_candidates = matched_source_sentences[idx] if idx < len(matched_source_sentences) else []

            if not source_candidates:
                checks.append(
                    EntailmentCheck(
                        rewritten_index=idx,
                        premise="",
                        hypothesis=rewritten_sentence,
                        entailment=0.0,
                        contradiction=0.0,
                        neutral=1.0,
                        accepted=False,
                        notes="No mapped source sentence available for semantic verification.",
                    )
                )
                failed_indices.append(idx)
                failed_sentences.append(rewritten_sentence)
                continue

            premise = " ".join(source_candidates)
            scores = self.score_pair(premise=premise, hypothesis=rewritten_sentence)

            accepted = (
                scores["entailment"] >= self.entailment_threshold
                and scores["contradiction"] <= self.contradiction_threshold
            )

            notes = "Entailment check passed." if accepted else "Low entailment or high contradiction."

            checks.append(
                EntailmentCheck(
                    rewritten_index=idx,
                    premise=premise,
                    hypothesis=rewritten_sentence,
                    entailment=scores["entailment"],
                    contradiction=scores["contradiction"],
                    neutral=scores["neutral"],
                    accepted=accepted,
                    notes=notes,
                )
            )

            if not accepted:
                failed_indices.append(idx)
                failed_sentences.append(rewritten_sentence)

        accepted = len(failed_indices) == 0

        summary = {
            "num_checked_sentences": len(checks),
            "num_failed_sentences": len(failed_indices),
            "entailment_threshold": self.entailment_threshold,
            "contradiction_threshold": self.contradiction_threshold,
            "model_name": self.model_name,
        }

        return SemanticVerificationResult(
            checks=checks,
            failed_indices=failed_indices,
            failed_sentences=failed_sentences,
            accepted=accepted,
            summary=summary,
        )