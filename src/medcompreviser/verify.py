from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
import re


@dataclass
class SentenceMapping:
    rewritten_index: int
    rewritten_sentence: str
    matched_source_indices: List[int]
    matched_source_sentences: List[str]
    overlap_score: float
    unsupported: bool
    notes: str


@dataclass
class VerificationResult:
    source_sentences: List[str]
    rewritten_sentences: List[str]
    mappings: List[SentenceMapping]
    possibly_dropped_source_indices: List[int]
    possibly_dropped_source_sentences: List[str]
    unsupported_rewritten_indices: List[int]
    unsupported_rewritten_sentences: List[str]
    numeric_mismatches: List[Dict[str, Any]]
    accepted: bool
    summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_sentences": self.source_sentences,
            "rewritten_sentences": self.rewritten_sentences,
            "mappings": [asdict(m) for m in self.mappings],
            "possibly_dropped_source_indices": self.possibly_dropped_source_indices,
            "possibly_dropped_source_sentences": self.possibly_dropped_source_sentences,
            "unsupported_rewritten_indices": self.unsupported_rewritten_indices,
            "unsupported_rewritten_sentences": self.unsupported_rewritten_sentences,
            "numeric_mismatches": self.numeric_mismatches,
            "accepted": self.accepted,
            "summary": self.summary,
        }


_SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?])\s+")
_TOKEN_REGEX = re.compile(r"\b[a-z0-9]+\b", re.IGNORECASE)
_NUMBER_REGEX = re.compile(
    r"\b(?:\d+(?:\.\d+)?(?:-\d+(?:\.\d+)?)?|one|two|three|four|five|six|seven|eight|nine|ten)\b",
    re.IGNORECASE,
)


def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = _SENTENCE_SPLIT_REGEX.split(text)
    return [p.strip() for p in parts if p.strip()]


def normalize_tokens(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_REGEX.findall(text)]


def remove_stopwords(tokens: List[str]) -> List[str]:
    stopwords = {
        "the", "a", "an", "and", "or", "but", "if", "then", "this", "that", "these",
        "those", "is", "are", "was", "were", "be", "been", "being", "to", "of", "in",
        "on", "for", "with", "at", "by", "from", "as", "it", "its", "into", "than",
        "about", "after", "before", "during", "over", "under", "again", "you", "your",
        "they", "them", "their", "he", "she", "his", "her", "we", "our", "i", "me",
        "my", "do", "does", "did", "can", "could", "should", "would", "may", "might",
        "will", "must", "have", "has", "had",
    }
    return [t for t in tokens if t not in stopwords]


def get_ngrams(tokens: List[str], n: int = 3) -> set[Tuple[str, ...]]:
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def overlap_score(source_sentence: str, rewritten_sentence: str) -> float:
    source_tokens = remove_stopwords(normalize_tokens(source_sentence))
    rewritten_tokens = remove_stopwords(normalize_tokens(rewritten_sentence))

    if not source_tokens or not rewritten_tokens:
        return 0.0

    source_unigrams = set(source_tokens)
    rewritten_unigrams = set(rewritten_tokens)

    unigram_overlap = len(source_unigrams & rewritten_unigrams) / max(1, len(rewritten_unigrams))

    source_trigrams = get_ngrams(source_tokens, n=3)
    rewritten_trigrams = get_ngrams(rewritten_tokens, n=3)

    if rewritten_trigrams:
        trigram_overlap = len(source_trigrams & rewritten_trigrams) / max(1, len(rewritten_trigrams))
    else:
        trigram_overlap = 0.0

    return 0.6 * unigram_overlap + 0.4 * trigram_overlap


def extract_number_like_strings(text: str) -> List[str]:
    return [m.group(0).lower() for m in _NUMBER_REGEX.finditer(text)]


def find_best_source_matches(
    source_sentences: List[str],
    rewritten_sentence: str,
    score_threshold: float = 0.18,
    top_k: int = 2,
) -> List[Tuple[int, float]]:
    scored = []
    for idx, src in enumerate(source_sentences):
        score = overlap_score(src, rewritten_sentence)
        scored.append((idx, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    keep = [(idx, score) for idx, score in scored[:top_k] if score >= score_threshold]
    return keep


def verify_rewrite(
    source_text: str,
    rewritten_text: str,
    score_threshold: float = 0.18,
    dropped_source_threshold: float = 0.10,
) -> VerificationResult:
    source_sentences = split_sentences(source_text)
    rewritten_sentences = split_sentences(rewritten_text)

    mappings: List[SentenceMapping] = []
    source_covered = set()
    unsupported_rewritten_indices: List[int] = []
    unsupported_rewritten_sentences: List[str] = []
    numeric_mismatches: List[Dict[str, Any]] = []

    for r_idx, r_sent in enumerate(rewritten_sentences):
        matches = find_best_source_matches(
            source_sentences=source_sentences,
            rewritten_sentence=r_sent,
            score_threshold=score_threshold,
            top_k=2,
        )

        matched_indices = [idx for idx, _ in matches]
        matched_sentences = [source_sentences[idx] for idx in matched_indices]
        best_score = matches[0][1] if matches else 0.0

        unsupported = len(matches) == 0
        notes = "No sufficiently similar source sentence found." if unsupported else "Mapped successfully."

        if unsupported:
            unsupported_rewritten_indices.append(r_idx)
            unsupported_rewritten_sentences.append(r_sent)
        else:
            source_covered.update(matched_indices)

            source_numbers = []
            for s in matched_sentences:
                source_numbers.extend(extract_number_like_strings(s))
            rewritten_numbers = extract_number_like_strings(r_sent)

            if source_numbers and rewritten_numbers:
                if sorted(source_numbers) != sorted(rewritten_numbers):
                    numeric_mismatches.append(
                        {
                            "rewritten_index": r_idx,
                            "rewritten_sentence": r_sent,
                            "source_indices": matched_indices,
                            "source_sentences": matched_sentences,
                            "source_numbers": source_numbers,
                            "rewritten_numbers": rewritten_numbers,
                        }
                    )
                    notes = "Mapped, but numbers/timings may have changed."

        mappings.append(
            SentenceMapping(
                rewritten_index=r_idx,
                rewritten_sentence=r_sent,
                matched_source_indices=matched_indices,
                matched_source_sentences=matched_sentences,
                overlap_score=best_score,
                unsupported=unsupported,
                notes=notes,
            )
        )

    possibly_dropped_source_indices: List[int] = []
    possibly_dropped_source_sentences: List[str] = []

    for s_idx, s_sent in enumerate(source_sentences):
        best_rewrite_score = 0.0
        for r_sent in rewritten_sentences:
            score = overlap_score(s_sent, r_sent)
            best_rewrite_score = max(best_rewrite_score, score)

        if best_rewrite_score < dropped_source_threshold:
            possibly_dropped_source_indices.append(s_idx)
            possibly_dropped_source_sentences.append(s_sent)

    accepted = len(unsupported_rewritten_indices) == 0 and len(numeric_mismatches) == 0

    summary = {
        "num_source_sentences": len(source_sentences),
        "num_rewritten_sentences": len(rewritten_sentences),
        "num_unsupported_rewritten_sentences": len(unsupported_rewritten_indices),
        "num_possibly_dropped_source_sentences": len(possibly_dropped_source_indices),
        "num_numeric_mismatches": len(numeric_mismatches),
    }

    return VerificationResult(
        source_sentences=source_sentences,
        rewritten_sentences=rewritten_sentences,
        mappings=mappings,
        possibly_dropped_source_indices=possibly_dropped_source_indices,
        possibly_dropped_source_sentences=possibly_dropped_source_sentences,
        unsupported_rewritten_indices=unsupported_rewritten_indices,
        unsupported_rewritten_sentences=unsupported_rewritten_sentences,
        numeric_mismatches=numeric_mismatches,
        accepted=accepted,
        summary=summary,
    )