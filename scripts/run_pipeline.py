import argparse
import json
import os
from dataclasses import asdict

from medcompreviser.io_utils import read_pdf_text, ensure_parent_dir
from medcompreviser.eval import readability_metrics
from medcompreviser.llm import VLLMChatClient
from medcompreviser.rewrite import QwenRewriter
from medcompreviser.verify import verify_rewrite
from medcompreviser.definitions import DefinitionRefiner
from medcompreviser.semantic_verify import EntailmentVerifier


def main():
    parser = argparse.ArgumentParser(description="Run Medcompreviser pipeline on a PDF.")
    parser.add_argument("--input", required=True, help="Path to input PDF")
    parser.add_argument("--output", required=True, help="Path to output JSON")
    parser.add_argument("--target-grade", type=int, default=6, help="Target reading grade")
    parser.add_argument("--model-name", default="qwen14b", help="Served vLLM model name")
    args = parser.parse_args()

    source_text = read_pdf_text(args.input)
    source_metrics = readability_metrics(source_text)

    client = VLLMChatClient(
        base_url=os.getenv("VLLM_BASE_URL"),
        model_name=args.model_name,
    )

    rewriter = QwenRewriter(
        client=client,
        target_grade=args.target_grade,
        max_attempts=3,
        grade_tolerance=0.5,
        min_grade_drop=1.0,
    )

    definition_refiner = DefinitionRefiner(
        client=client,
        max_terms=8,
    )

    entailment_verifier = EntailmentVerifier(
        model_name="facebook/bart-large-mnli",
        device="cpu",  # keep CPU to avoid GPU contention with vLLM
        entailment_threshold=0.45,
        contradiction_threshold=0.35,
    )

    # For now, no personalization yet
    patient_profile = {}
    personalization_plan = {}

    result = rewriter.rewrite(
        source_text=source_text,
        patient_profile=patient_profile,
        personalization_plan=personalization_plan,
        track_definitions=True,
    )

    verification = verify_rewrite(
        source_text=source_text,
        rewritten_text=result.rewritten_text,
    )

    definition_result = definition_refiner.refine(
        rewritten_text=result.rewritten_text,
        existing_glossary=result.glossary,
    )

    matched_source_sentences = [
        mapping.matched_source_sentences for mapping in verification.mappings
    ]

    semantic_verification = entailment_verifier.verify_from_mapping(
        rewritten_sentences=verification.rewritten_sentences,
        matched_source_sentences=matched_source_sentences,
    )

    output = {
        "input_path": args.input,
        "target_grade": args.target_grade,
        "source_text": source_text,
        "source_readability": source_metrics,
        "rewritten_text": result.rewritten_text,
        "glossary": [asdict(item) for item in definition_result.glossary],
        "rewrite_result": {
            "attempts": result.attempts,
            "source_grade": result.source_grade,
            "final_grade": result.final_grade,
            "raw_response": result.raw_response,
        },
        "verification": verification.to_dict(),
        "definition_result": definition_result.to_dict(),
        "semantic_verification": semantic_verification.to_dict(),
    }

    ensure_parent_dir(args.output)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Attempts: {result.attempts}")
    print(f"Source grade: {result.source_grade}")
    print(f"Final grade: {result.final_grade}")
    print(f"Output written to: {args.output}")

    print("\nREWRITTEN TEXT:\n")
    print(result.rewritten_text)

    print("\nVERIFICATION SUMMARY:\n", verification.summary)
    print("\nSEMANTIC VERIFICATION SUMMARY:\n", semantic_verification.summary)


if __name__ == "__main__":
    main()