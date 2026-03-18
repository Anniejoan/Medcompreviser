from medcompreviser.llm import VLLMChatClient
from medcompreviser.rewrite import QwenRewriter
from medcompreviser.verify import verify_rewrite
from medcompreviser.definitions import DefinitionRefiner
from medcompreviser.semantic_verify import EntailmentVerifier
import os


if __name__ == "__main__":
    source_text = """
Hypertension means high blood pressure. You should take your medication twice daily after meals.
Reduce sodium intake and follow up with your clinician in two weeks.
"""

    patient_profile = {
        "age_group": "older adult",
        "health_literacy": "low",
        "language_preference": "English",
    }

    personalization_plan = {
        "diet": "Use familiar examples when explaining low-salt food choices.",
        "definitions": "Define medical words in plain language.",
    }

    client = VLLMChatClient(
        base_url=os.getenv("VLLM_BASE_URL"),
        model_name="qwen14b",
    )

    rewriter = QwenRewriter(
        client=client,
        target_grade=6,
        max_attempts=3,
        grade_tolerance=0.5,
        min_grade_drop=1.0,
    )

    definition_refiner = DefinitionRefiner(
        client=client,
        max_terms=8,
    )

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

    entailment_verifier = EntailmentVerifier(
        model_name="facebook/bart-large-mnli",
         device="cpu",
        entailment_threshold=0.45,
        contradiction_threshold=0.35,
    )

    matched_source_sentences = [
        mapping.matched_source_sentences for mapping in verification.mappings
    ]

    semantic_verification = entailment_verifier.verify_from_mapping(
        rewritten_sentences=verification.rewritten_sentences,
        matched_source_sentences=matched_source_sentences,
    )


    definition_result = definition_refiner.refine(
        rewritten_text=result.rewritten_text,
        existing_glossary=result.glossary,
    )

    print("Attempts:", result.attempts)
    print("Source grade:", result.source_grade)
    print("Final grade:", result.final_grade)
    print("Accepted by verifier:", verification.accepted)

    print("\nREWRITTEN TEXT:\n", result.rewritten_text)
    print("\nFINAL GLOSSARY:\n", [item.__dict__ for item in definition_result.glossary])
    print("\nVERIFICATION SUMMARY:\n", verification.summary)
    print("\nDEFINITION NOTES:\n", definition_result.notes)
    print("\nSEMANTIC VERIFICATION SUMMARY:\n", semantic_verification.summary)

    if semantic_verification.failed_sentences:
        print("\nSEMANTICALLY FLAGGED SENTENCES:")
        for s in semantic_verification.failed_sentences:
            print("-", s)

    if verification.unsupported_rewritten_sentences:
        print("\nUNSUPPORTED REWRITTEN SENTENCES:")
        for s in verification.unsupported_rewritten_sentences:
            print("-", s)

    if verification.possibly_dropped_source_sentences:
        print("\nPOSSIBLY DROPPED SOURCE SENTENCES:")
        for s in verification.possibly_dropped_source_sentences:
            print("-", s)

    if verification.numeric_mismatches:
        print("\nNUMERIC MISMATCHES:")
        for item in verification.numeric_mismatches:
            print(item)