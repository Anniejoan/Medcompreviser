from medcompreviser.llm import VLLMChatClient
from medcompreviser.rewrite import QwenRewriter
from medcompreviser.verify import verify_rewrite


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
        base_url="http://127.0.0.1:8000/v1",
        model_name="Qwen/Qwen2.5-14B-Instruct",
    )

    rewriter = QwenRewriter(
        client=client,
        target_grade=6,
        max_attempts=3,
        grade_tolerance=0.5,
        min_grade_drop=1.0,
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

    print("Attempts:", result.attempts)
    print("Source grade:", result.source_grade)
    print("Final grade:", result.final_grade)
    print("Accepted by verifier:", verification.accepted)

    print("\nREWRITTEN TEXT:\n", result.rewritten_text)
    print("\nGLOSSARY:\n", result.glossary)
    print("\nVERIFICATION SUMMARY:\n", verification.summary)

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