from medcompreviser.eval import readability_metrics
from medcompreviser.rewrite import build_rewrite_prompt, parse_rewrite_response


if __name__ == "__main__":
    sample_text = (
        "Hypertension means high blood pressure. "
        "You should take your medication twice daily and reduce salt intake."
    )

    prompt = build_rewrite_prompt(
        source_text=sample_text,
        target_grade=6,
        patient_profile={"age_group": "older adult", "health_literacy": "low"},
        personalization_plan={"diet": "reduce sodium intake using familiar foods"},
        track_definitions=True,
    )

    print("PROMPT PREVIEW:\n")
    print(prompt[:1000])

    fake_response = """
REWRITTEN_TEXT:
High blood pressure means your blood pressure is too high. Take your medicine two times each day. Eat less salt to help control your blood pressure.

GLOSSARY:
- hypertension: high blood pressure
- sodium: salt
""".strip()

    parsed = parse_rewrite_response(fake_response)

    print("\nPARSED OUTPUT:\n")
    print(parsed.rewritten_text)
    print(parsed.glossary)

    print("\nREADABILITY:\n")
    print(readability_metrics(parsed.rewritten_text))