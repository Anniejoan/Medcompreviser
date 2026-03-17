import textstat

def readability_metrics(text: str) -> dict:
    return {
        "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
        "smog_index": textstat.smog_index(text),
        "flesch_reading_ease": textstat.flesch_reading_ease(text)
    }