"""
medication_extractor.py

Pulls a structured list of medications out of free-text discharge notes.

Approach: spaCy's PhraseMatcher against a known medication vocabulary.
This is deliberately simple and explainable for a one-day build. It is
NOT a full clinical NLP system, it's good enough to demo and extend.

If you have time during the hackathon, swap the vocabulary list for a
bigger one (RxNorm export) without changing anything else in this file.
"""

import re
import spacy
from spacy.matcher import PhraseMatcher

try:
    _nlp = spacy.load("en_core_web_sm")
except OSError:
    raise RuntimeError(
        "spaCy model not found. Run: python -m spacy download en_core_web_sm"
    )

# Starter vocabulary. Add more as you test with real synthetic cases.
# Keep lowercase, matcher is case-insensitive via attr="LOWER".
KNOWN_MEDICATIONS = [
    "warfarin", "metoprolol", "omeprazole", "ibuprofen",
    "insulin glargine", "metformin", "lisinopril",
    "aspirin", "clopidogrel", "atorvastatin",
    "levothyroxine", "multivitamin",
    "amiodarone", "digoxin", "furosemide", "prednisone",
]

_matcher = PhraseMatcher(_nlp.vocab, attr="LOWER")
_patterns = [_nlp.make_doc(med) for med in KNOWN_MEDICATIONS]
_matcher.add("MEDICATION", _patterns)

# Pulls out a dosage right after a matched medication, e.g. "warfarin 5mg daily"
_DOSAGE_RE = re.compile(
    r"(\d+\s?(?:mg|mcg|units?|ml))\s*(daily|nightly|twice daily|once daily|as needed)?",
    re.IGNORECASE,
)


def extract_medications(text: str) -> list[dict]:
    """
    Returns a list of dicts:
        [{"name": "warfarin", "dosage": "5mg daily"}, ...]
    Deduplicated by medication name.
    """
    doc = _nlp(text)
    matches = _matcher(doc)

    found = {}
    for match_id, start, end in matches:
        span = doc[start:end]
        med_name = span.text.lower()

        # Look for a dosage in the ~40 chars right after the mention
        tail = text[span.end_char: span.end_char + 40]
        dosage_match = _DOSAGE_RE.search(tail)
        dosage = dosage_match.group(0).strip() if dosage_match else "unspecified"

        if med_name not in found:
            found[med_name] = dosage

    return [{"name": name, "dosage": dosage} for name, dosage in found.items()]


if __name__ == "__main__":
    sample = (
        "Home medications include warfarin 5mg daily for atrial fibrillation, "
        "metoprolol 25mg twice daily, and omeprazole 20mg daily. Patient also "
        "takes ibuprofen occasionally for knee pain."
    )
    for med in extract_medications(sample):
        print(med)
