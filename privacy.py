"""
privacy.py

Strips PII out of patient text BEFORE it ever reaches Gemma, and can restore
it afterward. Same core idea as the LLM Privacy Shield project: regex for
known patterns (MRNs, phone numbers), spaCy NER for names, reversible token
mapping so the text still reads naturally to the model.

Usage:
    shield = PrivacyShield()
    clean_text, mapping = shield.anonymize(raw_text)
    # ... send clean_text to Gemma ...
    restored_text = shield.deanonymize(model_output, mapping)
"""

import re
import spacy

# Load once at import time. If this fails, run:
#   python -m spacy download en_core_web_sm
try:
    _nlp = spacy.load("en_core_web_sm")
except OSError:
    raise RuntimeError(
        "spaCy model not found. Run: python -m spacy download en_core_web_sm"
    )

# Regex patterns for structured PII that spaCy NER won't reliably catch.
_PATTERNS = {
    "MRN": re.compile(r"\bMRN-?\d{4,10}\b", re.IGNORECASE),
    "PHONE": re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}\b"),
}


class PrivacyShield:
    def __init__(self):
        self.nlp = _nlp

    def anonymize(self, text: str):
        """
        Replaces PII in `text` with reversible placeholder tokens.
        Returns (clean_text, mapping) where mapping lets you restore
        the original values later.
        """
        mapping = {}
        working_text = text
        counter = {"PERSON": 0, "MRN": 0, "PHONE": 0, "SSN": 0, "EMAIL": 0}

        # 1. Regex-based structured PII first (deterministic, high confidence)
        for label, pattern in _PATTERNS.items():
            def _replace(match, label=label):
                counter[label] += 1
                token = f"[{label}_{counter[label]}]"
                mapping[token] = match.group(0)
                return token

            working_text = pattern.sub(_replace, working_text)

        # 2. spaCy NER for names (contextual, catches what regex can't)
        doc = self.nlp(working_text)
        # Collect spans first to avoid mutating text while iterating
        spans = [(ent.start_char, ent.end_char, ent.text)
                 for ent in doc.ents if ent.label_ == "PERSON"]

        # Replace from the end so earlier offsets stay valid
        for start, end, original in sorted(spans, key=lambda s: s[0], reverse=True):
            counter["PERSON"] += 1
            token = f"[PERSON_{counter['PERSON']}]"
            mapping[token] = original
            working_text = working_text[:start] + token + working_text[end:]

        return working_text, mapping

    def deanonymize(self, text: str, mapping: dict) -> str:
        """Restores original PII values into text using the saved mapping."""
        restored = text
        for token, original in mapping.items():
            restored = restored.replace(token, original)
        return restored


if __name__ == "__main__":
    # Quick manual test: python privacy.py
    shield = PrivacyShield()
    sample = (
        "Patient Jordan Ellis (MRN-88213) presents with dark stools. "
        "Contact number 555-0110."
    )
    clean, mapping = shield.anonymize(sample)
    print("Anonymized:", clean)
    print("Mapping:", mapping)
    print("Restored:", shield.deanonymize(clean, mapping))
