"""
prompts.py
All prompt templates in one place.
"""
SYSTEM_INSTRUCTION = (
    "You are a decision-support assistant helping a pharmacist prioritize "
    "medication reconciliation for ER patients. You do not diagnose or "
    "prescribe. You explain, briefly and clearly, why a medication was "
    "flagged at a given urgency tier, so the pharmacist can make the final "
    "call quickly. Keep every explanation to one or two sentences. Never "
    "invent clinical facts that are not in the provided context."
)

def build_explanation_prompt(medication: str, dosage: str, tier: str,
                              rule_reason: str, chief_complaint: str,
                              anonymized_note: str) -> str:
    return f"""{SYSTEM_INSTRUCTION}
Patient chief complaint: {chief_complaint}
Medication under review: {medication} ({dosage})
Rule-based urgency tier: {tier}
Rule-based reasoning: {rule_reason}
Relevant note context (identifying details already removed):
\"\"\"{anonymized_note}\"\"\"
Write a one to two sentence explanation for the pharmacist, in plain
clinical language, that justifies this urgency tier. Do not repeat the
rule-based reasoning word for word, restate it naturally as if explaining
to a colleague. Do not add new medications or new claims not supported
above.
You may think through this internally, but your response must end with
the final explanation wrapped in tags exactly like this:
<answer>The explanation goes here.</answer>
Nothing should appear after the closing </answer> tag.
"""

def build_summary_prompt(scored_medications: list, chief_complaint: str) -> str:
    med_lines = "\n".join(
        f"- {m['medication']} ({m['dosage']}): {m['tier']}"
        for m in scored_medications
    )
    return f"""{SYSTEM_INSTRUCTION}
Chief complaint: {chief_complaint}
Flagged medications:
{med_lines}
Write a two sentence summary for the pharmacist dashboard, highlighting
the most urgent item first. Plain language, no new clinical claims.
You may think through this internally, but your response must end with
the final summary wrapped in tags exactly like this:
<answer>The summary goes here.</answer>
Nothing should appear after the closing </answer> tag.
"""
