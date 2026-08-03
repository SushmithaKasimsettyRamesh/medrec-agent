"""
prompts.py
All prompt templates in one place so they're easy to tune fast during
the hackathon without hunting through app.py.
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
    """
    Builds the prompt sent to Gemma to generate a pharmacist-facing
    explanation for one flagged medication. Note: `anonymized_note` must
    already have PII stripped out by privacy.py before it reaches here.
    """
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

Output format: respond with ONLY the final one-to-two sentence explanation.
Do not restate or repeat any part of this prompt. Do not show reasoning,
steps, or self-corrections. Do not prefix with labels like "Explanation:".
Your entire response must be just the explanation sentence(s), nothing else.
"""

def build_summary_prompt(scored_medications: list, chief_complaint: str) -> str:
    """
    Optional: builds a prompt for a one-paragraph overall case summary,
    useful if you want a top-of-dashboard summary line.
    """
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

Output format: respond with ONLY the two sentence summary. Do not restate
this prompt, show reasoning, or add labels. Your entire response must be
just the summary.
"""
