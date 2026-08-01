"""
risk_engine.py

Rule-based risk scoring: medication x chief complaint -> urgency tier.

Deliberately rule-based, not ML, for the hackathon. This is a feature,
not a corner cut: a transparent rule table is easier to demo, easier for
a clinician (Kien) to sanity-check live, and matches the "explainability
isn't optional" positioning of the whole project.

Extend RISK_RULES with real high-risk pairs Kien knows from practice.
"""

# Each rule: medication name (lowercase, matches medication_extractor output)
# -> list of (complaint_keyword, tier, reason)
# tier is one of: "urgent", "priority", "routine"
RISK_RULES = {
    "warfarin": [
        ("bleeding", "urgent", "Warfarin + active bleeding: high risk of hemorrhage, hold and reassess immediately."),
        ("stool", "urgent", "Warfarin with GI bleeding symptoms: hold dose, urgent INR check needed."),
        ("fall", "priority", "Warfarin + fall risk: assess for occult bleeding before resuming."),
    ],
    "insulin glargine": [
        ("confus", "urgent", "Insulin + altered mental status: rule out hypoglycemia before resuming dose."),
        ("mental status", "urgent", "Insulin + altered mental status: rule out hypoglycemia before resuming dose."),
    ],
    "metformin": [
        ("confus", "priority", "Metformin + altered mental status: check renal function and lactic acidosis risk."),
    ],
    "clopidogrel": [
        ("chest pain", "priority", "Clopidogrel + chest pain post-stent: do not hold without cardiology input."),
        ("bleeding", "urgent", "Clopidogrel + bleeding: antiplatelet effect increases hemorrhage risk."),
    ],
    "aspirin": [
        ("bleeding", "urgent", "Aspirin + active bleeding: antiplatelet effect increases hemorrhage risk."),
    ],
    "digoxin": [
        ("confus", "urgent", "Digoxin + altered mental status: assess for digoxin toxicity."),
    ],
    "amiodarone": [
        ("chest pain", "priority", "Amiodarone + chest pain: arrhythmia workup takes priority, do not resume blindly."),
    ],
    "furosemide": [
        ("confus", "priority", "Furosemide + altered mental status: check for dehydration/electrolyte imbalance."),
    ],
}

DEFAULT_TIER = "routine"
DEFAULT_REASON = "No specific interaction flagged for this medication and complaint. Routine reconciliation."

TIER_ORDER = {"urgent": 0, "priority": 1, "routine": 2}


def score_medication(med_name: str, chief_complaint: str) -> dict:
    """
    Returns: {"medication": ..., "tier": ..., "reason": ...}
    """
    med_name = med_name.lower().strip()
    complaint = chief_complaint.lower()

    rules = RISK_RULES.get(med_name, [])
    for keyword, tier, reason in rules:
        if keyword in complaint:
            return {"medication": med_name, "tier": tier, "reason": reason}

    return {"medication": med_name, "tier": DEFAULT_TIER, "reason": DEFAULT_REASON}


def score_all(medications: list[dict], chief_complaint: str) -> list[dict]:
    """
    medications: output of medication_extractor.extract_medications()
    Returns a list sorted urgent -> priority -> routine.
    """
    scored = []
    for med in medications:
        result = score_medication(med["name"], chief_complaint)
        result["dosage"] = med.get("dosage", "unspecified")
        scored.append(result)

    scored.sort(key=lambda r: TIER_ORDER.get(r["tier"], 99))
    return scored


if __name__ == "__main__":
    meds = [{"name": "warfarin", "dosage": "5mg daily"}, {"name": "omeprazole", "dosage": "20mg daily"}]
    for r in score_all(meds, "GI bleeding, dark stools"):
        print(r)
