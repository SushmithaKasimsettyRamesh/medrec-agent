"""
app.py

Streamlit dashboard. Orchestrates the full pipeline:

  1. Load a synthetic patient case
  2. Anonymize the discharge note (privacy.py)          <- privacy layer
  3. Extract medications from the anonymized note (medication_extractor.py)
  4. Score each medication for urgency (risk_engine.py)
  5. Generate a plain-language explanation per flag (gemma_client.py + prompts.py)
  6. Display a prioritized, pharmacist-facing list

Run with:  streamlit run app.py
"""

import json
import streamlit as st

from privacy import PrivacyShield
from medication_extractor import extract_medications
from risk_engine import score_all
from prompts import build_explanation_prompt
from gemma_client import GemmaClient

st.set_page_config(page_title="Med Rec Agent", page_icon="💊", layout="wide")

TIER_COLORS = {"urgent": "🔴", "priority": "🟡", "routine": "🟢"}


@st.cache_resource
def get_privacy_shield():
    return PrivacyShield()


@st.cache_resource
def get_gemma_client():
    try:
        return GemmaClient()
    except RuntimeError:
        return None  # allow the app to load even without an API key set yet


@st.cache_data
def load_cases():
    with open("data/synthetic_cases.json") as f:
        return json.load(f)


def _fallback_text(item: dict) -> str:
    reason = item.get("reason")
    if reason:
        return reason
    return (
        f"No specific concern flagged for {item['medication']} given this "
        "complaint; routine reconciliation."
    )


def run_pipeline(case: dict, gemma: GemmaClient | None):
    shield = get_privacy_shield()

    # Step 1: strip PII before anything touches the model
    anonymized_note, mapping = shield.anonymize(case["discharge_note"])

    # Step 2: extract medications from the anonymized text
    medications = extract_medications(anonymized_note)

    # Step 3: rule-based risk scoring
    scored = score_all(medications, case["chief_complaint"])

    # Step 4: generate an explanation per flagged medication
    for item in scored:
        if gemma is None:
            item["explanation"] = (
                "[Add GEMMA_API_KEY in .env to generate live explanations. "
                f"Rule-based reason: {item['reason']}]"
            )
            continue
        prompt = build_explanation_prompt(
            medication=item["medication"],
            dosage=item["dosage"],
            tier=item["tier"],
            rule_reason=item["reason"],
            chief_complaint=case["chief_complaint"],
            anonymized_note=anonymized_note,
        )
        try:
            item["explanation"] = gemma.generate(prompt, fallback=_fallback_text(item))
        except RuntimeError as e:
            item["explanation"] = f"[Gemma call failed: {e}]"

    return anonymized_note, mapping, scored


def main():
    st.title("💊 Medication Reconciliation Assistant")
    st.caption(
        "Privacy-first, agentic decision support for ER medication reconciliation. "
        "Patient data is anonymized before it ever reaches the model."
    )

    cases = load_cases()
    gemma = get_gemma_client()

    if gemma is None:
        st.warning(
            "No GEMMA_API_KEY found in .env. Explanations will show the "
            "rule-based reasoning only. Add your key from aistudio.google.com "
            "to enable live Gemma explanations.",
            icon="⚠️",
        )

    case_labels = [f"{c['case_id']} — {c['chief_complaint']}" for c in cases]
    selected_idx = st.selectbox(
        "Select a patient case", range(len(cases)),
        format_func=lambda i: case_labels[i],
    )
    case = cases[selected_idx]

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Chief complaint")
        st.write(case["chief_complaint"])

        st.subheader("Discharge note (raw, contains PII)")
        st.text_area("Raw note", case["discharge_note"], height=160, disabled=True, label_visibility="collapsed")

    if st.button("Run medication reconciliation", type="primary"):
        with st.spinner("Anonymizing, extracting medications, scoring risk..."):
            anonymized_note, mapping, scored = run_pipeline(case, gemma)

        with col2:
            st.subheader("Anonymized note (what the model actually sees)")
            st.text_area("Anonymized", anonymized_note, height=160, disabled=True, label_visibility="collapsed")
            st.caption(f"{len(mapping)} PII item(s) stripped before reaching Gemma.")

        st.divider()
        st.subheader("Prioritized medication list")

        if not scored:
            st.info("No known medications detected in this note.")
        else:
            for item in scored:
                icon = TIER_COLORS.get(item["tier"], "⚪")
                with st.container(border=True):
                    st.markdown(
                        f"### {icon} {item['medication'].title()} — {item['dosage']}  "
                        f"`{item['tier'].upper()}`"
                    )
                    st.write(item["explanation"])

    st.divider()
    st.caption(
        "Decision-support only. Not a diagnostic tool. All patient data on "
        "this screen is synthetic. Built for Build with Gemma NYC."
    )


if __name__ == "__main__":
    main()
