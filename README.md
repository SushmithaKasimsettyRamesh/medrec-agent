# Medication Reconciliation Assistant

Privacy-first, agentic decision support for ER medication reconciliation.
Built for Build with Gemma NYC.

## What it does

1. Loads a synthetic ER patient case (note: contains fake PII on purpose, to prove the privacy layer works)
2. Strips PII (names, MRNs, phone numbers) before anything touches Gemma
3. Extracts the patient's home medications from the note
4. Scores each medication's urgency based on medication x chief complaint (rule-based, transparent)
5. Asks Gemma to write a short plain-language explanation for each flag
6. Shows a prioritized list a pharmacist could actually use

## Setup (5 minutes)

```bash
# 1. Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the spaCy English model (only needed once)
python -m spacy download en_core_web_sm

# 4. Add your Gemma API key
# Go to https://aistudio.google.com -> "Get API Key"
# Paste it into .env as GEMMA_API_KEY=your_key_here

# 5. Run the app
streamlit run app.py
```

The app opens at http://localhost:8501

## Testing individual pieces without the full app

Each file has a small test at the bottom, runnable directly:

```bash
python privacy.py               # test PII stripping
python medication_extractor.py  # test medication extraction
python risk_engine.py           # test risk scoring
python gemma_client.py          # test Gemma API connection
```

Useful for debugging one layer at a time instead of the whole Streamlit app.

## Scope

Decision-support only. Not diagnosis or treatment. All patient data is
synthetic, no real EMR or patient data is used anywhere in this project.
