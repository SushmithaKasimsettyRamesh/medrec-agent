"""
gemma_client.py

Thin wrapper around Google's Generative Language API for calling Gemma
models. Uses the same API surface as Gemini, just with a Gemma model name.

Get your API key at https://aistudio.google.com -> "Get API Key", then
either put it in your .env file locally as GEMMA_API_KEY=..., or in
Streamlit Cloud's Secrets panel as GEMMA_API_KEY = "..." for the deployed app.
"""

import os
import re
import requests
from dotenv import load_dotenv

load_dotenv()


def _get_secret(key: str, default=None):
    """
    Looks for a config value in this order:
    1. Streamlit secrets (works when deployed on Streamlit Cloud)
    2. Environment variables (works with .env locally)
    This covers both local runs and Streamlit Cloud deployment without
    needing separate code paths.
    """
    try:
        import streamlit as st
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass  # not running inside Streamlit, or no secrets configured

    return os.getenv(key, default)


def _extract_answer(text: str) -> str:
    """
    Pulls just the content inside <answer>...</answer> tags. Falls back
    to the raw text if the model didn't produce the tag for some reason,
    so the app never shows a blank explanation.
    """
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


_API_KEY = _get_secret("GEMMA_API_KEY")
_MODEL = _get_secret("GEMMA_MODEL", "gemma-4-26b-a4b-it")
_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"


class GemmaClient:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or _API_KEY
        self.model = model or _MODEL
        if not self.api_key or self.api_key == "your_api_key_here":
            raise RuntimeError(
                "GEMMA_API_KEY is not set. Add it to your .env file locally, "
                "or to Streamlit Cloud's Secrets panel when deployed. "
                "Get a key at https://aistudio.google.com"
            )

    def generate(self, prompt: str, temperature: float = 0.3, max_tokens: int = 400) -> str:
        """
        Sends a single-turn prompt to Gemma and returns the text response.
        Kept intentionally simple (no chat history) since every call in
        this project is a fresh, self-contained request. Extracts only
        the <answer>...</answer> portion so any model "thinking out loud"
        beforehand never reaches the UI.
        """
        url = f"{_BASE_URL}/{self.model}:generateContent?key={self.api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }

        response = requests.post(url, json=payload, timeout=30)

        if response.status_code != 200:
            raise RuntimeError(
                f"Gemma API error {response.status_code}: {response.text}"
            )

        data = response.json()
        try:
            raw = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except (KeyError, IndexError):
            raise RuntimeError(f"Unexpected Gemma API response shape: {data}")

        return _extract_answer(raw)


if __name__ == "__main__":
    # Quick manual test: python gemma_client.py
    client = GemmaClient()
    print(client.generate("Say hello in five words."))
