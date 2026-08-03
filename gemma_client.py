"""
gemma_client.py
Thin wrapper around Google's Generative Language API for calling Gemma models.
"""

import os
import re
import requests
from dotenv import load_dotenv

load_dotenv()


def _get_secret(key: str, default=None):
    try:
        import streamlit as st
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key, default)


def _extract_answer(text: str, fallback: str = "") -> str:
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match and match.group(1).strip():
        return match.group(1).strip()
    return fallback if fallback else text.strip()[:200]


_API_KEY = _get_secret("GEMMA_API_KEY")
_MODEL = _get_secret("GEMMA_MODEL", "gemma-4-26b-a4b-it")
_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"


class GemmaClient:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or _API_KEY
        self.model = model or _MODEL
        if not self.api_key or self.api_key == "your_api_key_here":
            raise RuntimeError(
                "GEMMA_API_KEY is not set. Add it to Streamlit Cloud's Secrets panel."
            )

    def generate(self, prompt: str, temperature: float = 0.1, max_tokens: int = 1024, fallback: str = "") -> str:
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
            raise RuntimeError(f"Gemma API error {response.status_code}: {response.text}")
        data = response.json()
        try:
            raw = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except (KeyError, IndexError):
            raise RuntimeError(f"Unexpected Gemma API response shape: {data}")
        return _extract_answer(raw, fallback=fallback)


if __name__ == "__main__":
    client = GemmaClient()
    print(client.generate("Say hello in five words."))
