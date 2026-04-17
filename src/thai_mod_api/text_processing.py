from __future__ import annotations

import re

import emoji
import pandas as pd


def preprocess_text(text: str) -> str:
    if pd.isna(text):
        return ""

    cleaned = str(text)
    cleaned = re.sub(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
        "",
        cleaned,
    )
    cleaned = re.sub(
        r"www\\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
        "",
        cleaned,
    )
    cleaned = emoji.demojize(cleaned, language="en").strip()

    lowered = []
    for char in cleaned:
        lowered.append(char.lower() if ord(char) < 128 else char)
    return "".join(lowered)
