"""Prompt safety checks."""

from __future__ import annotations

import re


INJECTION_PATTERNS = [
    r"ignore (?:all|previous) instructions",
    r"reveal (?:the )?system prompt",
    r"show me your hidden prompt",
    r"developer message",
    r"tool call",
    r"bypass safety",
    r"act as system",
]


def is_safe_text(text: str) -> bool:
    lowered = text.lower()
    return not any(re.search(pattern, lowered) for pattern in INJECTION_PATTERNS)


def assert_safe_text(text: str) -> None:
    if not is_safe_text(text):
        raise ValueError("Potential prompt injection detected.")


def sanitize_output(text: str) -> str:
    return text.strip()
