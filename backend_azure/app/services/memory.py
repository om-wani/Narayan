"""Conversation memory helpers."""

from __future__ import annotations

from typing import TypedDict


class ChatMessage(TypedDict):
    role: str
    content: str


_history_store: dict[str, list[ChatMessage]] = {}


def save_message(session_id: str, role: str, content: str) -> None:
    _history_store.setdefault(session_id, []).append({"role": role, "content": content})


def get_history(session_id: str) -> list[ChatMessage]:
    return [dict(message) for message in _history_store.get(session_id, [])]


def summarize_history(history: list[ChatMessage], keep_last: int = 4) -> list[ChatMessage]:
    if len(history) <= keep_last:
        return [dict(message) for message in history]

    older = history[:-keep_last]
    recent = history[-keep_last:]
    summary_bits: list[str] = []
    for message in older:
        content = message.get("content", "").strip()
        if content:
            summary_bits.append(f"{message.get('role', 'unknown')}: {content}")

    summary_text = "; ".join(summary_bits)
    summary_message: ChatMessage = {
        "role": "system",
        "content": f"Conversation summary: {summary_text}",
    }
    return [summary_message] + [dict(message) for message in recent]


def compact_session_history(session_id: str, keep_last: int = 4) -> list[ChatMessage]:
    return summarize_history(get_history(session_id), keep_last=keep_last)


def clear_session(session_id: str) -> None:
    _history_store.pop(session_id, None)
