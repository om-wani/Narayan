"""Azure OpenAI client helpers."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Iterable

from app.core.config import settings


def _require_config() -> None:
    required = [
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_CHAT_DEPLOYMENT",
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
    ]
    missing = [name for name in required if not getattr(settings, name)]
    if missing:
        raise RuntimeError(f"Missing Azure OpenAI config: {', '.join(missing)}")


@lru_cache(maxsize=1)
def get_client() -> Any:
    _require_config()
    from openai import AzureOpenAI

    return AzureOpenAI(
        api_key=settings.AZURE_OPENAI_API_KEY,
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        api_version=settings.AZURE_OPENAI_API_VERSION,
    )


def chat_completion(messages: list[dict[str, str]], *, temperature: float = 0.1, max_tokens: int | None = None) -> Any:
    client = get_client()
    return client.chat.completions.create(
        model=settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens or settings.ANSWER_MAX_TOKENS,
    )


def stream_chat_completion(
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.1,
    max_tokens: int | None = None,
) -> Iterable[dict[str, Any]]:
    client = get_client()
    stream = client.chat.completions.create(
        model=settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens or settings.ANSWER_MAX_TOKENS,
        stream=True,
        stream_options={"include_usage": True},
    )

    usage: dict[str, int] | None = None
    for event in stream:
        if getattr(event, "usage", None):
            usage_obj = event.usage
            usage = {
                "prompt_tokens": int(getattr(usage_obj, "prompt_tokens", 0) or 0),
                "completion_tokens": int(getattr(usage_obj, "completion_tokens", 0) or 0),
                "total_tokens": int(getattr(usage_obj, "total_tokens", 0) or 0),
            }

        choices = getattr(event, "choices", None) or []
        if not choices:
            continue

        delta = getattr(choices[0], "delta", None)
        content = getattr(delta, "content", None)
        if content:
            yield {"type": "delta", "text": content}

    yield {"type": "done", "usage": usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}


def embed_texts(texts: list[str]) -> list[list[float]]:
    client = get_client()
    response = client.embeddings.create(
        model=settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        input=texts,
    )
    return [list(item.embedding) for item in response.data]


def embed_text(text: str) -> list[float]:
    return embed_texts([text])[0]
