"""The AI layer: turns raw data into plain English.

Two jobs:
  1. ``summarise_event`` — expand a dev event into the 4-section breakdown
     (What they built / Why it matters / In simple terms / The Alpha take).
  2. ``answer_question`` — the TAO Oracle, grounded on the data we collected.

Uses OpenRouter or Anthropic when a key is configured; otherwise falls back to
deterministic templates so the features always render.
"""
from __future__ import annotations

import json

import httpx

from .config import Settings
from .providers.base import DevEvent

_SYSTEM_FEED = (
    "You are AlphaGap's analyst. Given a Bittensor subnet development event, return "
    "STRICT JSON with keys: what_built, why_matters, simple_terms, alpha_take. Each "
    "value is 1-2 sentences, plain English, no jargon in simple_terms."
)

_SYSTEM_ORACLE = (
    "You are the TAO Oracle, AlphaGap's analyst for the Bittensor ecosystem. Answer "
    "ONLY using the provided context data about subnets (scores, signals, whales, dev "
    "activity). Be concise, specific, and cite subnet numbers (e.g. SN19). If the "
    "context does not contain the answer, say so."
)


def _llm_chat(settings: Settings, system: str, user: str, max_tokens: int = 600) -> str | None:
    if settings.openrouter_api_key:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {"Authorization": f"Bearer {settings.openrouter_api_key}"}
        model = settings.oracle_model
    elif settings.anthropic_api_key:
        # Use Anthropic via its OpenAI-compatible-ish messages endpoint.
        return _anthropic_chat(settings, system, user, max_tokens)
    else:
        return None
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.4,
    }
    try:
        with httpx.Client(timeout=40.0) as c:
            r = c.post(url, headers=headers, json=payload)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
    except Exception:
        return None


def _anthropic_chat(settings: Settings, system: str, user: str, max_tokens: int) -> str | None:
    try:
        with httpx.Client(timeout=40.0) as c:
            r = c.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": settings.anthropic_api_key or "",
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-3-5-sonnet-20241022",
                    "max_tokens": max_tokens,
                    "system": system,
                    "messages": [{"role": "user", "content": user}],
                },
            )
            r.raise_for_status()
            return r.json()["content"][0]["text"]
    except Exception:
        return None


def summarise_event(settings: Settings, event: DevEvent, subnet_name: str) -> dict:
    if settings.has_llm:
        user = (
            f"Subnet: {subnet_name} (SN{event.netuid})\n"
            f"Event: {event.title}\nDetails: {event.detail}\nRaw: {event.raw_text}"
        )
        raw = _llm_chat(settings, _SYSTEM_FEED, user)
        if raw:
            try:
                start, end = raw.find("{"), raw.rfind("}")
                parsed = json.loads(raw[start:end + 1])
                return {
                    "what_built": parsed.get("what_built", ""),
                    "why_matters": parsed.get("why_matters", ""),
                    "simple_terms": parsed.get("simple_terms", ""),
                    "alpha_take": parsed.get("alpha_take", ""),
                }
            except Exception:
                pass
    # Deterministic fallback.
    return {
        "what_built": event.title + ("." if not event.title.endswith(".") else ""),
        "why_matters": event.detail or "Signals active development on this subnet.",
        "simple_terms": f"The {subnet_name} team just shipped something new.",
        "alpha_take": "Dev activity often precedes market re-rating — watch for a gap.",
    }


def answer_question(settings: Settings, question: str, context: str, sources: list[str]) -> dict:
    if settings.has_llm:
        user = f"Context data:\n{context}\n\nQuestion: {question}"
        raw = _llm_chat(settings, _SYSTEM_ORACLE, user, max_tokens=500)
        if raw:
            return {"answer": raw.strip(), "sources": sources, "grounded": True}
    # Fallback: synthesise a deterministic answer from the context summary lines.
    answer = (
        "Based on the latest scan:\n" + context
        if context else
        "I don't have enough scanned data to answer that yet. Try a broader question "
        "like 'top alpha gaps right now' or 'which subnets are whales accumulating'."
    )
    return {"answer": answer, "sources": sources, "grounded": bool(context)}
