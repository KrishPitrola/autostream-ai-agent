"""
intent.py — LLM-based intent classifier for AutoStream AI Agent.

Uses Gemini gemini-2.5-flash-lite to classify user intent into one of
the defined categories, returning a structured confidence + reasoning result.
"""

from __future__ import annotations

import json
import os

import google.generativeai as genai


# ---------------------------------------------------------------------------
# Intent categories
# ---------------------------------------------------------------------------

INTENTS = [
    "greeting",
    "inquiry_pricing",
    "inquiry_features",
    "inquiry_policies",
    "high_intent",
    "off_topic",
    "lead_followup",
    "general",
]

# ---------------------------------------------------------------------------
# Classifier system prompt
# ---------------------------------------------------------------------------

CLASSIFIER_SYSTEM_PROMPT = """
You are an intent classifier for AutoStream, a video streaming SaaS.
Classify user message into exactly one intent.

Intents:
- greeting: hello, hi, hey, how are you
- inquiry_pricing: asking about price, cost, plans, tiers, billing, annual
- inquiry_features: asking what AutoStream does, capabilities, integrations
- inquiry_policies: asking about refund, trial, cancel, support, seats
- high_intent: wants to sign up, buy, start trial, get access, purchase, ready to go
- off_topic: completely unrelated to AutoStream
- lead_followup: looks like they're answering a name/email/platform question
- general: anything else about AutoStream

Last 3 conversation turns provided for context.

Respond ONLY in JSON:
{"intent": "...", "confidence": "high|medium|low", "reasoning": "one sentence"}
"""

_PARSE_ERROR_RESULT: dict = {
    "intent": "general",
    "confidence": "low",
    "reasoning": "parse error",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_history(chat_history: list, max_turns: int = 3) -> str:
    """Format the last N turns of chat history as a readable string."""
    # Each entry is {"role": str, "content": str}
    recent = chat_history[-(max_turns * 2):]
    lines = [
        f"{turn.get('role', 'unknown').capitalize()}: {turn.get('content', '')}"
        for turn in recent
    ]
    return "\n".join(lines) if lines else "(no prior conversation)"


def _strip_code_fence(text: str) -> str:
    """Remove markdown ```json ... ``` fences if present."""
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        # parts[1] is the inner block (possibly prefixed with 'json\n')
        inner = parts[1] if len(parts) > 1 else text
        if inner.startswith("json"):
            inner = inner[4:]
        return inner.strip()
    return text


# ---------------------------------------------------------------------------
# Public classifier
# ---------------------------------------------------------------------------

def classify_intent(user_input: str, chat_history: list, client) -> dict:
    """
    Classify the intent of a user message using Gemini.

    Args:
        user_input:   The latest user message.
        chat_history: List of {"role": str, "content": str} dicts (full history).
        client:       A genai.GenerativeModel instance configured for classification.

    Returns:
        {"intent": str, "confidence": "high|medium|low", "reasoning": str}
    """
    history_text = _format_history(chat_history)
    prompt = (
        f"Recent conversation:\n{history_text}\n\n"
        f"New user message: {user_input}\n\n"
        "Classify the intent of the new user message."
    )

    try:
        response = client.generate_content(prompt)
        raw = _strip_code_fence(response.text)
        result = json.loads(raw)

        # Ensure required key exists and value is valid
        if "intent" not in result:
            return _PARSE_ERROR_RESULT
        if result["intent"] not in INTENTS:
            result["intent"] = "general"

        # Fill missing optional keys with safe defaults
        result.setdefault("confidence", "low")
        result.setdefault("reasoning", "")
        return result

    except Exception:  # noqa: BLE001
        return _PARSE_ERROR_RESULT


# ---------------------------------------------------------------------------
# Self-test — run directly: python intent.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("[ERROR] GEMINI_API_KEY not set. Add it to your .env file.")
        raise SystemExit(1)

    genai.configure(api_key=api_key)
    intent_model = genai.GenerativeModel(
        model_name="gemini-2.5-flash-lite",
        system_instruction=CLASSIFIER_SYSTEM_PROMPT,
    )

    # 5 sample inputs: (user_message, chat_history)
    samples = [
        (
            "Hey! What's up?",
            [],
        ),
        (
            "How much does the Pro plan cost per year?",
            [],
        ),
        (
            "Can AutoStream publish to TikTok and YouTube at the same time?",
            [{"role": "user", "content": "Tell me about AutoStream"}],
        ),
        (
            "john@example.com",
            [
                {"role": "user", "content": "I want to sign up"},
                {"role": "assistant", "content": "Great! What is your email address?"},
            ],
        ),
        (
            "What's the capital of France?",
            [],
        ),
    ]

    print("=" * 60)
    print("AutoStream Intent Classifier — Self-Test")
    print("=" * 60)

    for msg, history in samples:
        result = classify_intent(msg, history, intent_model)
        print(f"\n  Input:      {msg!r}")
        print(f"  Intent:     {result['intent']}")
        print(f"  Confidence: {result['confidence']}")
        print(f"  Reasoning:  {result['reasoning']}")

    print("\n" + "=" * 60)
