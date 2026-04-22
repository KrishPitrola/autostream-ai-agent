"""
rag.py — Retrieval-Augmented Generation for AutoStream AI Agent.

Provides:
  - retrieve_context()    : Keyword-scored retrieval of top KB sections
  - format_section()      : Recursive human-readable section formatter
  - get_agent_response()  : Streaming Gemini response as Maya, AutoStream's AI assistant
"""

from __future__ import annotations

import re
import warnings
from typing import Any

# Suppress the google.generativeai deprecation FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="google.generativeai")


# ---------------------------------------------------------------------------
# Section formatter
# ---------------------------------------------------------------------------

def format_section(key: str, value: Any, depth: int = 0) -> str:
    """
    Recursively format a KB section into a readable text block.

    Example output:
      "PRICING - PRO PLAN: $79/month. Features: Unlimited videos, 4K resolution..."
    """
    label = key.upper().replace("_", " ")

    if isinstance(value, dict):
        parts = []
        for sub_key, sub_val in value.items():
            parts.append(format_section(sub_key, sub_val, depth + 1))
        joined = ". ".join(p for p in parts if p)
        if depth == 0:
            return f"{label}: {joined}"
        return f"{label} — {joined}"

    if isinstance(value, list):
        items = ", ".join(str(v) for v in value)
        if depth == 0:
            return f"{label}: {items}"
        return f"{label}: {items}"

    # Scalar
    text = str(value)
    if depth == 0:
        return f"{label}: {text}"
    return f"{label}: {text}"


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

def retrieve_context(user_input: str, knowledge_base: dict) -> str:
    """
    Score each top-level KB section by expanded keyword overlap.
    Alias map bridges natural-language words to KB section names so queries
    like 'pro plan price' reliably hit the 'pricing' section.
    Returns the top 2 scoring sections as a formatted string.
    """
    user_tokens = set(user_input.lower().split())

    # Synonym / alias map — expand user tokens into KB-relevant terms
    aliases: dict[str, list[str]] = {
        "price":       ["pricing", "cost", "plan", "plans", "cheap", "expensive", "fee", "fees", "monthly", "annual"],
        "pro":         ["pricing", "plans", "tier"],
        "basic":       ["pricing", "plans", "tier"],
        "enterprise":  ["pricing", "plans", "tier"],
        "feature":     ["features", "what", "does", "capabilities", "include"],
        "features":    ["features", "capabilities"],
        "refund":      ["policies", "policy", "money", "back", "cancel"],
        "support":     ["policies", "help", "contact"],
        "trial":       ["policies", "free", "try"],
        "cancel":      ["policies"],
        "integration": ["integrations", "connect", "youtube", "twitch"],
        "caption":     ["features", "ai", "subtitle"],
        "storage":     ["faqs", "space", "gb"],
        "team":        ["faqs", "seats", "members"],
        "upgrade":     ["faqs", "downgrade", "change"],
        "plan":        ["pricing", "plans", "tier"],
        "plans":       ["pricing"],
        "cost":        ["pricing"],
        "billing":     ["pricing"],
        "connect":     ["integrations"],
        "publish":     ["features", "integrations"],
        "youtube":     ["integrations", "features"],
        "twitch":      ["integrations", "features"],
        "tiktok":      ["integrations"],
        "policy":      ["policies"],
        "faq":         ["faqs"],
        "question":    ["faqs"],
    }

    # Expand user tokens with aliases
    expanded = set(user_tokens)
    for token in user_tokens:
        if token in aliases:
            expanded.update(aliases[token])

    def flatten_to_str(obj) -> str:
        if isinstance(obj, dict):
            return " ".join(flatten_to_str(v) for v in obj.values())
        elif isinstance(obj, list):
            return " ".join(str(i) for i in obj)
        return str(obj).lower()

    scores: dict[str, int] = {}
    for key, value in knowledge_base.items():
        section_str = (key + " " + flatten_to_str(value)).lower()
        section_tokens = set(section_str.split())
        score = len(expanded & section_tokens)
        # Boost if the top-level key is directly referenced in the query
        if key in expanded or key in user_input.lower():
            score += 3
        scores[key] = score

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top = [k for k, s in ranked[:2] if s > 0]

    if not top:
        return (
            f"Product: {flatten_to_str(knowledge_base.get('product', {}))}"
            f"\nAvailable topics: {', '.join(knowledge_base.keys())}"
        )

    return "\n\n".join(format_section(k, knowledge_base[k]) for k in top)


# ---------------------------------------------------------------------------
# Streaming agent response
# ---------------------------------------------------------------------------

_MAYA_SYSTEM = """\
You are Maya, AutoStream's friendly AI assistant. You're helpful, concise, \
and a little enthusiastic about helping creators grow.

Answer using ONLY this context:
{context}

Rules:
- If info not in context, say "I don't have that detail — want me to connect you with our team?"
- Never invent prices, features, or policies
- Keep answers under 100 words unless user asks for detail
- Use natural conversational tone, not bullet dumps
"""


def get_agent_response(
    user_input: str,
    context: str,
    chat_history: list[dict[str, str]],
    client,
) -> str:
    """
    Generate a streaming response as Maya using the provided Gemini client.

    Args:
        user_input:   Latest user message.
        context:      Formatted KB context string from retrieve_context().
        chat_history: List of {"role": str, "content": str} dicts (full history).
        client:       A genai.GenerativeModel instance.

    Prints tokens as they stream. Returns the complete response text.
    """
    system = _MAYA_SYSTEM.format(context=context)

    # Build conversation block from last 5 turns (10 entries)
    recent = chat_history[-(10):]
    convo_lines = []
    for turn in recent:
        role_label = "User" if turn["role"] == "user" else "Maya"
        convo_lines.append(f"{role_label}: {turn['content']}")
    convo_lines.append(f"User: {user_input}")
    convo_lines.append("Maya:")

    full_prompt = system + "\n\n" + "\n".join(convo_lines)

    # Stream response
    response = client.generate_content(full_prompt, stream=True)
    full_text = ""
    for chunk in response:
        token = getattr(chunk, "text", "")
        if token:
            print(token, end="", flush=True)
            full_text += token
    print()  # newline to end the streamed line

    return full_text


# ---------------------------------------------------------------------------
# Self-test — run directly: python rag.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    from pathlib import Path

    kb_path = Path(__file__).parent / "knowledge_base.json"
    with kb_path.open("r", encoding="utf-8") as fh:
        kb = json.load(fh)

    test_cases = [
        ("What is the Pro plan price?",      "pricing"),
        ("do you have a refund policy?",      "policies"),
        ("what integrations do you support",  "integrations"),
    ]

    print("=" * 60)
    print("retrieve_context() — Self-Test")
    print("=" * 60)

    all_passed = True
    for query, expected_section in test_cases:
        context = retrieve_context(query, kb)
        hit = expected_section.upper() in context.upper()
        status = "PASS" if hit else "FAIL"
        if not hit:
            all_passed = False
        print(f"\nQuery:    {query!r}")
        print(f"Expected: {expected_section!r} in context -> {status}")
        print(f"Context preview: {context[:120].strip()}...")

    print("\n" + "=" * 60)
    print("All tests passed!" if all_passed else "Some tests FAILED -- check alias map")
    print("=" * 60)
