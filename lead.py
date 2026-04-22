"""
lead.py — Lead capture flow for AutoStream AI Agent.

Manages a multi-step conversational lead capture (name → email → platform)
with validation, retry logic, and fuzzy platform matching.

Exports: lead_state, start_lead_flow, process_lead_input
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Lead state
# ---------------------------------------------------------------------------

lead_state: dict = {
    "active": False,
    "step": None,       # "name" | "email" | "platform"
    "data": {
        "name": None,
        "email": None,
        "platform": None,
    },
    "attempts": {},     # tracks retry count per field
}

# ---------------------------------------------------------------------------
# Constants & Storage
# ---------------------------------------------------------------------------

VALID_PLATFORMS = ["youtube", "twitch", "tiktok", "linkedin", "podcast", "other"]

MAX_RETRIES = 2  # max invalid attempts per field before skipping

captured_leads: list[dict] = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mock_lead_capture(name: str, email: str, platform: str) -> None:
    """Simulate persisting the lead (replace with CRM call in production)."""
    captured_leads.append({"name": name, "email": email, "platform": platform})
    print(f"\n[Lead captured successfully: {name} | {email} | {platform}]\n")


def validate_email(email: str) -> bool:
    """
    Simple email check: must contain '@' and at least one '.' after the '@'.
    No external libraries used.
    """
    if "@" not in email:
        return False
    local, _, domain = email.partition("@")
    return bool(local) and "." in domain and not domain.startswith(".")


def validate_platform(platform: str) -> str | None:
    """
    Fuzzy match user input against VALID_PLATFORMS.
    Returns the matched platform name (lowercase) or None.
    """
    normalized = platform.strip().lower()
    for valid in VALID_PLATFORMS:
        if valid in normalized or normalized in valid:
            return valid
    return None


def _reset_lead() -> None:
    """Reset lead state to default."""
    lead_state["active"] = False
    lead_state["step"] = None
    lead_state["data"] = {"name": None, "email": None, "platform": None}
    lead_state["attempts"] = {}


# ---------------------------------------------------------------------------
# Step handlers
# ---------------------------------------------------------------------------

def _handle_name(user_input: str) -> str:
    name = user_input.strip()

    # Reject: too short or all digits (gibberish check)
    if len(name) < 2 or name.isdigit():
        lead_state["attempts"]["name"] = lead_state["attempts"].get("name", 0) + 1
        return "Hmm, I didn't catch that! What's your name?"

    lead_state["data"]["name"] = name
    lead_state["step"] = "email"
    lead_state["attempts"].pop("name", None)
    return f"Nice to meet you, {name}! What's your email address?"


def _handle_email(user_input: str) -> str:
    email = user_input.strip()
    attempts = lead_state["attempts"].get("email", 0)

    if not validate_email(email):
        if attempts >= MAX_RETRIES:
            # Max retries reached — skip with a note and move on
            lead_state["data"]["email"] = "not provided"
            lead_state["step"] = "platform"
            lead_state["attempts"].pop("email", None)
            return (
                "No worries, we can sort that out later! "
                "Last one — which platform do you mainly create for? "
                "(YouTube, Twitch, TikTok, LinkedIn, Podcast, or Other)"
            )
        lead_state["attempts"]["email"] = attempts + 1
        return "Hmm, that doesn't look right. Try again?"

    lead_state["data"]["email"] = email
    lead_state["step"] = "platform"
    lead_state["attempts"].pop("email", None)
    return (
        "Got it! Last one — which platform do you mainly create for? "
        "(YouTube, Twitch, TikTok, LinkedIn, Podcast, or Other)"
    )


def _handle_platform(user_input: str) -> str:
    matched = validate_platform(user_input)

    if not matched:
        return (
            "We support: YouTube, Twitch, TikTok, LinkedIn, Podcast, or Other "
            "— which fits you best?"
        )

    lead_state["data"]["platform"] = matched
    name = lead_state["data"]["name"] or "there"
    email = lead_state["data"]["email"] or "your email"

    # Fire the capture callback
    mock_lead_capture(name, email, matched)

    # Reset state
    _reset_lead()

    return (
        f"You're all set, {name}! Our team will reach out to {email} "
        f"within 24 hours. Anything else I can help with?"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def start_lead_flow() -> str:
    """
    Initialise lead state and return the opening prompt.
    Call this when high_intent is detected.
    """
    _reset_lead()
    lead_state["active"] = True
    lead_state["step"] = "name"
    return "Awesome! Let's get you set up. First -- what's your name?"


def process_lead_input(user_input: str) -> str:
    """
    Route the user's input to the correct step handler.
    Call this on every message while lead_state['active'] is True.
    """
    step = lead_state.get("step")

    if step == "name":
        return _handle_name(user_input)
    elif step == "email":
        return _handle_email(user_input)
    elif step == "platform":
        return _handle_platform(user_input)
    else:
        # Shouldn't happen; reset gracefully
        _reset_lead()
        return "Something went wrong with the sign-up flow. Let's start over — type 'sign me up' anytime!"

def load_all_leads() -> list[dict]:
    """Return all captured leads for the current session."""
    return captured_leads
