"""
agent.py — Core AI agent orchestrator for AutoStream.
"""
from __future__ import annotations

import os
import time
import json
import google.generativeai as genai

from intent import classify_intent, CLASSIFIER_SYSTEM_PROMPT
from rag import retrieve_context, get_agent_response
from lead import lead_state, start_lead_flow, process_lead_input

client = None
intent_client = None

def init_clients():
    global client, intent_client
    if client is not None:
        return
    
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY is not set. Please add it to your .env file.")
    genai.configure(api_key=api_key)
    
    model_name = "gemini-2.5-flash-lite"
    
    client = genai.GenerativeModel(model_name=model_name)
    intent_client = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=CLASSIFIER_SYSTEM_PROMPT,
    )

def load_kb() -> dict:
    from pathlib import Path
    kb_path = Path(__file__).parent / "knowledge_base.json"
    with kb_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)

chat_history: list[dict[str, str]] = []
was_streamed = False  # Track if the last response was streamed for CLI formatting

def call_with_retry(fn, max_retries=3):
    for i in range(max_retries):
        try:
            return fn()
        except Exception as e:
            if "429" in str(e) and i < max_retries - 1:
                wait = 30 * (i + 1)
                print(f"\nRate limited. Waiting {wait}s before retry {i + 2}/{max_retries}...")
                time.sleep(wait)
            else:
                raise

def chat(user_input: str, kb: dict) -> str:
    global chat_history, was_streamed
    init_clients()
    
    # Always append user input
    chat_history.append({"role": "user", "content": user_input})
    
    # If mid lead flow, skip intent classification
    if lead_state["active"]:
        response = process_lead_input(user_input)
        chat_history.append({"role": "assistant", "content": response})
        was_streamed = False
        return response
    
    # Classify intent
    intent_result = classify_intent(user_input, chat_history[-6:], intent_client)
    intent = intent_result["intent"]
    
    if intent == "greeting":
        response = "Hey! 👋 I'm Maya, AutoStream's AI assistant. I can help with pricing, features, or get you signed up. What's up?"
        was_streamed = False
        
    elif intent in ("inquiry_pricing", "inquiry_features", "inquiry_policies", "general"):
        print("Maya: ", end="", flush=True)
        context = retrieve_context(user_input, kb)
        response = call_with_retry(lambda: get_agent_response(user_input, context, chat_history[-10:], client))
        was_streamed = True
        
    elif intent == "high_intent":
        response = start_lead_flow()
        was_streamed = False
        
    elif intent == "off_topic":
        response = "Ha, I'm pretty much only good at AutoStream stuff 😅 Got any questions about the product?"
        was_streamed = False
        
    else:  # fallback
        print("Maya: ", end="", flush=True)
        context = retrieve_context(user_input, kb)
        response = call_with_retry(lambda: get_agent_response(user_input, context, chat_history[-10:], client))
        was_streamed = True
    
    chat_history.append({"role": "assistant", "content": response})
    
    # Keep history from growing unbounded
    if len(chat_history) > 20:
        chat_history = chat_history[-20:]
    
    return response
