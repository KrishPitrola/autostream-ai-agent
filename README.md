# AutoStream AI Agent

> Conversational AI sales assistant for AutoStream — a fictional video streaming SaaS. Handles product queries, detects user intent, and captures leads via a CLI chat interface.

---

## How to Run Locally

### Prerequisites
- Python 3.9+
- GEMINI API key → get free at [aistudio.google.com](https://aistudio.google.com/)

### Setup

```bash
# 1. Clone and enter project
git clone <your-repo-url>
cd autostream

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Open .env and add your key:
# GEMINI_API_KEY=sk-ant-xxxxxxxxxx

# 5. Run
python main.py
```

### Expected Output
```
╔══════════════════════════════╗
║   AutoStream AI Assistant    ║
║   Type 'exit' to quit        ║
╚══════════════════════════════╝

You: hi
Maya: Hey! 👋 I'm Maya, AutoStream's AI assistant...
```

Type `exit` or `quit` to stop.

---

## Architecture

### Why No LangChain or LangGraph?

LangGraph and AutoGen excel at multi-agent systems with complex DAG workflows, parallel execution, and dozens of chained tool calls. This project is a single conversational agent with a linear flow — adding LangGraph would mean framework overhead, hidden abstractions, and harder debugging for zero benefit. The deliberate choice was raw GEMINI SDK: every LLM call is visible, every failure is traceable, and the codebase stays under 500 lines total.

### How It Works

**Intent Classification** — Every user message is classified by Claude Haiku into one of 8 intents (`greeting`, `inquiry_pricing`, `inquiry_features`, `inquiry_policies`, `high_intent`, `off_topic`, `lead_followup`, `general`). LLM classification handles paraphrasing that keyword rules miss — "what'll it run me monthly" correctly maps to `inquiry_pricing`.

**RAG (Retrieval-Augmented Generation)** — `knowledge_base.json` is not dumped wholesale into the prompt. A scoring function with synonym expansion maps user tokens to relevant KB sections and injects only the top 2 matches. This keeps prompts small and prevents hallucination from noisy context.

**State Management** — Two plain Python dicts handle all state. `chat_history` (capped at 20 turns, last 5 passed to RAG, last 6 to intent classifier) provides conversational memory. `lead_state` tracks the lead capture flow with per-field validation — email format checked, platform fuzzy-matched, `mock_lead_capture()` only fires when all 3 fields are confirmed.

**Module Breakdown**
```
intent.py  → LLM-based intent classification
rag.py     → context retrieval + streaming LLM response
lead.py    → lead state machine + validation
agent.py   → orchestrator, routing, history management
main.py    → CLI entrypoint + retry logic
```

---

## WhatsApp Deployment via Webhooks

### Overview

To deploy this agent on WhatsApp, use the **Twilio WhatsApp API** with a **Flask webhook**. Twilio receives WhatsApp messages, forwards them to your server via HTTP POST, your server runs the agent, and returns the response.

### Architecture

```
User (WhatsApp)
      ↓
Twilio WhatsApp API
      ↓ HTTP POST
Flask Webhook Server  →  agent.chat(message)  →  Maya response
      ↓
Twilio sends reply back to user
```

### Implementation

```python
# whatsapp_webhook.py
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from agent import AutoStreamAgent

app = Flask(__name__)

# One agent instance per user session (keyed by phone number)
sessions = {}

@app.route("/webhook", methods=["POST"])
def webhook():
    sender = request.form.get("From", "")       # e.g. whatsapp:+919876543210
    user_msg = request.form.get("Body", "").strip()

    # Get or create agent session for this user
    if sender not in sessions:
        sessions[sender] = AutoStreamAgent()

    response_text = sessions[sender].chat(user_msg)

    resp = MessagingResponse()
    resp.message(response_text)
    return str(resp)

if __name__ == "__main__":
    app.run(port=5000)
```

### Setup Steps

1. **Twilio account** → enable WhatsApp Sandbox at [twilio.com/console](https://twilio.com/console)
2. **Expose local server** → use `ngrok http 5000` to get public URL
3. **Set webhook URL** in Twilio console → `https://your-ngrok-url/webhook`
4. **Add dependency** → `pip install twilio flask`
5. **Run** → `python whatsapp_webhook.py`

### Production Considerations

- **Per-user state** — `sessions` dict is in-memory. For production, store `lead_state` and `chat_history` in Redis keyed by sender phone number. Survives server restarts and scales across workers.
- **Async** — Replace Flask with FastAPI + async handlers for concurrent users.
- **Message length** — WhatsApp caps messages at 1600 characters. Add response truncation if needed.

---

## Project Structure

```
autostream/
├── main.py               # CLI entrypoint, REPL loop
├── agent.py              # Orchestrator, routing, retry logic
├── intent.py             # LLM intent classifier
├── rag.py                # Context retrieval + LLM response
├── lead.py               # Lead state machine + validation
├── knowledge_base.json   # Product KB (pricing, features, policies, FAQs)
├── requirements.txt
├── .env.example
└── README.md
```

## Requirements

```
google-generativeai
python-dotenv
numpy
```

---

## Demo Flow

```
You: hi
Maya: Hey! 👋 I'm Maya...

You: what are the pricing plans?
Maya: AutoStream has three plans...  [pulls from KB]

You: i want to sign up for pro
Maya: Great choice! Let's get you set up 🚀 First — what's your name?

You: John
Maya: Nice to meet you, John! What's your email address?

You: john@example.com
Maya: Got it! Which platform do you mainly create for?

You: YouTube
✅ Lead captured successfully: John | john@example.com | youtube
Maya: You're all set, John! 🎉 Our team will reach out within 24 hours.
```