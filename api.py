# api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

# Import the agents you already wrote
from agents import (
    sentiment_agent,
    triage_agent,
    policy_agent,
    ticket_agent,
    email_agent,
)

app = FastAPI(title="Customer‑Support AI Pipeline")


class SupportRequest(BaseModel):
    query: str                     # customer’s message
    customer_id: str = "guest_user"   # optional identifier


@app.post("/process")
def run_pipeline(req: SupportRequest) -> Dict[str, Any]:
    """Execute the full agent chain and return the assembled response."""
    # 1️⃣ Sent```python
# api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

# Import the agents you already wrote
from agents import (
    sentiment_agent,
    triage_agent,
    policy_agent,
    ticket_agent,
    email_agent,
)

app = FastAPI(title="Customer‑Support AI Pipeline")


class SupportRequest(BaseModel):
    query: str                     # customer’s message
    customer_id: str = "guest_user"   # optional identifier


@app.post("/process")
def run_pipeline(req: SupportRequest) -> Dict[str, Any]:
    """Execute the full agent chain and return the assembled response."""
    # 1️⃣ Sentiment analysis
    state: Dict[str, Any] = {"query": req.query, "customer_id": req.customer_id}
    state.update(sentiment_agent(state))

    # 2️⃣ Intent triage & context retrieval
    state.update(triage_agent(state))

    # 3️⃣ Policy lookup
    state.update(policy_agent(state))

    # 4️⃣ Ticket creation
    state.update(ticket_agent(state))

    # 5️⃣ Email draft
    state.update(email_agent(state))

    # Return the pieces the UI will need
    return {
        "sentiment": state.get("sentiment"),
        "urgency": state.get("urgency"),
        "intent": state.get("current_intent"),
        "confidence": state.get("confidence_score"),
        "policy": state.get("policy_guideline"),
        "ticket": state.get("ticket"),
        "draft_email": state.get("draft_response"),
    }