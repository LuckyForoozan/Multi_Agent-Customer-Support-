import os
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Header, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from observability import metrics

# Import workflow and state manager
from workflow import process_customer_query
from state_manager import get_conversation, get_ticket

app = FastAPI(
    title="Multi-Agent Customer Support Gateway",
    description="Autonomous Agentic Support System with Human-in-the-Loop Escalation",
    version="1.0.0"
)

# Optional API Key for security (can be set in .env)
API_SECRET_KEY = os.getenv("API_SECRET_KEY", "dev_secret_key_123")


# ============================================================
# 1. 📋 Request & Response Schemas
# ============================================================
class QueryRequest(BaseModel):
    query: str
    customer_id: str = "guest_user"


class QueryResponse(BaseModel):
    session_id: str
    intent: Optional[str]
    confidence_score: float
    sentiment: str
    urgency: str
    escalated_to_human: bool
    escalation_reason: Optional[str]
    ticket_id: Optional[str]
    response: Optional[str]


class FeedbackRequest(BaseModel):
    session_id: str
    rating: int = Field(..., ge=1, le=5, description="Rating from 1 to 5")
    comment: Optional[str] = None


class WebhookPayload(BaseModel):
    source: str  # "slack", "zendesk", "email"
    sender: str
    message: str


# ============================================================
# 2. 🔀 Endpoints
# ============================================================
@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "service": "Multi-Agent Customer Support Gateway",
        "docs_url": "/docs"
    }


@app.post("/query", response_model=QueryResponse)
def handle_query(payload: QueryRequest):
    """
    Main endpoint: runs customer inquiry through the LangGraph
    multi-agent workflow and returns the agent decision.
    """
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    # Run through LangGraph workflow
    result = process_customer_query(query=payload.query, customer_id=payload.customer_id)
    ticket_info = result.get("ticket", {})

    return QueryResponse(
        session_id=result.get("session_id", ""),
        intent=result.get("current_intent"),
        confidence_score=result.get("confidence_score", 0.0),
        sentiment=result.get("sentiment", "neutral"),
        urgency=result.get("urgency", "normal"),
        escalated_to_human=result.get("escalated_to_human", False),
        escalation_reason=result.get("escalation_reason"),
        ticket_id=ticket_info.get("ticket_id") if isinstance(ticket_info, dict) else None,
        response=result.get("draft_response")
    )


@app.get("/ticket/{ticket_id}")
def fetch_ticket(ticket_id: str):
    """Fetch ticket details from durable storage."""
    ticket = get_ticket(ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail=f"Ticket '{ticket_id}' not found.")
    return ticket


@app.get("/conversation/{session_id}")
def fetch_conversation(session_id: str):
    """Fetch conversation history from Redis/FakeRedis."""
    conv = get_conversation(session_id)
    if not conv:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return conv


@app.post("/feedback")
def submit_feedback(payload: FeedbackRequest):
    """Log customer or operator feedback for RAG/fine-tuning signals."""
    print(f"⭐ [Feedback Received] Session: {payload.session_id} | Rating: {payload.rating}/5 | Comment: {payload.comment}")
    return {
        "status": "success",
        "message": "Feedback recorded successfully for continuous improvement."
    }


@app.post("/webhook/inbound")
def inbound_webhook(payload: WebhookPayload):
    """Webhook listener for inbound customer messages from Zendesk, Slack, or Email."""
    print(f"📥 [Inbound Webhook] Received from {payload.source} by {payload.sender}")
    # Process inquiry through agents
    result = process_customer_query(query=payload.message, customer_id=payload.sender)
    return {
        "status": "processed",
        "source": payload.source,
        "escalated_to_human": result.get("escalated_to_human"),
        "draft_response": result.get("draft_response")
    }

@app.get("/metrics")
def get_metrics():
    return metrics.get_summary()
# ============================================================
# 👤 Human Operator Dashboard UI
# ============================================================
@app.get("/dashboard", response_class=HTMLResponse)
def operator_dashboard():
    """Web UI for human operators to inspect, approve, or override escalated tickets."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Operator Escalation Dashboard</title>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-gray-100 text-gray-800 p-8 font-sans">
        <div class="max-w-4xl mx-auto bg-white rounded-xl shadow-md p-8 border border-gray-200">
            <div class="flex items-center justify-between border-b pb-4 mb-6">
                <div>
                    <h1 class="text-2xl font-bold text-indigo-700">👤 Human-in-the-Loop Operator Dashboard</h1>
                    <p class="text-sm text-gray-500">Review low-confidence queries, approve drafts, or provide custom overrides.</p>
                </div>
                <span class="bg-amber-100 text-amber-800 text-xs font-semibold px-3 py-1 rounded-full border border-amber-300">Live Queue</span>
            </div>

            <!-- Lookup Section -->
            <div class="mb-6 bg-slate-50 p-4 rounded-lg border">
                <label class="block text-sm font-semibold mb-2 text-gray-700">Search Session ID to Review:</label>
                <div class="flex gap-2">
                    <input id="sessionIdInput" type="text" placeholder="e.g. sess_39172e07" class="flex-1 p-2 border rounded-md">
                    <button onclick="loadSession()" class="bg-indigo-600 hover:bg-indigo-700 text-white px-4 py-2 rounded-md font-semibold text-sm">Fetch Inquiry</button>
                </div>
            </div>

            <!-- Details Display -->
            <div id="detailsCard" class="hidden space-y-4">
                <div class="grid grid-cols-2 gap-4">
                    <div class="bg-gray-50 p-3 rounded border">
                        <span class="text-xs text-gray-500 uppercase font-bold">Detected Intent</span>
                        <p id="dispIntent" class="font-semibold text-gray-800 mt-1">-</p>
                    </div>
                    <div class="bg-gray-50 p-3 rounded border">
                        <span class="text-xs text-gray-500 uppercase font-bold">Confidence Score</span>
                        <p id="dispScore" class="font-semibold text-gray-800 mt-1">-</p>
                    </div>
                    <div class="bg-gray-50 p-3 rounded border">
                        <span class="text-xs text-gray-500 uppercase font-bold">Sentiment / Urgency</span>
                        <p id="dispTone" class="font-semibold text-gray-800 mt-1">-</p>
                    </div>
                    <div class="bg-gray-50 p-3 rounded border">
                        <span class="text-xs text-gray-500 uppercase font-bold">Escalated Status</span>
                        <p id="dispEscalated" class="font-semibold mt-1">-</p>
                    </div>
                </div>

                <!-- Operator Response Editor -->
                <div>
                    <label class="block text-sm font-semibold mb-1 text-gray-700">Response (Edit to Override or Leave to Approve):</label>
                    <textarea id="responseDraft" rows="6" class="w-full p-3 border rounded-md font-mono text-sm"></textarea>
                </div>

                <!-- Action Buttons -->
                <div class="flex gap-4 pt-2">
                    <button onclick="submitDecision('approved')" class="flex-1 bg-emerald-600 hover:bg-emerald-700 text-white font-semibold py-2 px-4 rounded-md">
                        ✅ Approve Response
                    </button>
                    <button onclick="submitDecision('overridden')" class="flex-1 bg-amber-600 hover:bg-amber-700 text-white font-semibold py-2 px-4 rounded-md">
                        ✏️ Save Custom Override
                    </button>
                </div>
                
                <div id="statusMessage" class="hidden p-3 rounded text-sm text-center font-medium"></div>
            </div>
        </div>

        <script>
            async function loadSession() {
                const sid = document.getElementById('sessionIdInput').value.trim();
                if (!sid) return alert('Please enter a session ID');

                const res = await fetch('/conversation/' + sid);
                if (!res.ok) {
                    alert('Session not found in Redis!');
                    return;
                }
                const data = await res.json();
                
                document.getElementById('detailsCard').classList.remove('hidden');
                document.getElementById('dispIntent').innerText = data.current_intent || 'None';
                document.getElementById('dispScore').innerText = (data.confidence_score * 100).toFixed(1) + '%';
                document.getElementById('dispTone').innerText = (data.sentiment || 'neutral') + ' / ' + (data.urgency || 'normal');
                
                const escBadge = document.getElementById('dispEscalated');
                if (data.escalated_to_human) {
                    escBadge.innerText = '🚨 Escalated to Human (' + (data.escalation_reason || 'Low confidence') + ')';
                    escBadge.className = 'font-semibold text-red-600 mt-1';
                } else {
                    escBadge.innerText = '✅ Autonomous Resolution';
                    escBadge.className = 'font-semibold text-emerald-600 mt-1';
                }

                // Pre-fill response draft
                document.getElementById('responseDraft').value = "Hello,\\n\\nThank you for reaching out. Here is our official response regarding your inquiry.\\n\\nBest regards,\\nSupport Team";
            }

            async function submitDecision(action) {
                const sid = document.getElementById('sessionIdInput').value.trim();
                const text = document.getElementById('responseDraft').value;

                // Send feedback back to system
                await fetch('/feedback', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({session_id: sid, rating: 5, comment: action.toUpperCase() + ': ' + text.slice(0, 50)})
                });

                const msg = document.getElementById('statusMessage');
                msg.classList.remove('hidden');
                msg.className = 'p-3 rounded text-sm text-center font-semibold bg-emerald-100 text-emerald-800';
                msg.innerText = '✅ Decision saved! Operator ' + action + ' the ticket response.';
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


# ============================================================
# 3. 🚀 Direct Execution
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)