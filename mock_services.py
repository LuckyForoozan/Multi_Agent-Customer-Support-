import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Mock service app running on port 8001 (matching your .env ZENDESK_API_URL)
mock_app = FastAPI(title="Zendesk & SMTP Mock Services", version="1.0.0")

# In-memory storage for tickets and emails
MOCK_ZENDESK_DB: Dict[str, Dict[str, Any]] = {}
MOCK_EMAIL_INBOX: List[Dict[str, Any]] = []


# ============================================================
# 1. 🎫 Zendesk Mock API (Standard Zendesk REST Format)
# ============================================================
class TicketPayload(BaseModel):
    subject: str
    description: str
    priority: str = "normal"  # low, normal, high, urgent
    status: str = "open"      # open, pending, solved, closed
    recipient: Optional[str] = None
    custom_fields: Optional[Dict[str, Any]] = None


@mock_app.post("/api/v2/tickets.json")
def create_zendesk_ticket(payload: TicketPayload):
    """Simulates creating a ticket in Zendesk."""
    ticket_id = f"ZD-{uuid.uuid4().hex[:6].upper()}"
    new_ticket = {
        "id": ticket_id,
        "subject": payload.subject,
        "description": payload.description,
        "priority": payload.priority,
        "status": payload.status,
        "recipient": payload.recipient,
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat()
    }
    MOCK_ZENDESK_DB[ticket_id] = new_ticket
    print(f"🎫 [Zendesk Mock] Ticket Created: {ticket_id} (Priority: {payload.priority})")
    return {"ticket": new_ticket}


@mock_app.get("/api/v2/tickets/{ticket_id}.json")
def get_zendesk_ticket(ticket_id: str):
    """Fetches a ticket from Zendesk by ID."""
    ticket = MOCK_ZENDESK_DB.get(ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail=f"Ticket {ticket_id} not found in Zendesk.")
    return {"ticket": ticket}


@mock_app.put("/api/v2/tickets/{ticket_id}.json")
def update_zendesk_ticket(ticket_id: str, payload: Dict[str, Any]):
    """Updates status or details of a Zendesk ticket."""
    ticket = MOCK_ZENDESK_DB.get(ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail=f"Ticket {ticket_id} not found.")
    ticket.update(payload)
    ticket["updated_at"] = datetime.utcnow().isoformat()
    print(f"🎫 [Zendesk Mock] Ticket {ticket_id} updated.")
    return {"ticket": ticket}


@mock_app.get("/api/v2/tickets.json")
def list_zendesk_tickets():
    """Lists all tickets currently stored in the Zendesk mock."""
    return {"tickets": list(MOCK_ZENDESK_DB.values()), "count": len(MOCK_ZENDESK_DB)}


# ============================================================
# 2. ✉️ SMTP Email Mock Service
# ============================================================
class EmailPayload(BaseModel):
    to_email: str
    subject: str
    body: str


@mock_app.post("/api/smtp/send")
def send_mock_email(payload: EmailPayload):
    """Simulates dispatching an email via SMTP."""
    email_record = {
        "email_id": f"MSG-{uuid.uuid4().hex[:6].upper()}",
        "to": payload.to_email,
        "subject": payload.subject,
        "body": payload.body,
        "sent_at": datetime.utcnow().isoformat(),
        "status": "delivered"
    }
    MOCK_EMAIL_INBOX.append(email_record)
    print(f"✉️ [SMTP Mock] Email sent to '{payload.to_email}' with subject: '{payload.subject}'")
    return {"status": "sent", "email": email_record}


@mock_app.get("/api/smtp/inbox")
def view_sent_emails():
    """View all dispatched emails."""
    return {"outbox": MOCK_EMAIL_INBOX, "total_sent": len(MOCK_EMAIL_INBOX)}


# ============================================================
# 3. 🚀 Runner
# ============================================================
if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Starting Zendesk & SMTP Mock Server on http://127.0.0.1:8001 ...")
    uvicorn.run("mock_services:mock_app", host="127.0.0.1", port=8001, reload=True)