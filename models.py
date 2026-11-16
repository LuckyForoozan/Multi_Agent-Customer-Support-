from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str  # "customer", "agent", "system", "human"
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class TicketState(BaseModel):
    ticket_id: str
    customer_id: str
    category: Optional[str] = None
    intent: Optional[str] = None
    priority: str = "normal"  # "low", "normal", "high", "urgent"
    status: str = "open"  # "open", "pending_human", "resolved", "closed"
    assigned_to: Optional[str] = None  # agent name or human operator
    draft_response: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class ConversationState(BaseModel):
    session_id: str
    customer_id: str
    messages: List[Message] = []
    
    # Classification & Sentiment
    current_intent: Optional[str] = None
    confidence_score: float = 0.0
    sentiment: str = "neutral"  # "positive", "neutral", "negative", "angry"
    urgency: str = "normal"  # "low", "normal", "high"
    
    # Workflow routing & escalation
    retrieved_context: List[Dict[str, Any]] = []
    escalated_to_human: bool = False
    escalation_reason: Optional[str] = None
    
    # Linked ticket
    ticket: Optional[TicketState] = None


class SessionSnapshot(BaseModel):
    snapshot_id: str
    session_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    state: ConversationState