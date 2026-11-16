import os
import uuid
from typing import TypedDict, Optional, List, Dict, Any
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

# Import your specialized agents and state manager
from agents import (
    sentiment_agent,
    triage_agent,
    policy_agent,
    ticket_agent,
    email_agent
)
from slack_notifier import send_slack_escalation_alert

from state_manager import save_conversation, ConversationState

load_dotenv()
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.70"))
MAX_RETRIES = 2


# ============================================================
# 1. 📋 Graph State Schema (TypedDict for LangGraph)
# ============================================================
class SupportState(TypedDict, total=False):
    query: str
    customer_id: str
    session_id: str
    
    # Analysis fields
    sentiment: Optional[str]
    urgency: Optional[str]
    current_intent: Optional[str]
    confidence_score: Optional[float]
    category: Optional[str]
    retrieved_context: Optional[List[Dict[str, Any]]]
    policy_guideline: Optional[str]
    
    # Output fields
    ticket: Optional[Dict[str, Any]]
    draft_response: Optional[str]
    escalated_to_human: Optional[bool]
    escalation_reason: Optional[str]
    
    # Cycle detection & retries
    retry_count: int


# ============================================================
# 2. 👤 Human Escalation Node
# ============================================================
def human_escalation_node(state: SupportState) -> Dict[str, Any]:
    """Triggered when confidence is low or query is too complex/critical."""
    reason = state.get("escalation_reason", "Confidence score below required threshold.")
    print(f"🚨 [Human Escalation] Flagged for Operator review! Reason: {reason}")
    
    # Trigger real-time alert to Slack
    send_slack_escalation_alert(
        ticket_id=state.get("session_id", "PENDING_TICKET"),
        customer_id=state.get("customer_id", "guest"),
        query=state.get("query", ""),
        confidence=state.get("confidence_score", 0.0),
        reason=reason,
        sentiment=state.get("sentiment", "neutral"),
        urgency=state.get("urgency", "normal")
    )
    
    draft = (
        "Thank you for contacting us. Your request has been transferred to a "
        "senior human support representative who will review your details and respond shortly."
    )
    return {
        "escalated_to_human": True,
        "escalation_reason": reason,
        "draft_response": draft
    }
# ============================================================
# 3. 🧠 Supervisor / Conditional Router
# ============================================================
def supervisor_router(state: SupportState) -> str:
    """
    Decides the next step after Triage:
    - If confidence >= threshold -> proceed to Policy & Resolution
    - If confidence < threshold  -> escalate to Human
    - If retry limit exceeded    -> prevent cycles & escalate
    """
    confidence = state.get("confidence_score", 0.0)
    retries = state.get("retry_count", 0)

    # 1. Cycle & infinite loop prevention
    if retries >= MAX_RETRIES:
        state["escalation_reason"] = "Exceeded maximum retry attempts without clear resolution."
        return "human_escalation"

    # 2. Confidence threshold check
    print(f"🧠 [Supervisor] Evaluating confidence: {confidence:.2f} (Threshold: {CONFIDENCE_THRESHOLD})")
    if confidence >= CONFIDENCE_THRESHOLD:
        print("🧠 [Supervisor] Decision: High confidence -> Route to Policy Agent")
        return "policy_agent"
    else:
        print("🧠 [Supervisor] Decision: Low confidence -> Route to Human Escalation")
        return "human_escalation"


# ============================================================
# 4. 🏗️ Build & Assemble the LangGraph Workflow
# ============================================================
workflow = StateGraph(SupportState)

# Add all specialized nodes
workflow.add_node("sentiment_agent", sentiment_agent)
workflow.add_node("triage_agent", triage_agent)
workflow.add_node("policy_agent", policy_agent)
workflow.add_node("ticket_agent", ticket_agent)
workflow.add_node("email_agent", email_agent)
workflow.add_node("human_escalation", human_escalation_node)

# Define edges (Execution Flow)
workflow.add_edge(START, "sentiment_agent")
workflow.add_edge("sentiment_agent", "triage_agent")

# Conditional Edge from Triage via Supervisor Router
workflow.add_conditional_edges(
    "triage_agent",
    supervisor_router,
    {
        "policy_agent": "policy_agent",
        "human_escalation": "human_escalation"
    }
)

# Automated path: Policy -> Ticket -> Email -> END
workflow.add_edge("policy_agent", "ticket_agent")
workflow.add_edge("ticket_agent", "email_agent")
workflow.add_edge("email_agent", END)

# Escalated path: Human Escalation -> Ticket -> END
workflow.add_edge("human_escalation", "ticket_agent")

# Compile graph
app = workflow.compile()


# ============================================================
# 5. 🚀 Helper to Run and Persist to Redis
# ============================================================
def process_customer_query(query: str, customer_id: str = "cust_001") -> Dict[str, Any]:
    session_id = f"sess_{uuid.uuid4().hex[:8]}"
    initial_state: SupportState = {
        "query": query,
        "customer_id": customer_id,
        "session_id": session_id,
        "retry_count": 0,
        "escalated_to_human": False
    }

    print(f"\n{'='*60}")
    print(f"🚀 Running Multi-Agent Workflow for Session: {session_id}")
    print(f"Customer Query: '{query}'")
    print(f"{'='*60}")

    final_output = app.invoke(initial_state)

    # Persist durable conversation state to Redis / FakeRedis
    conv_state = ConversationState(
        session_id=session_id,
        customer_id=customer_id,
        current_intent=final_output.get("current_intent"),
        confidence_score=final_output.get("confidence_score", 0.0),
        sentiment=final_output.get("sentiment", "neutral"),
        urgency=final_output.get("urgency", "normal"),
        escalated_to_human=final_output.get("escalated_to_human", False),
        escalation_reason=final_output.get("escalation_reason"),
        retrieved_context=final_output.get("retrieved_context", [])
    )
    save_conversation(conv_state)

    return final_output


# ============================================================
# --- Self Test ---
# ============================================================
if __name__ == "__main__":
    # Test 1: High confidence query (Autonomous resolution)
    result_auto = process_customer_query(
        "I need to cancel my order #48291 please", 
        customer_id="cust_alice"
    )
    print("\n--- TEST 1 OUTCOME ---")
    print(f"Escalated to Human? : {result_auto.get('escalated_to_human')}")
    print(f"Ticket Created ID   : {result_auto.get('ticket', {}).get('ticket_id')}")
    print(f"Response Preview    : {result_auto.get('draft_response')[:150]}...\n")

    # Test 2: Unclear/nonsense query (Triggers Human-in-the-Loop Escalation)
    result_escalated = process_customer_query(
        "Can a quantum giraffe fly over Mars yesterday?", 
        customer_id="cust_bob"
    )
    print("\n--- TEST 2 OUTCOME ---")
    print(f"Escalated to Human? : {result_escalated.get('escalated_to_human')}")
    print(f"Escalation Reason   : {result_escalated.get('escalation_reason')}")
    print(f"Ticket Priority     : {result_escalated.get('ticket', {}).get('priority')}")
    print(f"Response Preview    : {result_escalated.get('draft_response')}\n")