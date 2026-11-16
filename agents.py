import os
import json
import uuid
from typing import Dict, Any
from dotenv import load_dotenv
from groq import Groq
from retriever import retrieve_context
from models import TicketState

# Load environment variables
load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL_NAME = "openai/gpt-oss-120b"  # Fast, accurate, and free on Groq


def call_llm(system_prompt: str, user_prompt: str, json_mode: bool = False) -> str:
    """Helper to query Groq LLM."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2,
        response_format={"type": "json_object"} if json_mode else None
    )
    return response.choices[0].message.content


# ============================================================
# 1. 📊 Sentiment Agent
# ============================================================
def sentiment_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Analyzes customer tone and urgency."""
    query = state.get("query", "")
    
    prompt = f"""
    Analyze the customer message and return a JSON object with:
    - "sentiment": one of ["positive", "neutral", "negative", "angry"]
    - "urgency": one of ["low", "normal", "high"]
    
    Customer message: "{query}"
    """
    
    raw = call_llm("You are a customer sentiment analyzer. Respond in JSON only.", prompt, json_mode=True)
    parsed = json.loads(raw)
    
    print(f"📊 [Sentiment Agent] Tone: {parsed.get('sentiment')}, Urgency: {parsed.get('urgency')}")
    return {
        "sentiment": parsed.get("sentiment", "neutral"),
        "urgency": parsed.get("urgency", "normal")
    }


# ============================================================
# 2. 🔍 Triage Agent
# ============================================================
def triage_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Classifies intent using vector retrieval score + LLM confirmation."""
    query = state.get("query", "")
    
    # 1. Vector similarity search against Bitext dataset
    matches = retrieve_context(query, top_k=2)
    top_score = matches[0]["similarity_score"] if matches else 0.0
    top_intent = matches[0]["intent"] if matches else "unknown"
    
    # 2. Ask LLM to confirm classification
    prompt = f"""
    Given the customer inquiry, verify if it matches the detected intent:
    Customer inquiry: "{query}"
    Suggested intent: "{top_intent}" (Vector similarity: {top_score:.2f})
    
    Respond in JSON:
    {{
        "intent": "{top_intent}",
        "confidence": <float between 0.0 and 1.0>,
        "category": "{matches[0]['category'] if matches else 'GENERAL'}"
    }}
    """
    raw = call_llm("You are an intent triage specialist. Respond in JSON only.", prompt, json_mode=True)
    parsed = json.loads(raw)
    
    # Combined confidence: 60% vector similarity + 40% LLM score
    llm_conf = float(parsed.get("confidence", 0.7))
    combined_score = round((top_score * 0.6) + (llm_conf * 0.4), 2)
    
    print(f"🔍 [Triage Agent] Intent: {parsed.get('intent')}, Confidence: {combined_score}")
    return {
        "current_intent": parsed.get("intent", top_intent),
        "confidence_score": combined_score,
        "category": parsed.get("category", "GENERAL"),
        "retrieved_context": matches
    }


# ============================================================
# 3. 📋 Policy Agent
# ============================================================
def policy_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Retrieves standard company policy/response guidelines."""
    intent = state.get("current_intent", "")
    context = state.get("retrieved_context", [])
    
    # Use best approved response from Bitext dataset
    policy_reference = context[0]["response"] if context else "Follow standard customer service guidelines."
    
    print(f"📋 [Policy Agent] Policy guidelines loaded for intent '{intent}'.")
    return {
        "policy_guideline": policy_reference
    }


# ============================================================
# 4. 🎫 Ticket Agent
# ============================================================
def ticket_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Creates or updates ticket in support system (Zendesk mock)."""
    customer_id = state.get("customer_id", "guest_user")
    intent = state.get("current_intent", "general")
    urgency = state.get("urgency", "normal")
    category = state.get("category", "GENERAL")
    
    # Map urgency to ticket priority
    priority = "urgent" if urgency == "high" else "normal"
    
    ticket = TicketState(
        ticket_id=f"TICK-{uuid.uuid4().hex[:6].upper()}",
        customer_id=customer_id,
        category=category,
        intent=intent,
        priority=priority,
        status="open"
    )
    
    print(f"🎫 [Ticket Agent] Created Ticket: {ticket.ticket_id} (Priority: {ticket.priority})")
    return {"ticket": ticket.model_dump()}


# ============================================================
# 5. ✉️ Email Agent
# ============================================================
def email_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Drafts personalized, professional customer response."""
    query = state.get("query", "")
    policy = state.get("policy_guideline", "")
    sentiment = state.get("sentiment", "neutral")
    
    prompt = f"""
    Draft a helpful, empathetic customer support email response.
    
    Customer Question: "{query}"
    Customer Tone: "{sentiment}"
    Approved Company Policy/Answer: "{policy}"
    
    Rules:
    - Be professional and clear.
    - Tailor empathy to the tone (apologize if customer is negative/angry).
    - Do NOT make up fake policies; adhere to the approved guidelines.
    """
    
    draft = call_llm("You are a professional customer support email specialist.", prompt)
    
    print("✉️ [Email Agent] Draft generated successfully.")
    return {"draft_response": draft}


# ============================================================
# --- Self-Test All 5 Agents ---
# ============================================================
if __name__ == "__main__":
    print("\n--- Testing All 5 Specialized Agents with Groq ---")
    
    test_state = {
        "query": "I am furious! My order arrived completely broken and I want an immediate refund!",
        "customer_id": "cust_123"
    }
    
    # 1. Run Sentiment
    sentiment_out = sentiment_agent(test_state)
    test_state.update(sentiment_out)
    
    # 2. Run Triage
    triage_out = triage_agent(test_state)
    test_state.update(triage_out)
    
    # 3. Run Policy
    policy_out = policy_agent(test_state)
    test_state.update(policy_out)
    
    # 4. Run Ticket
    ticket_out = ticket_agent(test_state)
    test_state.update(ticket_out)
    
    # 5. Run Email
    email_out = email_agent(test_state)
    test_state.update(email_out)
    
    print("\n" + "="*50)
    print("🎯 FINAL EMAIL DRAFT:")
    print("="*50)
    print(test_state["draft_response"])