import os
import json
import uuid
from typing import Optional
import redis
import fakeredis
from dotenv import load_dotenv
from models import ConversationState, TicketState, SessionSnapshot

# Load environment variables from .env
load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# Initialize Redis client with automatic fallback to FakeRedis
try:
    redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    redis_client.ping()
    print("✅ Connected to live Redis server.")
except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError):
    print("ℹ️ Live Redis not running. Using in-memory FakeRedis.")
    redis_client = fakeredis.FakeRedis(decode_responses=True)


# --- Conversation State Helpers ---

def save_conversation(state: ConversationState, ttl_seconds: int = 86400) -> bool:
    """
    Save or update a conversation state in Redis.
    ttl_seconds defaults to 24 hours.
    """
    key = f"conversation:{state.session_id}"
    # Use Pydantic's model_dump_json for clean serialization
    data_json = state.model_dump_json()
    return bool(redis_client.set(key, data_json, ex=ttl_seconds))


def get_conversation(session_id: str) -> Optional[ConversationState]:
    """
    Retrieve and reconstruct a conversation state from Redis.
    """
    key = f"conversation:{session_id}"
    data_json = redis_client.get(key)
    if not data_json:
        return None
    return ConversationState.model_validate_json(data_json)


# --- Ticket State Helpers ---

def save_ticket(ticket: TicketState, ttl_seconds: int = 86400 * 7) -> bool:
    """
    Save or update a ticket state in Redis (retained for 7 days).
    """
    key = f"ticket:{ticket.ticket_id}"
    return bool(redis_client.set(key, ticket.model_dump_json(), ex=ttl_seconds))


def get_ticket(ticket_id: str) -> Optional[TicketState]:
    """
    Retrieve a ticket by its ID.
    """
    key = f"ticket:{ticket_id}"
    data_json = redis_client.get(key)
    if not data_json:
        return None
    return TicketState.model_validate_json(data_json)


# --- Session Snapshot Helper ---

def save_snapshot(state: ConversationState) -> SessionSnapshot:
    """
    Creates an immutable snapshot of the conversation state for audits and human review.
    """
    snapshot = SessionSnapshot(
        snapshot_id=str(uuid.uuid4()),
        session_id=state.session_id,
        state=state
    )
    key = f"snapshot:{snapshot.snapshot_id}"
    redis_client.set(key, snapshot.model_dump_json(), ex=86400 * 30)  # keep 30 days
    return snapshot


# --- Self Test ---
if __name__ == "__main__":
    print("Testing Redis connection...")
    try:
        redis_client.ping()
        print("✅ Redis server connected successfully!\n")

        # Test creating and saving state
        sample_state = ConversationState(
            session_id="session_101",
            customer_id="cust_abc",
            current_intent="cancel_order",
            confidence_score=0.92,
            sentiment="neutral"
        )
        
        print("Saving sample conversation state to Redis...")
        save_conversation(sample_state)

        print("Reading state back from Redis...")
        loaded_state = get_conversation("session_101")
        print(f"Loaded Session ID: {loaded_state.session_id}")
        print(f"Loaded Intent    : {loaded_state.current_intent}")
        print(f"Loaded Confidence: {loaded_state.confidence_score}")
        print("\n✅ State layer working perfectly!")

    except redis.exceptions.ConnectionError:
        print("⚠️ Could not connect to a local Redis server at localhost:6379.")
        print("Tip: If you do not have Redis installed on Windows, let me know and we can use a zero-install local mock (fakeredis)!")