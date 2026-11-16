import os
import json
import requests
from dotenv import load_dotenv

# Load tokens from .env
load_dotenv()

SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_CHANNEL_ID = os.getenv("SLACK_CHANNEL_ID")


def send_slack_escalation_alert(
    ticket_id: str,
    customer_id: str,
    query: str,
    confidence: float,
    reason: str,
    sentiment: str = "neutral",
    urgency: str = "normal"
) -> bool:
    """
    Sends a formatted notification to your Slack channel alerting human
    operators that an inquiry has been escalated for manual review.
    """
    if not SLACK_BOT_TOKEN or not SLACK_CHANNEL_ID:
        print("⚠️ Slack token or channel ID not configured in .env. Skipping Slack alert.")
        return False

    url = "https://slack.com/api/chat.postMessage"
    headers = {
        "Authorization": f"Bearer {SLACK_BOT_TOKEN}",
        "Content-Type": "application/json; charset=utf-8"
    }

    message_text = (
        f"🚨 *HUMAN ESCALATION REQUIRED*\n"
        f"• *Ticket ID:* `{ticket_id}`\n"
        f"• *Customer:* `{customer_id}`\n"
        f"• *Query:* \"{query}\"\n"
        f"• *Confidence Score:* `{confidence:.2f}` (Threshold: 0.70)\n"
        f"• *Tone / Urgency:* `{sentiment}` / `{urgency}`\n"
        f"• *Reason:* {reason}\n"
        f"👉 *Action:* Please review in Human Dashboard."
    )

    payload = {
        "channel": SLACK_CHANNEL_ID,
        "text": message_text
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=5)
        res_data = response.json()
        
        if res_data.get("ok"):
            print(f"📢 [Slack Alert Sent] Notification delivered to Slack channel {SLACK_CHANNEL_ID}.")
            return True
        else:
            print(f"⚠️ [Slack Alert Failed] Slack API responded with error: {res_data.get('error')}")
            return False
            
    except Exception as e:
        print(f"⚠️ [Slack Alert Error] Could not connect to Slack: {e}")
        return False


# --- Test Sending an Alert ---
if __name__ == "__main__":
    print("\n--- Testing Slack Operator Alert ---")
    send_slack_escalation_alert(
        ticket_id="TICK-DEMO01",
        customer_id="alice_99",
        query="Can your quantum giraffe fly over Mars yesterday?",
        confidence=0.22,
        reason="Confidence score below threshold (0.22 < 0.70)",
        sentiment="neutral",
        urgency="normal"
    )