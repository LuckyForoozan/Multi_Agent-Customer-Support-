import time
import json
from typing import List, Dict, Any
from workflow import process_customer_query

# ============================================================
# 🧪 Test Dataset: Standard Requests, Angry Queries & Edge Cases
# ============================================================
TEST_CASES: List[Dict[str, Any]] = [
    {
        "description": "Standard Order Cancellation",
        "query": "I would like to cancel my order #98412 please.",
        "expected_intent": "cancel_order",
        "should_escalate": False
    },
    {
        "description": "Standard Refund Request",
        "query": "Can I get a refund for the item I returned last week?",
        "expected_intent": "get_refund",
        "should_escalate": False
    },
    {
        "description": "High-Urgency & Negative Tone",
        "query": "I am furious! My order arrived completely broken and nobody is answering me!",
        "expected_intent": "get_refund",
        "should_escalate": False
    },
    {
        "description": "Out-of-Scope / Nonsense Query (Low Confidence)",
        "query": "Can a quantum giraffe fly to Jupiter yesterday?",
        "expected_intent": "unknown",
        "should_escalate": True
    },
    {
        "description": "Vague Incomplete Inquiry",
        "query": "Hello I have an issue with something.",
        "expected_intent": "unknown",
        "should_escalate": True
    }
]


# ============================================================
# 🚀 Benchmark Runner
# ============================================================
def run_evaluation():
    print("\n" + "="*70)
    print("🧪 MULTI-AGENT WORKFLOW EVALUATION & BENCHMARK SUITE")
    print("="*70)

    results = []
    total_start_time = time.time()

    for idx, test in enumerate(TEST_CASES, 1):
        print(f"\n[{idx}/{len(TEST_CASES)}] Testing: {test['description']}")
        print(f"     Query: \"{test['query']}\"")

        start = time.time()
        try:
            output = process_customer_query(test["query"], customer_id=f"test_user_{idx}")
            duration = round(time.time() - start, 2)

            detected_intent = output.get("current_intent")
            confidence = output.get("confidence_score", 0.0)
            escalated = output.get("escalated_to_human", False)
            ticket_id = output.get("ticket", {}).get("ticket_id") if isinstance(output.get("ticket"), dict) else "N/A"
            has_response = bool(output.get("draft_response"))

            # Evaluate pass/fail conditions
            escalation_correct = (escalated == test["should_escalate"])

            result_entry = {
                "test": test["description"],
                "duration_sec": duration,
                "confidence": confidence,
                "detected_intent": detected_intent,
                "escalated": escalated,
                "escalation_matched": escalation_correct,
                "ticket_created": ticket_id,
                "has_email_draft": has_response
            }
            results.append(result_entry)

            status_icon = "✅ PASS" if escalation_correct else "⚠️ REVIEW"
            print(f"     Outcome   : {status_icon} | Escalated: {escalated} | Conf: {confidence:.2f} | Time: {duration}s")

        except Exception as e:
            print(f"     ❌ ERROR : {e}")
            results.append({
                "test": test["description"],
                "error": str(e),
                "escalation_matched": False
            })

    total_duration = round(time.time() - total_start_time, 2)

    # ============================================================
    # 📊 Final Evaluation Summary Report
    # ============================================================
    valid_results = [r for r in results if "error" not in r]
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r.get("escalation_matched"))
    escalation_count = sum(1 for r in valid_results if r["escalated"])
    auto_count = len(valid_results) - escalation_count
    avg_conf = sum(r["confidence"] for r in valid_results) / len(valid_results) if valid_results else 0.0
    avg_latency = sum(r["duration_sec"] for r in valid_results) / len(valid_results) if valid_results else 0.0

    print("\n" + "="*70)
    print("📈 FINAL BENCHMARK REPORT")
    print("="*70)
    print(f"• Total Test Cases Evaluated : {total_tests}")
    print(f"• Successful Decision Routing: {passed_tests}/{total_tests} ({(passed_tests/total_tests)*100:.1f}%)")
    print(f"• Autonomous Resolutions     : {auto_count} ({(auto_count/total_tests)*100:.1f}%)")
    print(f"• Human Escalations (HITL)   : {escalation_count} ({(escalation_count/total_tests)*100:.1f}%)")
    print(f"• Average Confidence Score   : {avg_conf:.2f}")
    print(f"• Average Resolution Latency : {avg_latency:.2f} seconds")
    print(f"• Total Evaluation Duration  : {total_duration:.2f} seconds")
    print("="*70 + "\n")


if __name__ == "__main__":
    run_evaluation()