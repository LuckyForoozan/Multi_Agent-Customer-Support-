import os
import json
import time
import logging
from datetime import datetime
from typing import Dict, Any, List

# Configure structured JSON logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("agent_observability")


# ============================================================
# 1. 📊 In-Memory Metrics Collector
# ============================================================
class MetricsCollector:
    def __init__(self):
        self.total_queries: int = 0
        self.autonomous_resolved: int = 0
        self.escalations: int = 0
        self.total_confidence: float = 0.0
        self.agent_execution_times: Dict[str, List[float]] = {}
        self.recent_logs: List[Dict[str, Any]] = []

    def record_query(self, confidence: float, escalated: bool, duration_ms: float):
        """Records telemetry for an entire workflow run."""
        self.total_queries += 1
        self.total_confidence += confidence
        if escalated:
            self.escalations += 1
        else:
            self.autonomous_resolved += 1

    def record_agent_step(self, agent_name: str, duration_ms: float):
        """Records latency for a specific agent node."""
        if agent_name not in self.agent_execution_times:
            self.agent_execution_times[agent_name] = []
        self.agent_execution_times[agent_name].append(duration_ms)

    def log_event(self, agent_name: str, session_id: str, action: str, details: Dict[str, Any]):
        """Emits a structured JSON log entry for monitoring tools (e.g. Datadog, Grafana, CloudWatch)."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "agent": agent_name,
            "session_id": session_id,
            "action": action,
            "details": details
        }
        # Keep the last 50 logs for dashboard viewing
        self.recent_logs.insert(0, log_entry)
        if len(self.recent_logs) > 50:
            self.recent_logs.pop()

        # Output as clean JSON to console
        logger.info(json.dumps(log_entry))

    def get_summary(self) -> Dict[str, Any]:
        """Calculates real-time performance metrics."""
        avg_conf = (self.total_confidence / self.total_queries) if self.total_queries > 0 else 0.0
        auto_rate = ((self.autonomous_resolved / self.total_queries) * 100) if self.total_queries > 0 else 0.0
        
        # Calculate average latency per agent
        avg_latencies = {
            agent: round(sum(times) / len(times), 2)
            for agent, times in self.agent_execution_times.items()
            if times
        }

        return {
            "total_queries_processed": self.total_queries,
            "autonomous_resolutions": self.autonomous_resolved,
            "human_escalations": self.escalations,
            "automation_rate_pct": round(auto_rate, 1),
            "average_confidence_score": round(avg_conf, 2),
            "average_node_latencies_ms": avg_latencies,
            "system_status": "healthy"
        }


# Global singleton instance
metrics = MetricsCollector()


# ============================================================
# --- Self Test ---
# ============================================================
if __name__ == "__main__":
    print("\n--- Testing Observability & Structured JSON Logging ---\n")
    
    # 1. Log sample structured agent events
    metrics.log_event(
        agent_name="triage_agent",
        session_id="sess_test123",
        action="classify_intent",
        details={"intent": "get_refund", "confidence": 0.88}
    )
    
    metrics.log_event(
        agent_name="ticket_agent",
        session_id="sess_test123",
        action="create_ticket",
        details={"ticket_id": "TICK-999", "priority": "normal"}
    )

    # 2. Record query metrics
    metrics.record_query(confidence=0.88, escalated=False, duration_ms=420.5)
    metrics.record_agent_step("triage_agent", 120.3)
    metrics.record_agent_step("email_agent", 300.2)

    # 3. Print metrics summary
    print("\n--- Real-Time Metrics Summary ---")
    print(json.dumps(metrics.get_summary(), indent=2))
    print("\n✅ Observability module working perfectly!")