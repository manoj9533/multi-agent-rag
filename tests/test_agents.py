import os
import sys
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agents.planner import planner_node
from agents.critic import critic_node, should_retry


class TestAgents(unittest.TestCase):
    def test_planner_splits_on_and(self):
        state = {
            "query": "refund policy and shipping timeline",
            "sub_queries": [],
            "context": [],
            "response": "",
            "confidence": 0.0,
            "iterations": 0,
        }
        out = planner_node(state)
        self.assertGreaterEqual(len(out["sub_queries"]), 2)

    def test_critic_sets_confidence_and_retry(self):
        state = {
            "query": "What is return policy?",
            "sub_queries": ["What is return policy?"],
            "context": ["Returns are allowed within 30 days with receipt."],
            "response": "Returns are allowed within 30 days with receipt.",
            "confidence": 0.0,
            "iterations": 0,
        }
        out = critic_node(state)
        self.assertIn("confidence", out)
        decision = should_retry(out)
        self.assertIn(decision, {"retry", "approved"})


if __name__ == "__main__":
    unittest.main()
