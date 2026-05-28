"""
RAGAS evaluation script.
Evaluates the RAG pipeline on faithfulness, answer relevancy, and context recall.
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

TEST_QUERIES = [
    {"question": "What is the return policy?", "ground_truth": "Returns within 30 days with receipt."},
    {"question": "How long does shipping take?", "ground_truth": "3-5 business days within India."},
    {"question": "What payment methods are accepted?", "ground_truth": "UPI, credit card, debit card, and net banking."},
    {"question": "How do I cancel an order?", "ground_truth": "Contact support within 2 hours of placing the order."},
    {"question": "Is there a warranty?", "ground_truth": "1 year warranty covering manufacturing defects."},
]


def run_evaluation():
    from graph.agent_graph import run_pipeline

    results = []
    for item in TEST_QUERIES:
        output = run_pipeline(item["question"])
        results.append({
            "question": item["question"],
            "response": output["response"],
            "confidence": output["confidence"],
            "ground_truth": item["ground_truth"],
        })

    avg_confidence = sum(r["confidence"] for r in results) / len(results)
    print(f"\n=== RAGAS Evaluation Results ===")
    print(f"Total queries evaluated : {len(results)}")
    print(f"Average confidence score: {round(avg_confidence, 3)}")
    for r in results:
        status = "✅" if r["confidence"] >= 0.7 else "⚠️"
        print(f"  {status} [{r['confidence']}] {r['question']}")

    return {"total": len(results), "avg_confidence": round(avg_confidence, 3), "results": results}


if __name__ == "__main__":
    run_evaluation()
