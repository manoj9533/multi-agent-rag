import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from agents.planner import AgentState
from retrieval.faiss_index import FAISSRetriever
from retrieval.bm25_retrieval import BM25Retriever
from retrieval.reranker import CrossEncoderReranker

# Sample knowledge base — replace with real documents
SAMPLE_DOCS = [
    "Our return policy allows returns within 30 days with a receipt.",
    "Product warranty covers manufacturing defects for 1 year.",
    "Customer support is available Monday to Friday, 9am to 6pm IST.",
    "Shipping typically takes 3-5 business days within India.",
    "Premium members get free shipping on all orders above Rs 499.",
    "Refunds are processed within 7 business days after return approval.",
    "To cancel an order, contact support within 2 hours of placing it.",
    "Our mobile app is available on iOS and Android platforms.",
    "Payment options include UPI, credit card, debit card, and net banking.",
    "Track your order using the tracking ID sent to your registered email.",
]

faiss_retriever = FAISSRetriever()
bm25_retriever = BM25Retriever()
reranker = CrossEncoderReranker()

faiss_retriever.build_index(SAMPLE_DOCS)
bm25_retriever.build_index(SAMPLE_DOCS)


def retriever_node(state: AgentState) -> AgentState:
    """Retriever Agent: runs hybrid FAISS + BM25 retrieval and reranks."""
    all_context = []
    for sub_query in state.get("sub_queries", [state["query"]]):
        faiss_docs = faiss_retriever.search(sub_query, k=5)
        bm25_docs = bm25_retriever.search(sub_query, k=5)
        top_docs = reranker.merge_and_rerank(sub_query, faiss_docs, bm25_docs, top_k=3)
        all_context.extend(top_docs)

    # Deduplicate
    seen = set()
    unique_context = []
    for doc in all_context:
        if doc not in seen:
            seen.add(doc)
            unique_context.append(doc)

    print(f"[Retriever] Retrieved {len(unique_context)} unique chunks")
    state["context"] = unique_context
    return state
