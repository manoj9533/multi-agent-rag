from typing import List, Tuple

try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False


class CrossEncoderReranker:
    """Reranks merged retrieval results using a cross-encoder model."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name) if CROSS_ENCODER_AVAILABLE else None

    def rerank(self, query: str, documents: List[str], top_k: int = 5) -> List[str]:
        """Rerank documents and return top-k."""
        if not documents:
            return []

        if not CROSS_ENCODER_AVAILABLE or self.model is None:
            return documents[:top_k]

        pairs = [(query, doc) for doc in documents]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(scores, documents), reverse=True)
        return [doc for _, doc in ranked[:top_k]]

    def merge_and_rerank(self, query: str, faiss_docs: List[str], bm25_docs: List[str], top_k: int = 5) -> List[str]:
        """Merge dense + sparse results, deduplicate, then rerank."""
        seen = set()
        merged = []
        for doc in faiss_docs + bm25_docs:
            key = doc[:100]
            if key not in seen:
                seen.add(key)
                merged.append(doc)
        return self.rerank(query, merged, top_k)
