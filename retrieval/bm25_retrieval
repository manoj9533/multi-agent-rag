from typing import List

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False


class BM25Retriever:
    """Sparse keyword-based retrieval using BM25."""

    def __init__(self):
        self.bm25 = None
        self.documents = []

    def build_index(self, documents: List[str]):
        self.documents = documents
        if BM25_AVAILABLE:
            tokenized = [doc.lower().split() for doc in documents]
            self.bm25 = BM25Okapi(tokenized)
            print(f"BM25 index built — {len(documents)} documents indexed")

    def search(self, query: str, k: int = 10) -> List[str]:
        if not BM25_AVAILABLE or self.bm25 is None:
            return self.documents[:k]
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.documents[i] for i in top_indices]
