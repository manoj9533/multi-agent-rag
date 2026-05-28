import os
import pickle
from pathlib import Path
from typing import List

try:
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class FAISSRetriever:
    """Dense vector retrieval using FAISS."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name) if FAISS_AVAILABLE else None
        self.index = None
        self.documents = []

    def build_index(self, documents: List[str]):
        """Build FAISS index from documents."""
        if not FAISS_AVAILABLE:
            print("FAISS not available — skipping index build")
            self.documents = documents
            return

        self.documents = documents
        embeddings = self.model.encode(documents, show_progress_bar=True)
        embeddings = np.array(embeddings).astype("float32")
        faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        print(f"FAISS index built — {len(documents)} documents indexed")

    def search(self, query: str, k: int = 10) -> List[str]:
        """Retrieve top-k documents for a query."""
        if not FAISS_AVAILABLE or self.index is None:
            return self.documents[:k]

        q_emb = self.model.encode([query])
        q_emb = np.array(q_emb).astype("float32")
        faiss.normalize_L2(q_emb)

        scores, indices = self.index.search(q_emb, k)
        return [self.documents[i] for i in indices[0] if i < len(self.documents)]

    def save(self, path: str = "data/faiss_index"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if FAISS_AVAILABLE and self.index:
            faiss.write_index(self.index, f"{path}.idx")
        with open(f"{path}_docs.pkl", "wb") as f:
            pickle.dump(self.documents, f)

    def load(self, path: str = "data/faiss_index"):
        if FAISS_AVAILABLE:
            self.index = faiss.read_index(f"{path}.idx")
        with open(f"{path}_docs.pkl", "rb") as f:
            self.documents = pickle.load(f)
