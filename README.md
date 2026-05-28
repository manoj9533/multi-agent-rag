# Multi-Agent AI System — Hybrid RAG + Self-Evaluation

A LangGraph-based agentic system I built for enterprise customer support — using hybrid retrieval, cross-encoder reranking, and a self-critique loop to reduce hallucinations and improve answer quality on large knowledge bases.

---

## The Problem

Standard RAG setups have two major failure modes: retrieval misses relevant chunks, and the LLM still hallucinates even when good context is available. A single vector search isn't enough, and there's no mechanism to catch bad responses before they reach the user.

I wanted to build something that handled both — better retrieval and a way to validate the response before returning it.

---

## What It Does

- Breaks down user queries using a Planner agent before retrieval
- Runs hybrid retrieval — FAISS (dense) + BM25 (sparse) in parallel
- Reranks merged results using a cross-encoder
- Generates response via GPT-4
- Passes response through a Critic agent that scores quality and re-routes if confidence is below 0.7
- Evaluates every release automatically using RAGAS (faithfulness, context recall, answer relevancy)

---

## Agent Architecture

```
User Query
    │
    ▼
Planner Agent
Breaks query into sub-tasks, decides routing
    │
    ├──────────────────────────┐
    ▼                          ▼
Retriever Agent           Search Agent
FAISS dense search        BM25 sparse search
    │                          │
    └──────────┬───────────────┘
               ▼
       Cross-Encoder Reranker
       Scores & reranks top chunks
               │
               ▼
       GPT-4 Response Generation
               │
               ▼
       Critic Agent
       Scores faithfulness + relevancy
       confidence < 0.7 → back to Planner
               │
               ▼
       Final Response
```

---

## Tech Stack

- **Orchestration:** LangGraph
- **LLM:** GPT-4 (OpenAI)
- **Embeddings:** OpenAI text-embedding-3-small
- **Dense Retrieval:** FAISS
- **Sparse Retrieval:** BM25 (rank_bm25)
- **Reranking:** Cross-Encoder (sentence-transformers)
- **Evaluation:** RAGAS
- **Framework:** LangChain
- **API:** FastAPI
- **Infra:** Docker, AWS EC2

---

## Why Hybrid Retrieval

Dense search (FAISS) is good at semantic similarity but misses exact keyword matches. BM25 handles keywords well but doesn't understand intent. Combining both and reranking the merged results consistently outperformed either method alone.

```python
# Retrieve from both
faiss_results = faiss_index.similarity_search(query, k=10)
bm25_results = bm25.get_top_n(query.split(), corpus, n=10)

# Merge and rerank
merged = merge_and_deduplicate(faiss_results, bm25_results)
scores = cross_encoder.predict([(query, doc.page_content) for doc in merged])
top_k = [doc for _, doc in sorted(zip(scores, merged), reverse=True)][:5]
```

---

## The Critic Loop

The Critic agent scores each response on faithfulness and answer relevancy. If the combined confidence score falls below 0.7, it re-routes the query back to the Planner with context about what went wrong. This loop was the single biggest factor in reducing hallucinations.

```python
graph.add_conditional_edges(
    "critic",
    lambda state: "approved" if state["confidence"] >= 0.7 else "retry",
    {"approved": END, "retry": "planner"}
)
```

---

## RAGAS Evaluation

| Metric | Naive RAG | This System |
|---|---|---|
| Answer Relevancy | 0.61 | 0.84 |
| Faithfulness | 0.58 | 0.81 |
| Context Recall | 0.54 | 0.79 |

Evaluated on 500+ test queries against a 10K+ document knowledge base. Ran 3 rounds of prompt and retrieval tuning based on RAGAS findings.

---

## Repo Structure

```
multi-agent-rag/
├── agents/
│   ├── planner.py
│   ├── retriever.py
│   ├── search.py
│   └── critic.py
├── retrieval/
│   ├── faiss_index.py
│   ├── bm25_retrieval.py
│   └── reranker.py
├── evaluation/
│   ├── ragas_eval.py
│   └── test_queries.json
├── graph/
│   └── agent_graph.py
├── api/
│   └── main.py
├── data/
│   └── knowledge_base/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Running Locally

```bash
git clone https://github.com/manojkumargowd/multi-agent-rag
cd multi-agent-rag
pip install -r requirements.txt

# Add OpenAI key
cp .env.example .env

# Index documents
python retrieval/faiss_index.py --docs_path data/knowledge_base/

# Start API
uvicorn api.main:app --reload

# Run evaluation
python evaluation/ragas_eval.py
```

**Docker**
```bash
docker-compose up --build
```

---

## Results

- Answer relevancy improved from 0.61 → 0.84 on 500+ test queries
- Hallucination rate reduced ~40% via Critic self-loop
- Customer escalations dropped 45%
- Handles concurrent sessions with LangGraph conversation memory
- API latency under 2s for complex multi-hop queries

---

## What I'd Improve Next

- Persistent memory across sessions
- Query result caching for frequently asked questions
- Fine-tune embeddings on domain-specific data
- Add Langfuse for production observability



