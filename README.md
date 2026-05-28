# Multi-Agent RAG System

LangGraph-style multi-agent Q&A pipeline with hybrid retrieval (FAISS + BM25), reranking, confidence scoring, and evaluation utilities.

## Current Status

- Planner, retriever, generator, and critic nodes are implemented.
- API endpoints are implemented in `api/main.py`.
- Hybrid retrieval is implemented with in-repo sample documents for quick demo runs.
- Evaluation script is available in `evaluation/ragas_eval.py`.

## Architecture

```text
User Query
	|
	v
Planner Agent -> Sub-queries
	|
	v
Retriever Agent -> FAISS + BM25 -> Reranker
	|
	v
Generator -> Response Draft
	|
	v
Critic -> Confidence + Retry Decision
```

## Project Structure

```text
multi-agent-rag/
├── .github/
│   └── workflows/
│       └── ci.yml
├── agents/
│   ├── planner.py
│   ├── retriever.py
│   └── critic.py
├── api/
│   └── main.py
├── evaluation/
│   └── ragas_eval.py
├── graph/
│   └── agent_graph.py
├── retrieval/
│   ├── bm25_retrieval.py
│   ├── faiss_index.py
│   └── reranker.py
├── tests/
│   └── test_agents.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Quick Start

```bash
git clone https://github.com/manoj9533/multi-agent-rag
cd multi-agent-rag
pip install -r requirements.txt
```

## Run Locally

```bash
# Start API
uvicorn api.main:app --reload

# Run evaluation script
python evaluation/ragas_eval.py
```

## API Endpoints

- `GET /`
- `GET /health`
- `POST /query`

### Example Request

```json
{
  "query": "What is the return policy?"
}
```

## Run Tests

```bash
python -m unittest discover -s tests -p "test_*.py"
```

## Notes

- If `OPENAI_API_KEY` is not configured, the generator falls back to heuristic responses.
- Retrieval currently uses sample in-code documents and should be replaced with production data sources for real deployments.
