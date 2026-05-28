import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import time

app = FastAPI(
    title="Multi-Agent RAG API",
    description="Enterprise Q&A using LangGraph, Hybrid RAG, and RAGAS evaluation.",
    version="1.0.0"
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class QueryRequest(BaseModel):
    query: str = Field(..., example="What is the return policy?")


class QueryResponse(BaseModel):
    query: str
    response: str
    confidence: float
    context_used: int
    iterations: int
    latency_ms: float


@app.get("/", tags=["health"])
def root():
    return {"status": "ok", "service": "multi-agent-rag", "version": "1.0.0"}


@app.get("/health", tags=["health"])
def health():
    return {"status": "healthy"}


@app.post("/query", response_model=QueryResponse, tags=["rag"])
def query(req: QueryRequest):
    start = time.time()
    from graph.agent_graph import run_pipeline
    result = run_pipeline(req.query)
    latency = round((time.time() - start) * 1000, 2)
    return QueryResponse(**result, latency_ms=latency)
