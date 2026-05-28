from typing import TypedDict, List


class AgentState(TypedDict):
    query: str
    sub_queries: List[str]
    context: List[str]
    response: str
    confidence: float
    iterations: int


def planner_node(state: AgentState) -> AgentState:
    """
    Planner Agent: breaks the user query into focused sub-queries
    to improve retrieval coverage.
    """
    query = state["query"]

    # Simple rule-based decomposition (replace with LLM call when API key available)
    sub_queries = [query]
    if "and" in query.lower():
        parts = query.lower().split("and")
        sub_queries = [p.strip() for p in parts if p.strip()]
    elif len(query.split()) > 10:
        words = query.split()
        mid = len(words) // 2
        sub_queries = [" ".join(words[:mid]), " ".join(words[mid:])]

    print(f"[Planner] Sub-queries: {sub_queries}")
    state["sub_queries"] = sub_queries
    return state
