import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from agents.planner import AgentState, planner_node
from agents.retriever import retriever_node
from agents.critic import critic_node, should_retry

try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False


def generate_response(state: AgentState) -> AgentState:
    """Response generation node (uses OpenAI if key available, else heuristic)."""
    query = state["query"]
    context = state.get("context", [])

    try:
        from openai import OpenAI
        import os
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
        context_str = "\n".join(context)
        completion = client.chat.completions.create(
            model=os.environ.get("LLM_MODEL", "gpt-4o"),
            messages=[
                {"role": "system", "content": f"Answer using this context:\n{context_str}"},
                {"role": "user", "content": query}
            ]
        )
        response = completion.choices[0].message.content
    except Exception:
        # Fallback without OpenAI
        if context:
            response = f"Based on available information: {context[0]}"
        else:
            response = "I could not find relevant information to answer your question."

    state["response"] = response
    return state


def build_graph():
    """Build and compile the LangGraph multi-agent workflow."""
    if not LANGGRAPH_AVAILABLE:
        print("LangGraph not available — using fallback pipeline")
        return None

    graph = StateGraph(AgentState)
    graph.add_node("planner", planner_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("generator", generate_response)
    graph.add_node("critic", critic_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "retriever")
    graph.add_edge("retriever", "generator")
    graph.add_edge("generator", "critic")
    graph.add_conditional_edges("critic", should_retry, {"retry": "planner", "approved": END})

    return graph.compile()


def run_pipeline(query: str) -> dict:
    """Run the full multi-agent RAG pipeline."""
    initial_state = AgentState(
        query=query, sub_queries=[], context=[],
        response="", confidence=0.0, iterations=0
    )

    graph = build_graph()
    if graph:
        result = graph.invoke(initial_state)
    else:
        # Fallback linear pipeline
        state = planner_node(initial_state)
        state = retriever_node(state)
        state = generate_response(state)
        state = critic_node(state)
        result = state

    return {
        "query": result["query"],
        "response": result["response"],
        "confidence": result["confidence"],
        "context_used": len(result.get("context", [])),
        "iterations": result.get("iterations", 1)
    }


if __name__ == "__main__":
    out = run_pipeline("What is the return policy?")
    print(out)
