from agents.planner import AgentState


def critic_node(state: AgentState) -> AgentState:
    """
    Critic Agent: scores the response quality.
    If confidence < threshold, routes back to planner for retry.
    """
    response = state.get("response", "")
    context = state.get("context", [])

    # Simple heuristic scoring (replace with LLM-based scoring when API key available)
    score = 0.5

    # Boost if response references context
    context_words = set(" ".join(context).lower().split())
    response_words = set(response.lower().split())
    overlap = len(context_words & response_words) / max(len(response_words), 1)
    score = min(0.5 + overlap, 1.0)

    # Penalise very short responses
    if len(response.split()) < 10:
        score *= 0.6

    score = round(score, 3)
    print(f"[Critic] Confidence score: {score}")
    state["confidence"] = score
    state["iterations"] = state.get("iterations", 0) + 1
    return state


def should_retry(state: AgentState) -> str:
    """Conditional edge: retry if confidence too low and under iteration limit."""
    if state.get("confidence", 0) < 0.7 and state.get("iterations", 0) < 3:
        print("[Critic] Score below threshold — re-routing to planner")
        return "retry"
    return "approved"
