"""
Utility functions for scoring and clamping in the Cloud AutoScaler environment.
Ensures consistency across inference, environment, and server logging.
MATCHES USER SUCCESS SNIPPET (Phase 2 Fixed).
"""

def safe_score(raw) -> str:
    """
    Implement the strict (0.01, 0.99) safety clamp and 2dp formatting.
    Ensures values like 0.0 or 1.0 are never returned, which is required
    by the Meta OpenEnv automated evaluator.
    Returns a string for logging/JSON.
    """
    try:
        val = float(raw if raw is not None else 0.01)
    except (ValueError, TypeError):
        val = 0.01
    
    # Strict clamping
    clamped = max(0.01, min(0.99, val))
    return f"{clamped:.2f}"

def clamp_reward(r, eps=0.01) -> float:
    """
    Returns the numeric clamped value as a float.
    Useful for internal calculations before string formatting.
    """
    try:
        val = float(r)
    except (ValueError, TypeError):
        val = eps
    return max(eps, min(1.0 - eps, val))
