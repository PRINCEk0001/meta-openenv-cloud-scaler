"""
Utility functions for scoring and clamping in the Cloud AutoScaler environment.
Ensures consistency across inference, environment, and server logging.
Includes bulletproof handling for NaN, Inf, and None values.
"""

import math

def safe_score(raw) -> str:
    """
    Implement the strict (0.01, 0.99) safety clamp and 2dp formatting.
    Guaranteed to return a finite numeric string.
    """
    try:
        if raw is None:
            val = 0.01
        else:
            val = float(raw)
            
        # Bulletproof NaN/Inf filtering
        if not math.isfinite(val):
            val = 0.01
    except (ValueError, TypeError):
        val = 0.01
    
    # Strict clamping to (0, 1) exclusive range
    clamped = max(0.01, min(0.99, val))
    return f"{clamped:.2f}"

def clamp_reward(r, eps=0.01) -> float:
    """
    Returns the numeric clamped value as a float.
    Handles NaN/Inf gracefully by defaulting to eps.
    """
    try:
        if r is None:
            val = eps
        else:
            val = float(r)
            
        if not math.isfinite(val):
            val = eps
    except (ValueError, TypeError):
        val = eps
        
    return max(eps, min(1.0 - eps, val))
