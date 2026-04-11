"""
Utility functions for scoring and clamping in the Cloud AutoScaler environment.
Ensures consistency across inference, environment, and server logging.
Compliant with Phase 2: NaN/Inf Protection and 0.01/0.99 Clipping.
"""

import math

def safe_score(raw) -> str:
    """
    Implement the strict [0.01, 0.99] safety clamp and 2dp formatting.
    Ensures values like 0.0 or 1.0 are never returned.
    """
    try:
        if raw is None:
            val = 0.01
        else:
            val = float(raw)
            
        # Phase 2: NaN/Inf Protection
        if not math.isfinite(val):
            val = 0.01
    except (ValueError, TypeError):
        val = 0.01
    
    # Phase 2: The "0.01/0.99" Clip
    clamped = min(max(val, 0.01), 0.99)
    return f"{clamped:.2f}"

def clamp_reward(r, eps=0.01) -> float:
    """
    Returns the numeric clamped value as a float.
    Handles NaN/Inf gracefully by defaulting to 0.01.
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
        
    return min(max(val, eps), 1.0 - eps)
