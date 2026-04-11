"""
Utility functions for scoring and clamping in the Cloud AutoScaler environment.
Ensures consistency across inference, environment, and server logging.
Compliant with Phase 2: "Last Mile" Ultra-Strict Hardening (Submission #26).
"""

import math

def safe_score(value):
    """
    Implements ultra-strict clamping for meta-hf hackathon.
    Guaranteed to return a float strictly within (0.01, 0.99).
    """
    try:
        # 1. Handle non-finite numbers
        if value is None:
            return 0.01
            
        val = float(value)
        
        if not math.isfinite(val):
            return 0.01
        
        # 2. STRICT CLAMPING: No 0.0, No 1.0
        # If the environment wants to give 100%, we give 99%.
        if val >= 1.0: 
            return 0.99
        if val <= 0.0: 
            return 0.01
        
        # Clamp to [0.01, 0.99] inclusive of those ends, but exclusive of 0.0/1.0
        clamped = max(0.01, min(0.99, val))
        return round(clamped, 4) # Keep precision but stay safely in range
    except Exception:
        return 0.01

def clamp_reward(reward):
    """Alias for safe_score to maintain backward compatibility."""
    return safe_score(reward)
