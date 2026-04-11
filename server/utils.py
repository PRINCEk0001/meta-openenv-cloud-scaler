"""
Utility functions for scoring and clamping in the Cloud AutoScaler environment.
Ensures consistency across inference, environment, and server logging.
REQUIRED FOR PHASE 2: Absolute Bulletproof Clamping (Submission #26 Refined).
"""

import math

def safe_score(value):
    """
    REQUIRED FOR PHASE 2:
    Ensures scores are strictly > 0 and < 1.
    This is the "Absolute Fix" for the Meta/Hugging Face hackathon validator.
    """
    try:
        # Handle None/NaN/Inf
        if value is None or not math.isfinite(float(value)):
            return 0.01
        
        val = float(value)
        
        # Mandatory strict clipping for hackathon validator
        if val >= 1.0:
            return 0.99
        if val <= 0.0:
            return 0.01
            
        return val
    except Exception:
        return 0.01

def clamp_reward(reward):
    """Alias for safe_score to maintain backward compatibility."""
    return safe_score(reward)
