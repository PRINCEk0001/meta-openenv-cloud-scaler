"""
server/code_review_logic.py — Trajectory-based grading for CodeReview tasks.
Implements multipliers, consistency bonuses, and ultra-strict clamping.
"""
import math
from typing import List

def clamp_ultra_strict(score: float) -> float:
    """Implement the [0.001, 0.999] ultra-strict safety clamp."""
    s = max(0.001, min(0.999, float(score)))
    s = round(s, 3)

    if s >= 1.0:
        s = 0.999
    if s <= 0.0:
        s = 0.001

    return float(s)

def grade_code_review_trajectory(step_rewards: List[float], difficulty: str) -> float:
    """
    Computes final task score based on a sequence of 5 step rewards.
    
    Rubric:
    - Base: mean of steps
    - Catastrophic Penalty (-0.4 to -0.6 if any step == 0.10)
    - Consistency Bonus (+0.05 to +0.15 if 4/5 steps >= 0.70)
    - Explanation Bonus (+0.05 if 4/5 steps == 0.90)
    """
    if not step_rewards:
        return 0.1
    
    # 1. Base Mean
    base_mean = sum(step_rewards) / len(step_rewards)
    final_score = base_mean
    
    # 2. Catastrophic Penalty (0.10 is the magic 'approve_vulnerability' score)
    has_catastrophe = any(abs(r - 0.10) < 0.001 for r in step_rewards)
    if has_catastrophe:
        if difficulty == "hard":
            final_score -= 0.60
        elif difficulty == "medium":
            final_score -= 0.50
        else:
            final_score -= 0.40
            
    # 3. Consistency/Explanation Bonuses (Only if no catastrophe)
    if not has_catastrophe:
        # Consistency Bonus (>= 80% steps are Partial Credit or better)
        correct_count = sum(1 for r in step_rewards if r >= 0.70)
        if correct_count >= 4:
            if difficulty == "hard":
                final_score += 0.15
            elif difficulty == "medium":
                final_score += 0.10
            else:
                final_score += 0.05
                
        # Explanation Bonus (>= 80% steps are Perfect)
        perfect_count = sum(1 for r in step_rewards if abs(r - 0.90) < 0.001)
        if perfect_count >= 4:
            final_score += 0.05
        
    # 5. Final Safety Clamp
    return clamp_ultra_strict(final_score)
