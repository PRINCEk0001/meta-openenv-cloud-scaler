"""
server/tasks.py — Logic for grading task performance in the Cloud AutoScaler environment.
Ensures all scores are strictly within (0.001, 0.999).
"""

import math

def normalize_score(raw_score: float) -> float:
    """Clamp score strictly away from (0, 1) boundaries with an ultra-safe buffer."""
    if raw_score is None or math.isnan(raw_score) or math.isinf(raw_score):
        return 0.1
    
    # 1. First clamp to [0.01, 0.99]
    score = max(0.01, min(0.99, float(raw_score)))
    
    # 2. Round to 2 decimals as suggested
    score = round(score, 2)
    
    # 3. Final nudge away from boundaries if rounding pushed it to the edge
    if score <= 0.01:
        score = 0.02
    elif score >= 0.99:
        score = 0.98
        
    return float(score)

def _calculate_score_logic(state) -> float:
    """Internal shared logic for scoring based on latency and efficiency."""
    if not state:
        return 0.1

    # 1. Latency Component (Scale: 0.05 to 0.7)
    if state.avg_latency < 50.0:
        latency_score = 0.7
    elif state.avg_latency < 500.0:
        # linear decrease from 0.7 at 50ms to 0.1 at 500ms
        latency_score = 0.7 - ((state.avg_latency - 50.0) / 450.0) * 0.6
    else:
        latency_score = 0.05

    # 2. Efficiency Component (Scale: 0.0 to 0.25)
    if state.avg_latency < 500.0:
        efficiency_score = 0.25
    else:
        efficiency_score = 0.0

    return latency_score + efficiency_score

def grade_task_easy(state) -> float:
    """Grader for the easy auto-scaling task."""
    raw_score = _calculate_score_logic(state)
    return normalize_score(raw_score)

def grade_task_medium(state) -> float:
    """Grader for the medium auto-scaling task."""
    # Could add extra penalties for medium here if needed
    raw_score = _calculate_score_logic(state)
    return normalize_score(raw_score)

def grade_task_hard(state) -> float:
    """Grader for the hard auto-scaling task."""
    # Could add extra penalties for hard here if needed
    raw_score = _calculate_score_logic(state)
    return normalize_score(raw_score)

from server.code_review_logic import grade_code_review_trajectory

def grade_task(task_name: str, state) -> float:
    """
    Dispatches to the specific grader based on task name.
    Supports both CloudAutoScaler and CodeReview environments.
    """
    task_name = task_name.lower()
    
    # --- CodeReview Environment Tasks ---
    if "codereview" in task_name or "code_review" in task_name:
        difficulty = "easy"
        if "hard" in task_name: difficulty = "hard"
        elif "medium" in task_name: difficulty = "medium"
        
        # CodeReviewState stores step_rewards as a list
        return grade_code_review_trajectory(
            getattr(state, "step_rewards", []), 
            difficulty
        )

    # --- CloudAutoScaler Environment Tasks ---
    if "hard" in task_name:
        return grade_task_hard(state)
    elif "medium" in task_name:
        return grade_task_medium(state)
    else:
        # Default to easy logic (broad match for 'easy' or anything else)
        return grade_task_easy(state)

