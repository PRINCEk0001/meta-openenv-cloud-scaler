"""
server/tasks.py — Logic for grading task performance in the Cloud AutoScaler environment.
Ensures all scores are strictly within (0.01, 0.99).
"""

import math
from server.utils import safe_score


def normalize_score(raw_score: float) -> float:
    """Clamp score strictly away from (0, 1) boundaries using [0.01, 0.99]."""
    return float(safe_score(raw_score))

def _calculate_score_logic(state) -> float:
    """
    Implements the user's weighted rubric:
    - 40% closeness to target utilization (70%)
    - 25% response stability (latency consistency)
    - 20% cost efficiency (inverse server count)
    - 15% action smoothness (prevention of oscillation)
    """
    if not state or not getattr(state, "latency_history", []):
        return 0.1

    # 1. Utilization Score (40%) - Target 0.7 (70%)
    hist = getattr(state, "utilization_history", [])
    if not hist:
        util_score = 0.5
    else:
        avg_util = sum(hist) / len(hist)
        # 1.0 if exactly 0.7, linear drop-off
        util_score = max(0.0, 1.0 - abs(avg_util - 0.7) / 0.7)

    # 2. Stability Score (25%) - Low variance in latency
    avg_lat = state.avg_latency
    if len(state.latency_history) > 1:
        variance = sum((x - avg_lat) ** 2 for x in state.latency_history) / len(state.latency_history)
        # 0.99 if variance is 0, drops off. 100ms standard deviation is 10000 variance.
        stability_score = max(0.01, 0.99 - math.sqrt(variance) / 200.0)
    else:
        stability_score = 0.99

    # 3. Cost Score (20%) - Minimize servers (MAX_SERVERS=50, MIN_SERVERS=1)
    s_hist = getattr(state, "server_history", [])
    if not s_hist:
        avg_servers = 10
    else:
        avg_servers = sum(s_hist) / len(s_hist)
    # 0.99 if 1 server, 0.01 if 50 servers
    cost_score = max(0.01, 0.99 - (avg_servers - 1) / 49)

    # 4. Smoothness Score (15%) - Low action oscillation
    # action_history contains 0:hold, 1:add, 2:remove
    changes = 0
    for i in range(1, len(state.action_history)):
        if state.action_history[i] != state.action_history[i-1] and state.action_history[i] != 0:
            changes += 1
    # 0.99 if 0 changes, 0.01 if 25 changes (every other step)
    smoothness_score = max(0.01, 0.99 - (changes / 25.0))

    # Weighted Sum
    raw = (0.40 * util_score + 
           0.25 * stability_score + 
           0.20 * cost_score + 
           0.15 * smoothness_score)

    return raw

def grade_task_easy(state) -> float:
    """Grader for the easy auto-scaling task."""
    raw_score = _calculate_score_logic(state)
    return normalize_score(raw_score)

def grade_task_medium(state) -> float:
    """Grader for the medium auto-scaling task."""
    raw_score = _calculate_score_logic(state)
    return normalize_score(raw_score)

def grade_task_hard(state) -> float:
    """Grader for the hard auto-scaling task."""
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
