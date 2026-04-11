"""
server/environment.py — Simulation logic for Cloud AutoScaler and CodeReview tasks.
Matches user 'Winning Snippet' with local safe_score for maximum compliance.
"""

import math
import random
import sys
import os
from typing import Tuple, List, Any
from pydantic import BaseModel
from .models import ScalerAction, CodeReviewAction, ScalerObservation, CodeReviewObservation, ScalerState, CodeReviewState
from server.utils import clamp_reward

# Thresholds
SERVER_CAPACITY = 25
MIN_SERVERS = 1
MAX_SERVERS = 50
MAX_STEPS = 50

def safe_score(raw) -> str:
    """Implement the strict [0.01, 0.99] safety clamp and 2dp formatting locally for Phase 2."""
    try:
        val = float(raw if raw is not None else 0.10)
    except (ValueError, TypeError):
        val = 0.10
    clamped = max(0.01, min(0.99, val))
    return f"{clamped:.2f}"

class CloudAutoScalerEnvironment:
    def __init__(self):
        self._active_servers = 10
        self._step_count = 0
        self._done = False
        self._state = None

    def _generate_traffic(self, step: int) -> float:
        # Easy sinusoidal load
        base = 400.0 + 150.0 * math.sin(step * 0.1)
        return max(0.0, min(1000.0, base))

    def _calculate_latency(self, traffic: float, servers: int) -> float:
        if servers == 0: return 2000.0
        util = traffic / (servers * SERVER_CAPACITY)
        if util < 0.7: raw = 20.0 + (util / 0.7) * 20.0
        elif util < 0.9: raw = 50.0 + ((util - 0.7) / 0.2) * 100.0
        else: raw = 200.0 + ((util - 0.9) / 0.1) * 800.0
        return max(0.0, min(2000.0, raw))

    def _calculate_reward(self, latency: float, servers: int) -> float:
        if latency >= 500.0:
            base = 0.05
            penalty = 0.01
        elif latency < 50.0:
            base = 0.97
            penalty = (servers / MAX_SERVERS) * 0.20
        else:
            base = 0.60
            penalty = (servers / MAX_SERVERS) * 0.20
        
        raw = base - penalty
        return clamp_reward(raw)

    def reset(self, task_name: str = "autoscaling_easy") -> ScalerObservation:
        self._active_servers = 10
        self._step_count = 0
        self._done = False
        traffic = self._generate_traffic(0)
        latency = self._calculate_latency(traffic, self._active_servers)
        util = traffic / (self._active_servers * SERVER_CAPACITY)
        
        self._state = ScalerState(
            episode_id=f"ep_{random.randint(1000, 9999)}",
            step_count=0,
            total_reward=0.10,
            peak_traffic=traffic,
            avg_latency=latency,
            latency_history=[latency],
            action_history=[],
            utilization_history=[util],
            server_history=[self._active_servers],
            step_rewards=[0.10]
        )
        return ScalerObservation(
            current_traffic_load=round(traffic, 2),
            active_servers=self._active_servers,
            latency_ms=round(latency, 2),
            step_number=0,
            total_capacity=self._active_servers * SERVER_CAPACITY,
            utilization=round(util, 3)
        )

    def step(self, action: ScalerAction) -> Tuple[ScalerObservation, float, bool, dict]:
        if action.action == 1: self._active_servers = min(self._active_servers + 1, MAX_SERVERS)
        elif action.action == 2: self._active_servers = max(self._active_servers - 1, MIN_SERVERS)
        
        self._step_count += 1
        traffic = self._generate_traffic(self._step_count)
        latency = self._calculate_latency(traffic, self._active_servers)
        reward = self._calculate_reward(latency, self._active_servers)
        util = traffic / (self._active_servers * SERVER_CAPACITY)
        
        self._state.total_reward += float(reward)
        self._state.latency_history.append(latency)
        self._state.step_rewards.append(float(reward))
        self._state.step_count = self._step_count
        self._done = (self._step_count >= MAX_STEPS)
        
        obs = ScalerObservation(
            current_traffic_load=round(traffic, 2),
            active_servers=self._active_servers,
            latency_ms=round(latency, 2),
            step_number=self._step_count,
            total_capacity=self._active_servers * SERVER_CAPACITY,
            utilization=round(util, 3)
        )
        info = {"is_success": latency < 50.0, "step_count": self._step_count}
        return obs, float(safe_score(reward)), self._done, info

class CodeReviewEnvironment:
    def __init__(self):
        self._state = None
    
    def reset(self, task_name: str = "code_review_easy") -> CodeReviewObservation:
        self._state = CodeReviewState(
            episode_id=f"cr_{random.randint(1000, 9999)}",
            step_count=0,
            step_rewards=[]
        )
        return CodeReviewObservation(
            file_content="def auth(user): return True", 
            diff_summary="Add simple authentication bypass for testing.",
            step_number=0
        )
    
    def step(self, action: CodeReviewAction) -> Tuple[CodeReviewObservation, float, bool, dict]:
        self._state.step_count += 1
        # Hardcoded logic: Approve is bad, Reject is good
        reward = 0.10 if action.action_type == "approve" else 0.88
        self._state.step_rewards.append(reward)
        
        done = (self._state.step_count >= 5)
        obs = CodeReviewObservation(file_content="...", diff_summary="...", step_number=self._state.step_count)
        return obs, float(safe_score(reward)), done, {}
