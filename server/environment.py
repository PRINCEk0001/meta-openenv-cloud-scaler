"""
Simulation core for the Cloud AutoScaler environment.
This handles the logic for traffic spikes, latency scaling, and reward calculation.
"""

import math
import random
from typing import Tuple
import sys
import os

from pydantic import BaseModel

try:
    from openenv.core.environment import Environment as _BaseEnvironment
except ImportError:
    # fallback if not installed yet
    _BaseEnvironment = BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from models import ScalerObservation, ScalerAction, ScalerState, CodeReviewObservation, CodeReviewAction, CodeReviewState


MAX_STEPS = 50
SERVER_CAPACITY = 25
MIN_SERVERS = 1
MAX_SERVERS = 50

def safe_score(raw):
    """Implement the strict [0.01, 0.99] safety clamp and 2dp formatting."""
    clamped = max(0.01, min(0.99, float(raw or 0.01)))
    return f"{clamped:.2f}"

class CloudAutoScalerEnvironment(_BaseEnvironment):
    """
    Cloud server auto-scaling RL environment.
    Balances performance vs cost vs resilience.
    """

    def __init__(self):
        try:
            super().__init__()
        except TypeError:
            pass

        self._state = None
        self._active_servers = 10
        self._step_count = 0
        self._traffic_history = []
        self._done = False
        self._task_name = "autoscaling_easy"

    def _generate_traffic(self, step: int) -> float:
        # Task variables
        if self._task_name == "autoscaling_hard":
            wave_centre = 600.0
            wave_amp = 400.0
            spike_prob = 0.4
            spike_min = 300.0
            spike_max = 600.0
        elif self._task_name == "autoscaling_medium":
            wave_centre = 500.0
            wave_amp = 250.0
            spike_prob = 0.2
            spike_min = 200.0
            spike_max = 400.0
        else: # easy
            wave_centre = 400.0
            wave_amp = 150.0
            spike_prob = 0.0 # no spikes
            spike_min = 0.0
            spike_max = 0.0

        base = wave_centre + wave_amp * math.sin(step * 0.1)
        spike = 0.0
        
        # chance to spike every 5 steps
        if step % 5 == 0 and random.random() < spike_prob:
            spike = random.uniform(spike_min, spike_max)

        return float(max(0.0, min(1000.0, base + spike)))

    def _calculate_latency(self, traffic: float, servers: int) -> float:
        if servers == 0:
            return 1000.0  # instant crash if no servers left

        capacity = servers * SERVER_CAPACITY
        utilization = traffic / capacity

        # Deterministic linear interpolation based on utilization
        if utilization < 0.7:
            return 20.0 + (utilization / 0.7) * 20.0
        elif utilization < 0.9:
            return 50.0 + ((utilization - 0.7) / 0.2) * 100.0
        elif utilization < 1.0:
            return 200.0 + ((utilization - 0.9) / 0.1) * 200.0
        
        # overloaded
        raw = 600.0 + ((utilization - 1.0) / 0.5) * 400.0
        return float(max(0.0, min(2000.0, raw)))

    def _calculate_reward(self, latency: float, servers: int) -> float:
        # Calculate a step score strictly in the open interval (0, 1).
        # The grader requires: 0.0 < score < 1.0 (exclusive on both ends).
        #
        # Thresholds:
        #   latency >= 500ms  -> near-zero score (crash / severe outage)
        #   latency <  50ms   -> near-perfect score
        #   otherwise         -> moderate score
        #
        # An efficiency penalty is subtracted based on server count.

        if latency >= 500.0:
            # Critical outage — very low score but strictly > 0
            base_score = 0.05
            efficiency_penalty = 0.01
        elif latency < 50.0:
            # Excellent performance
            base_score = 0.97
            efficiency_penalty = (servers / MAX_SERVERS) * 0.20
        elif latency < 150.0:
            # Acceptable performance
            base_score = 0.60
            efficiency_penalty = (servers / MAX_SERVERS) * 0.20
        else:
            # Poor performance (high latency, not yet crashing)
            base_score = 0.30
            efficiency_penalty = (servers / MAX_SERVERS) * 0.20

        raw = base_score - efficiency_penalty
        return safe_score(raw)

    def reset(self, task_name: str = "autoscaling_easy") -> ScalerObservation:
        self._task_name = task_name
        self._active_servers = 10
        self._step_count = 0
        self._traffic_history = []
        self._done = False

        traffic = self._generate_traffic(0)
        capacity = float(self._active_servers * SERVER_CAPACITY)
        
        # shouldn't be 0 on reset but checking just in case
        utilization = traffic / capacity if capacity > 0 else 0.0
        latency = self._calculate_latency(traffic, self._active_servers)

        self._state = ScalerState(
            episode_id=f"ep_{random.randint(1000, 9999)}",
            step_count=0,
            total_reward=0.1,
            peak_traffic=traffic,
            avg_latency=latency,
        )

        return ScalerObservation(
            current_traffic_load=round(traffic, 2),
            active_servers=self._active_servers,
            latency_ms=round(latency, 2),
            step_number=0,
            total_capacity=round(capacity, 2),
            utilization=round(utilization, 3),
        )

    def step(self, action: ScalerAction) -> Tuple[ScalerObservation, float, bool, dict]:
        if self._done:
            raise RuntimeError("Episode already completed, call reset() first.")

        # action 0: hold
        # action 1: add
        # action 2: remove
        if action.action == 1:
            self._active_servers = min(self._active_servers + 1, MAX_SERVERS)
        elif action.action == 2:
            self._active_servers = max(self._active_servers - 1, MIN_SERVERS)

        self._step_count += 1
        traffic = self._generate_traffic(self._step_count)
        self._traffic_history.append(traffic)

        capacity = float(self._active_servers * SERVER_CAPACITY)
        utilization = traffic / capacity if capacity > 0 else 10.0
        
        latency = self._calculate_latency(traffic, self._active_servers)
        reward = self._calculate_reward(latency, self._active_servers)

        # update internal state trackers
        self._state.step_count = self._step_count
        self._state.total_reward += float(reward)
        self._state.peak_traffic = max(self._state.peak_traffic, traffic)
        
        # running average
        self._state.avg_latency = (
            (self._state.avg_latency * (self._step_count - 1) + latency)
            / self._step_count
        )

        self._done = self._step_count >= MAX_STEPS

        obs = ScalerObservation(
            current_traffic_load=round(traffic, 2),
            active_servers=self._active_servers,
            latency_ms=round(latency, 2),
            step_number=self._step_count,
            total_capacity=round(capacity, 2),
            utilization=round(utilization, 3),
        )

        info = {
            "error": None,
            "is_success": bool(latency < 50.0),
            "latency_ms": float(latency),
            "active_servers": int(self._active_servers),
            "step_count": int(self._step_count),
            "episode_id": self._state.episode_id,
            "total_raw_reward": round(self._state.total_reward, 3),
            "avg_latency": round(self._state.avg_latency, 2),
            "peak_traffic": round(self._state.peak_traffic, 2),
        }

        return obs, round(reward, 2), self._done, info

    @property
    def current_step(self) -> int:
        return self._step_count

    @property
    def active_servers(self) -> int:
        return self._active_servers

    @property
    def is_done(self) -> bool:
        return self._done

    @property
    def episode_return(self) -> float:
        return round(self._state.total_reward, 2) if self._state else 0.0

class CodeReviewEnvironment:
    """
    Server-side implementation of the CodeReview environment.
    Manages 5-step episodes and trajectory rewards.
    """
    def __init__(self):
        self._state = None
        self._difficulty = "easy"

    def reset(self, task_name: str = "code_review_easy") -> CodeReviewObservation:
        self._state = CodeReviewState(
            episode_id=f"cr_{random.randint(1000, 9999)}",
            step_count=0,
            step_rewards=[]
        )
        self._difficulty = "easy"
        if "hard" in task_name: self._difficulty = "hard"
        elif "medium" in task_name: self._difficulty = "medium"
        
        return self._make_obs()

    def _make_obs(self) -> CodeReviewObservation:
        return CodeReviewObservation(
            file_content="def solve():\n    return 42",
            diff_summary="Added a basic function.",
            step_number=self._state.step_count,
            total_steps=5
        )

    def step(self, action: CodeReviewAction) -> Tuple[CodeReviewObservation, float, bool, dict]:
        if not self._state:
            self.reset()
            
        # Determine reward based on the rubric constants provided by the user
        reward = 0.90 # Perfect
        if action.action_type == "approve" and self._state.step_count == 4:
            reward = 0.10 # Catastrophic
        elif action.action_type == "reject":
            reward = 0.88 # Near-Perfect
            
        self._state.step_count += 1
        self._state.step_rewards.append(reward)
        
        done = (self._state.step_count >= 5)
        obs = self._make_obs()
        
        info = {
            "is_success": reward >= 0.70,
            "step_reward": safe_score(reward)
        }
        
        return obs, float(safe_score(reward)), done, info

    @property
    def is_done(self) -> bool:
        return self._state.step_count >= 5 if self._state else True
