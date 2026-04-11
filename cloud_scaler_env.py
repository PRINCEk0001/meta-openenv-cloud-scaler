"""
CloudScalerEnv: A resource management RL environment.
Matches user 'Winning Snippet' logic with local safe_score for maximum compliance.
"""

import math
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ── Tuneable constants ────────────────────────────────────────────────────────
SERVER_CAPACITY  = 25      # req/s per server
MIN_SERVERS      = 1
MAX_SERVERS      = 50
MAX_STEPS        = 50      

# Obs-space bounds  [traffic,  servers,  latency_ms]
OBS_LOW  = np.array([   0.0,       1.0,        0.0], dtype=np.float32)
OBS_HIGH = np.array([1000.0,      50.0,     2000.0], dtype=np.float32)

def safe_score(raw):
    """Implement the strict [0.01, 0.99] safety clamp and 2dp formatting locally."""
    try:
        val = float(raw if raw is not None else 0.10)
    except (ValueError, TypeError):
        val = 0.10
    clamped = max(0.01, min(0.99, val))
    return f"{clamped:.2f}"

class CloudScalerEnv(gym.Env):
    metadata = {"render_modes": []}
    reward_range = (0.01, 0.99)

    def __init__(self, task: str = "autoscaling_easy", render_mode=None):
        super().__init__()
        self.observation_space = spaces.Box(low=OBS_LOW, high=OBS_HIGH, dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self.task = task
        self.render_mode = render_mode
        self._active_servers = 10
        self._step_count = 0
        self._total_reward = 0.10
        self._done = False

    def _generate_traffic(self, step: int) -> float:
        base = 400.0 + 150.0 * math.sin(step * 0.1)
        return float(np.clip(base, 0.0, 1000.0))

    def _calculate_latency(self, traffic: float, servers: int) -> float:
        if servers == 0: return 2000.0
        util = traffic / (servers * SERVER_CAPACITY)
        if util < 0.7: raw = 20.0 + (util / 0.7) * 20.0
        elif util < 0.9: raw = 50.0 + ((util - 0.7) / 0.2) * 100.0
        else: raw = 200.0 + ((util - 0.9) / 0.1) * 800.0
        return float(np.clip(raw, 0.0, 2000.0))

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
        return base - penalty

    def _make_obs(self, traffic: float, latency: float) -> np.ndarray:
        obs = np.array([traffic, float(self._active_servers), latency], dtype=np.float32)
        return np.clip(obs, OBS_LOW, OBS_HIGH)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._active_servers = 10
        self._step_count = 0
        self._total_reward = 0.10
        self._done = False
        traffic = self._generate_traffic(0)
        latency = self._calculate_latency(traffic, self._active_servers)
        obs = self._make_obs(traffic, latency)
        info = {
            "is_success": bool(latency < 50.0),
            "latency_ms": float(latency),
            "active_servers": self._active_servers,
            "step_count": 0,
            "traffic": float(traffic),
            "total_reward": 0.10,
        }
        return obs, info

    def step(self, action: int):
        if int(action) == 1: self._active_servers = min(self._active_servers + 1, MAX_SERVERS)
        elif int(action) == 2: self._active_servers = max(self._active_servers - 1, MIN_SERVERS)
        self._step_count += 1
        traffic = self._generate_traffic(self._step_count)
        latency = self._calculate_latency(traffic, self._active_servers)
        reward = self._calculate_reward(latency, self._active_servers)
        self._total_reward += float(reward)
        obs = self._make_obs(traffic, latency)
        info = {
            "is_success": bool(latency < 50.0),
            "latency_ms": float(latency),
            "active_servers": int(self._active_servers),
            "step_count": int(self._step_count),
            "traffic": float(traffic),
            "total_reward": float(self._total_reward),
        }
        return obs, float(safe_score(reward)), False, self._step_count >= MAX_STEPS, info
