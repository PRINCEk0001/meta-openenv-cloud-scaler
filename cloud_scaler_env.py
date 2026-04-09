"""
CloudScalerEnv: A resource management RL environment.
Goal: Minimize operational cost while maintaining latency < 50ms.
Trade-off: Cost of active servers vs. Penalty for high latency.

Gymnasium API compliant (v0.26+):
  - reset() -> (obs: np.ndarray, info: dict)
  - step(action) -> (obs, reward, terminated, truncated, info)
  - observation_space: Box([traffic, servers, latency])
  - action_space: Discrete(3)  — 0=hold, 1=add, 2=remove
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
MAX_STEPS        = 50      # truncated after this many steps (matches openenv.yaml)

# Obs-space bounds  [traffic,  servers,  latency_ms]
OBS_LOW  = np.array([   0.0,       1.0,        0.0], dtype=np.float32)
OBS_HIGH = np.array([1000.0,      50.0,     2000.0], dtype=np.float32)
# ─────────────────────────────────────────────────────────────────────────────


class CloudScalerEnv(gym.Env):
    """
    CloudScalerEnv: A resource management RL environment.
    Goal: Minimize operational cost while maintaining latency < 50ms.
    Trade-off: Cost of active servers vs. Penalty for high latency.

    Observation (Box, float32):
        [0] current_traffic_load  — requests/sec, range [0, 1000]
        [1] active_servers        — integer count, range [1, 50]
        [2] latency_ms            — milliseconds,  range [0, 2000]

    Actions (Discrete(3)):
        0 — hold     (no change)
        1 — add      (provision +1 server)
        2 — remove   (deprovision -1 server, floor = 1)

    Reward — strictly in open interval (0, 1), never 0.0 or 1.0:
        0.97  — latency < 50 ms  (excellent)
        0.60  — latency < 150 ms (degraded)
        0.30  — latency < 500 ms (bad)
        0.02  — latency >= 500ms (critical outage, base before penalty)
        efficiency_penalty: up to -0.20 for over-provisioning
        hard clamp: max(0.01, min(0.99, raw))

    Episode ends:
        terminated = False (no natural terminal state in this env)
        truncated  = True  after MAX_STEPS (200) steps
    """

    metadata = {"render_modes": []}
    reward_range = (0.0, 1.0)

    def __init__(self, task: str = "autoscaling_easy", render_mode=None):
        super().__init__()

        # Gymnasium required spaces ──────────────────────────────────────────
        self.observation_space = spaces.Box(
            low=OBS_LOW, high=OBS_HIGH, dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)          # {0, 1, 2}

        # Config ─────────────────────────────────────────────────────────────
        self.task         = task
        self.render_mode  = render_mode

        # Internal state ─────────────────────────────────────────────────────
        self._active_servers = MIN_SERVERS
        self._step_count     = 0
        self._total_reward   = 0.0
        self._done           = False

    # ── Private helpers ───────────────────────────────────────────────────────

    def _generate_traffic(self, step: int) -> float:
        """Sinusoidal base traffic + optional random spikes, tuned per task."""
        if self.task == "autoscaling_hard":
            centre, amp, spike_prob = 600.0, 400.0, 0.4
            spike_lo, spike_hi      = 300.0, 600.0
        elif self.task == "autoscaling_medium":
            centre, amp, spike_prob = 500.0, 250.0, 0.2
            spike_lo, spike_hi      = 200.0, 400.0
        else:                                           # easy
            centre, amp, spike_prob = 400.0, 150.0, 0.0
            spike_lo, spike_hi      = 0.0,   0.0

        base  = centre + amp * math.sin(step * 0.1)
        spike = 0.0
        if step % 5 == 0 and random.random() < spike_prob:
            spike = random.uniform(spike_lo, spike_hi)

        return float(np.clip(base + spike, 0.0, 1000.0))

    def _calculate_latency(self, traffic: float, servers: int) -> float:
        """Piecewise latency model; capped at 2000 ms for obs-space compliance."""
        if servers == 0:
            return 2000.0

        util = traffic / (servers * SERVER_CAPACITY)

        if util < 0.7:
            raw = 20.0 + (util / 0.7) * 20.0
        elif util < 0.9:
            raw = 50.0 + ((util - 0.7) / 0.2) * 100.0
        elif util < 1.0:
            raw = 200.0 + ((util - 0.9) / 0.1) * 200.0
        else:
            raw = 600.0 + ((util - 1.0) / 0.5) * 400.0

        return float(np.clip(raw, 0.0, 2000.0))

    def _calculate_reward(self, latency: float, servers: int) -> float:
        """
        Step reward strictly in the open interval (0, 1) — never 0.0 or 1.0.

        Base scores (before efficiency penalty):
        • latency < 50 ms  → base = 0.97  (excellent)
        • latency < 150 ms → base = 0.60  (degraded)
        • latency < 500 ms → base = 0.30  (bad)
        • latency >= 500ms → base = 0.02  (critical outage)

        efficiency_penalty: (servers / MAX_SERVERS) * 0.20
        Hard clamp: max(0.01, min(0.99, raw))
        """
        if latency >= 500.0:
            base = 0.02                                 # critical outage
            efficiency_penalty = 0.0                    # don't penalize cost during outage
        elif latency < 50.0:
            base = 0.97                                 # excellent
            efficiency_penalty = (servers / MAX_SERVERS) * 0.20
        elif latency < 150.0:
            base = 0.60                                 # degraded
            efficiency_penalty = (servers / MAX_SERVERS) * 0.20
        else:
            base = 0.30                                 # bad
            efficiency_penalty = (servers / MAX_SERVERS) * 0.20

        raw = base - efficiency_penalty
        # Hard clamp: guarantee strictly open (0, 1)
        return float(round(max(0.001, min(0.999, raw)), 4))

    def _make_obs(self, traffic: float, latency: float) -> np.ndarray:
        """Pack state into a float32 numpy array matching observation_space."""
        obs = np.array(
            [traffic, float(self._active_servers), latency], dtype=np.float32
        )
        return np.clip(obs, OBS_LOW, OBS_HIGH)

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        """
        Reset the environment to the initial state.

        Returns
        -------
        obs  : np.ndarray  shape (3,)
        info : dict        episode metadata
        """
        super().reset(seed=seed)

        self._active_servers = 10           # sensible warm-start
        self._step_count     = 0
        self._total_reward   = 0.0
        self._done           = False

        traffic = self._generate_traffic(0)
        latency = self._calculate_latency(traffic, self._active_servers)
        obs     = self._make_obs(traffic, latency)

        info = {
            # ── Meta grader required keys ────────────────────────────────
            "is_success"     : False,
            "latency_ms"     : float(latency),
            "active_servers" : self._active_servers,
            "step_count"     : 0,
            # ── Extra context ────────────────────────────────────────────
            "traffic"        : float(traffic),
            "total_reward"   : 0.0,
        }
        return obs, info

    def step(self, action: int):
        """
        Take one step in the environment.

        Parameters
        ----------
        action : int  — 0 (hold), 1 (add server), 2 (remove server)

        Returns
        -------
        obs        : np.ndarray  shape (3,)
        reward     : float       in [-1.0, +1.0]
        terminated : bool        always False (no natural terminal state)
        truncated  : bool        True after MAX_STEPS steps
        info       : dict        {is_success, latency_ms, active_servers, step_count, ...}
        """
        # ── Apply action ─────────────────────────────────────────────────────
        if int(action) == 1:
            self._active_servers = min(self._active_servers + 1, MAX_SERVERS)
        elif int(action) == 2:
            self._active_servers = max(self._active_servers - 1, MIN_SERVERS)
        # 0 → hold; also catches any unexpected value gracefully

        self._step_count += 1

        # ── Simulate ─────────────────────────────────────────────────────────
        traffic = self._generate_traffic(self._step_count)
        latency = self._calculate_latency(traffic, self._active_servers)
        reward  = self._calculate_reward(latency, self._active_servers)

        self._total_reward += reward

        obs = self._make_obs(traffic, latency)

        # ── Termination / truncation ──────────────────────────────────────────
        terminated = False                              # no natural end
        truncated  = self._step_count >= MAX_STEPS      # hard horizon

        # ── Info dict — keys required by Meta grader ─────────────────────────
        is_success = (latency < 50.0)
        info = {
            # Required by Meta/OpenEnv grader (Step 3 spec)
            "is_success"     : bool(is_success),
            "latency_ms"     : float(latency),
            "active_servers" : int(self._active_servers),
            "step_count"     : int(self._step_count),
            # Supplementary diagnostics
            "traffic"        : float(traffic),
            "total_reward"   : float(self._total_reward),
        }

        return obs, float(reward), terminated, truncated, info

    def render(self):
        """Text render for debugging."""
        print(
            f"[step={self._step_count:3d}] "
            f"servers={self._active_servers:2d} | "
            f"reward_so_far={self._total_reward:.2f}"
        )

    def close(self):
        pass
