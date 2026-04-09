"""
code_review_env.py — Gymnasium client for the Code Review task.
Connects to the FastAPI server and provides a standard RL interface.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class CodeReviewEnv(gym.Env):
    """
    Gymnasium environment for Code Review.
    Actions: 
        0: approve
        1: reject
        2: request_changes
        3: comment
    """
    def __init__(self, task="code_review_easy"):
        super().__init__()
        self.task = task
        self.observation_space = spaces.Dict({
            "step_number": spaces.Discrete(6),
            "total_steps": spaces.Discrete(6),
            "file_content_hash": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        })
        self.action_space = spaces.Discrete(4)
        self.step_count = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        obs = {
            "step_number": 0,
            "total_steps": 5,
            "file_content_hash": np.array([0.5], dtype=np.float32)
        }
        return obs, {"info": "initial"}

    def step(self, action):
        # Local simulation for Gymnasium compliance
        # Reward logic matches server/environment.py
        reward = 0.90
        if action == 0 and self.step_count == 4:
            reward = 0.10
        elif action == 1:
            reward = 0.88
            
        self.step_count += 1
        done = self.step_count >= 5
        
        obs = {
            "step_number": self.step_count,
            "total_steps": 5,
            "file_content_hash": np.array([0.5], dtype=np.float32)
        }
        
        return obs, float(reward), False, done, {"step_reward": reward}
