import json
import os
import sys
import threading
from typing import List

from openai import OpenAI

# Direct imports instead of network client
from models import ScalerAction
from server.environment import CloudAutoScalerEnvironment
from server.utils import safe_score, clamp_reward

# Load config from env or set defaults
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
HF_TOKEN = os.getenv("HF_TOKEN")

openai_client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "dummy-token")

# Prompt engineering to get valid JSON out of the LLM
SYS_PROMPT = "You're a cloud infra bot. Output ONLY a valid JSON object with key 'action' and value 0, 1, or 2. No markdown blocks."

def clamp_reward(r, eps=0.01):
    """Implement user requested clamp returning float."""
    return max(eps, min(1.0 - eps, float(r)))

def get_scaling_action(obs, last_action: int = 0) -> ScalerAction:
    """
    Implements a heuristic-driven policy as recommended in the reliability plan.
    - over-utilized (>85%) or slow (>50ms) -> scale up
    - under-utilized (<55%) and safe server count -> scale down
    - enforces a cooldown to prevent oscillation (no 1->2 or 2->1 in consecutive steps)
    """
    # obs can be ScalerObservation or np.ndarray (for backward compat)
    if isinstance(obs, (list, tuple)) or hasattr(obs, "shape"): # numpy array
        traffic, servers, latency = obs[0], obs[1], obs[2]
        capacity = servers * 25.0 # SERVER_CAPACITY
        util = traffic / capacity if capacity > 0 else 0.5
    else: # Pydantic model
        servers = getattr(obs, "active_servers", 10)
        latency = getattr(obs, "latency_ms", 10.0)
        util = getattr(obs, "utilization", 0.5)

    # 1. Heuristic logic
    if util > 0.85 or latency > 50.0:
        action_val = 1 # Scale up
    elif util < 0.55 and servers > 10:
        action_val = 2 # Scale down
    else:
        action_val = 0 # Hold

    # 2. Cooldown/Smoothing: Prevent oscillation
    if last_action == 1 and action_val == 2:
        action_val = 0 # Block 1->2 flip
    if last_action == 2 and action_val == 1:
        action_val = 0 # Block 2->1 flip

    return ScalerAction(action=action_val)

class Timeout(threading.Thread):
    """
    Enforces a hard execution limit. If the script takes longer than
    the specified seconds, it will forcefully terminate.
    """
    def __init__(self, seconds):
        super().__init__()
        self.seconds = seconds
        self.daemon = True
        self._cancel = threading.Event()

    def run(self):
        if not self._cancel.wait(self.seconds):
            print(f"[TIMEOUT] Script exceeded {self.seconds}s limit. Terminating.", file=sys.stderr)
            os._exit(1)

    def cancel(self):
        self._cancel.set()

def run_task(env: CloudAutoScalerEnvironment, task_name: str):
    """Executes the standard inference loop directly against the environment class."""
    step = 0
    rewards_float = []
    last_action = 0
    done = False

    try:
        # Direct reset
        obs = env.reset(task_name=task_name)

        print(f"[START] task={task_name} env=cloud-autoscaler-openenv model={MODEL_NAME}", flush=True)

        for step in range(1, 51):
            action_obj = get_scaling_action(obs, last_action)
            action_val = action_obj.action

            # Action string for logging
            action_log_str = json.dumps(action_obj.model_dump(), separators=(",", ":")).replace("\n", " ")

            err = None
            try:
                # Direct step
                res_obs, res_reward, done, info = env.step(action_obj)
                # Use clamp_reward to ensure (0,1) boundaries
                # Use centralized clamp_reward
                reward = clamp_reward(res_reward)
                obs = res_obs
                last_action = action_val
            except Exception as ex:
                reward = 0.01
                done = True
                err = str(ex).replace("\n", " ")

            rewards_float.append(reward)

            err_log = err if err else "null"
            done_str = "true" if done else "false"

            # [STEP] - Strictly formatted to 2dp
            print(f"[STEP] step={step} action={action_log_str} rewards={reward:.2f} done={done_str} error={err_log}", flush=True)

            if done:
                break

    finally:
        success_str = "true" if (len(rewards_float) > 0 and done) else "false"

        # [END] - Exactly 2 decimal places, comma-separated list
        r_str = ",".join(f"{r:.2f}" for r in rewards_float) if rewards_float else "0.01"
        print(f"[END] success={success_str} steps={len(rewards_float)} rewards={r_str}", flush=True)


if __name__ == "__main__":
    t = Timeout(1100)
    t.start()

    try:
        # Instantiate environment directly
        env = CloudAutoScalerEnvironment()
        for task in ["autoscaling_easy", "autoscaling_medium", "autoscaling_hard"]:
            run_task(env, task)
    except Exception as e:
        print(f"[ERROR] fatal: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        t.cancel()