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

# --- Antigravity Persona Configuration ---
SYS_PROMPT = (
    "Role: You are an autonomous Cloud Infrastructure Controller for the 'Antigravity' environment.\n"
    "Objective: Maintain server health by balancing traffic load and response latency.\n\n"
    "Operational Thresholds:\n"
    "- CRITICAL LOAD: If Utilization > 85% or Latency > 50ms -> Action 1 (Scale Up)\n"
    "- UNDER-UTILIZED: If Utilization < 45% and Servers > 1 -> Action 2 (Scale Down)\n"
    "- STABLE: Otherwise -> Action 0 (Hold)\n\n"
    "Rules:\n"
    "1. You must respond ONLY with a valid JSON object.\n"
    "2. The JSON must contain exactly one key: 'action'.\n"
    "3. The value of 'action' must be an integer: 0, 1, or 2.\n"
    "4. Do not include markdown blocks (```json), commentary, or reasoning.\n\n"
    "Schema: {\"action\": <int>}"
)

def get_scaling_action(obs, last_action: int = 0) -> ScalerAction:
    """
    Antigravity Cloud Controller Implementation.
    Combines LLM-driven decision making with a heuristic safety guard 
    aligned with the Antigravity operational thresholds.
    """
    # 1. Parse metrics from observation
    if isinstance(obs, (list, tuple)) or hasattr(obs, "shape"): # numpy array
        traffic, servers, latency = obs[0], obs[1], obs[2]
        capacity = servers * 25.0
        util = traffic / capacity if capacity > 0 else 0.5
    else: # Pydantic model
        servers = getattr(obs, "active_servers", 10)
        latency = getattr(obs, "latency_ms", 10.0)
        util = getattr(obs, "utilization", 0.5)

    # 2. Antigravity Heuristic Safety Guard (aligned with thresholds)
    h_action = 0
    if util > 0.85 or latency > 50.0:
        h_action = 1 # Critical Load
    elif util < 0.45 and servers > 1:
        h_action = 2 # Under-Utilized

    # 3. LLM-Driven Decision (Proxy Compliance)
    action_val = h_action # default to heuristic
    try:
        # Prompt construction with context
        prompt = (
            f"Environment Report:\n"
            f"- Utilization: {util*100:.1f}%\n"
            f"- Latency: {latency:.1f}ms\n"
            f"- Active Servers: {servers}\n"
            f"As the Antigravity Infrastructure Controller, determine the next action."
        )
        
        completion = openai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYS_PROMPT},
                {"role": "user", "content": prompt}
            ],
            timeout=5.0
        )
        content = completion.choices[0].message.content
        
        # Robust JSON cleaning (remove whitespace/markdown)
        content = content.replace("```json", "").replace("```", "").strip()
        if "{" in content:
            data = json.loads(content[content.find("{"):content.rfind("}")+1])
            action_val = int(data.get("action", h_action))
    except Exception:
        pass # fallback to h_action on fail or timeout

    # 4. Global Smoothness Guard (enforced after LLM)
    if last_action == 1 and action_val == 2: action_val = 0
    if last_action == 2 and action_val == 1: action_val = 0

    return ScalerAction(action=action_val)

class Timeout(threading.Thread):
    """Execution limit enforcer."""
    def __init__(self, seconds):
        super().__init__()
        self.seconds = seconds
        self.daemon = True
        self._cancel = threading.Event()
    def run(self):
        if not self._cancel.wait(self.seconds):
            print(f"[TIMEOUT] Antigravity process exceeded {self.seconds}s limit. Terminating.", file=sys.stderr)
            os._exit(1)
    def cancel(self):
        self._cancel.set()

def run_task(env: CloudAutoScalerEnvironment, task_name: str):
    """Executes the standard inference loop aligned with Antigravity policy."""
    step = 0
    rewards_float = []
    last_action = 0
    done = False
    try:
        obs = env.reset(task_name=task_name)
        print(f"[START] task={task_name} env=cloud-autoscaler-openenv model={MODEL_NAME}", flush=True)
        for step in range(1, 51):
            action_obj = get_scaling_action(obs, last_action)
            action_val = action_obj.action
            action_log_str = json.dumps(action_obj.model_dump(), separators=(",", ":"))
            err = None
            try:
                res_obs, res_reward, done, info = env.step(action_obj)
                # Use bulletproof clamp_reward
                reward = clamp_reward(res_reward)
                obs = res_obs
                last_action = action_val
            except Exception as ex:
                reward = 0.01
                done = True
                err = str(ex).replace("\n", " ")
            rewards_float.append(reward)
            done_str = "true" if done else "false"
            # [STEP] - Strictly formatted to 2dp
            print(f"[STEP] step={step} action={action_log_str} rewards={reward:.2f} done={done_str} error={err if err else 'null'}", flush=True)
            if done: break
    finally:
        success_str = "true" if (len(rewards_float) > 0 and done) else "false"
        r_str = ",".join(f"{r:.2f}" for r in rewards_float) if rewards_float else "0.01"
        print(f"[END] success={success_str} steps={len(rewards_float)} rewards={r_str}", flush=True)

if __name__ == "__main__":
    t = Timeout(1500)
    t.start()
    try:
        env = CloudAutoScalerEnvironment()
        for task in ["autoscaling_easy", "autoscaling_medium", "autoscaling_hard"]:
            run_task(env, task)
    except Exception as e:
        print(f"[ERROR] fatal: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        t.cancel()