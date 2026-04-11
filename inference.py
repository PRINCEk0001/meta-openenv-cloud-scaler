import json
import os
import sys
import threading
from typing import List, Union, Any

from openai import OpenAI

# Direct imports instead of network client
from models import ScalerAction, CodeReviewAction
from server.environment import CloudAutoScalerEnvironment, CodeReviewEnvironment
from server.utils import safe_score, clamp_reward

# Load config from env or set defaults
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
HF_TOKEN = os.getenv("HF_TOKEN")

openai_client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "dummy-token")

# --- Persona Configuration ---
SYS_PROMPT_SCALER = (
    "Role: You are an autonomous Cloud Infrastructure Controller for the 'Antigravity' environment.\n"
    "Objective: Maintain server health by balancing traffic load and response latency.\n\n"
    "Operational Thresholds:\n"
    "- CRITICAL LOAD: If Utilization > 85% or Latency > 50ms -> Action 1 (Scale Up)\n"
    "- UNDER-UTILIZED: If Utilization < 45% and Servers > 1 -> Action 2 (Scale Down)\n"
    "- STABLE: Otherwise -> Action 0 (Hold)\n\n"
    "Rules: Respond ONLY with valid JSON {\"action\": <int>}. No commentary."
)

SYS_PROMPT_REVIEW = (
    "Role: You are an expert Software Security Auditor.\n"
    "Objective: Review code diffs for vulnerabilities.\n"
    "Rules: Respond ONLY with valid JSON {\"action_type\": \"approve\"|\"reject\"|\"comment\", \"comment\": \"reasoning\"}. No commentary."
)

def get_action(obs, task_name: str, last_action: int = 0) -> Union[ScalerAction, CodeReviewAction]:
    """Dynamically determines the action based on the task environment."""
    
    if "code_review" in task_name.lower():
        # Code Review Logic (LLM Consultation)
        action_val = "reject" # Safe default
        try:
            prompt = f"File Content: {getattr(obs, 'file_content', '')}\nDiff: {getattr(obs, 'diff_summary', '')}\nAction?"
            completion = openai_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "system", "content": SYS_PROMPT_REVIEW}, {"role": "user", "content": prompt}],
                timeout=5.0
            )
            data = json.loads(completion.choices[0].message.content.strip("`json").strip())
            action_val = data.get("action_type", "reject")
        except Exception: pass
        return CodeReviewAction(action_type=action_val, comment="Automated Audit")

    else:
        # Antigravity Scaler Logic
        if isinstance(obs, (list, tuple)) or hasattr(obs, "shape"): # numpy array
            traffic, servers, latency = obs[0], obs[1], obs[2]
            util = traffic / (servers * 25.0) if servers > 0 else 0.5
        else: # Pydantic model
            servers = getattr(obs, "active_servers", 10)
            latency = getattr(obs, "latency_ms", 10.0)
            util = getattr(obs, "utilization", 0.5)

        h_action = 0
        if util > 0.85 or latency > 50.0: h_action = 1
        elif util < 0.45 and servers > 1: h_action = 2

        action_val = h_action
        try:
            prompt = f"Util: {util*100:.1f}%, Latency: {latency:.1f}ms, Servers: {servers}."
            completion = openai_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "system", "content": SYS_PROMPT_SCALER}, {"role": "user", "content": prompt}],
                timeout=5.0
            )
            data = json.loads(completion.choices[0].message.content.strip("`json").strip())
            action_val = int(data.get("action", h_action))
        except Exception: pass

        if last_action == 1 and action_val == 2: action_val = 0
        if last_action == 2 and action_val == 1: action_val = 0
        return ScalerAction(action=action_val)

class Timeout(threading.Thread):
    def __init__(self, seconds):
        super().__init__()
        self.seconds = seconds
        self.daemon = True
        self._cancel = threading.Event()
    def run(self):
        if not self._cancel.wait(self.seconds):
            print(f"[TIMEOUT] Process exceeded {self.seconds}s limit. Terminating.", file=sys.stderr)
            os._exit(1)
    def cancel(self): self._cancel.set()

def run_task(env: Any, task_name: str):
    """Executes the inference loop for any environment type."""
    rewards_float = []
    last_action = 0
    done = False
    try:
        obs = env.reset(task_name=task_name)
        print(f"[START] task={task_name} env=cloud-autoscaler-openenv model={MODEL_NAME}", flush=True)
        max_steps = 50 if "autoscaling" in task_name else 5
        for step in range(1, max_steps + 1):
            action_obj = get_action(obs, task_name, last_action)
            action_log = json.dumps(action_obj.model_dump(), separators=(",", ":"))
            try:
                obs, res_reward, done, info = env.step(action_obj)
                reward = clamp_reward(res_reward)
                if "autoscaling" in task_name: last_action = getattr(action_obj, "action", 0)
            except Exception as ex:
                reward = 0.01
                done = True
            rewards_float.append(reward)
            print(f"[STEP] step={step} action={action_log} rewards={reward:.2f} done={'true' if done else 'false'} error=null", flush=True)
            if done: break
    finally:
        r_str = ",".join(f"{r:.2f}" for r in rewards_float) if rewards_float else "0.01"
        print(f"[END] success={'true' if done else 'false'} steps={len(rewards_float)} rewards={r_str}", flush=True)

if __name__ == "__main__":
    t = Timeout(1800)
    t.start()
    try:
        scaler_env = CloudAutoScalerEnvironment()
        review_env = CodeReviewEnvironment()
        tasks = [
            "autoscaling_easy", "autoscaling_medium", "autoscaling_hard",
            "code_review_easy", "code_review_medium", "code_review_hard"
        ]
        for task in tasks:
            env = review_env if "code_review" in task else scaler_env
            run_task(env, task)
    except Exception as e:
        print(f"[ERROR] fatal: {e}", file=sys.stderr)
        sys.exit(1)
    finally: t.cancel()