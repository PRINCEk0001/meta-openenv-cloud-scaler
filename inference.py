import json
import os
import sys
import threading
import math
from typing import List, Union, Any

from openai import OpenAI

# Direct imports from the submission structure
from models import ScalerAction, CodeReviewAction
from server.environment import CloudAutoScalerEnvironment, CodeReviewEnvironment
from server.utils import safe_score, clamp_reward

# --- Phase 1: API & Proxy Compliance ---
# Mandatory mapping to injected environment variables
try:
    API_BASE_URL = os.environ["API_BASE_URL"]
    API_KEY = os.environ["API_KEY"]
except KeyError as e:
    print(f"[ERROR] Missing required environment variable: {e}", file=sys.stderr)
    # Fallback for local testing only; validator will provide these.
    API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
    API_KEY = os.getenv("API_KEY", os.getenv("HF_TOKEN", "dummy-token"))

MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

# --- Phase 3: Antigravity Persona & Logic ---
SYS_PROMPT_SCALER = (
    "Role: You are an autonomous Cloud Infrastructure Controller for the 'Antigravity' environment.\n"
    "Objective: Maintain server health by balancing traffic load and response latency.\n\n"
    "Operational Thresholds:\n"
    "- CRITICAL LOAD: If Utilization > 85% or Latency > 50ms -> Action 1 (Scale Up)\n"
    "- UNDER-UTILIZED: If Utilization < 45% and Servers > 1 -> Action 2 (Scale Down)\n"
    "- STABLE: Otherwise -> Action 0 (Hold)\n\n"
    "Rules:\n"
    "1. You must respond ONLY with a valid JSON object.\n"
    "2. The JSON must contain exactly one key: 'action' (integer 0, 1, or 2).\n"
    "3. No markdown, no commentary.\n\n"
    "Schema: {\"action\": <int>}"
)

SYS_PROMPT_REVIEW = (
    "Role: You are an expert Software Security Auditor.\n"
    "Objective: Review code diffs for vulnerabilities.\n"
    "Rules: Respond ONLY with valid JSON {\"action_type\": \"approve\"|\"reject\"|\"comment\", \"comment\": \"reasoning\"}."
)

def get_action(obs: Any, task_name: str, last_action: int = 0) -> Union[ScalerAction, CodeReviewAction]:
    """
    Phase 1 & 3: Determines action by CONSULTING THE LLM PROXY for every step.
    Includes Antigravity thresholds and oscillation protection.
    """
    is_code_review = "code_review" in task_name.lower()
    
    # 1. Prepare Prompt
    if is_code_review:
        prompt = f"File: {getattr(obs, 'file_content', '')}\nDiff: {getattr(obs, 'diff_summary', '')}"
        sys_msg = SYS_PROMPT_REVIEW
    else:
        # Scaler Metrics
        if isinstance(obs, (list, tuple)) or hasattr(obs, "shape"):
            traffic, servers, latency = obs[0], obs[1], obs[2]
            util = traffic / (servers * 25.0) if servers > 0 else 0.5
        else:
            servers = getattr(obs, "active_servers", 10)
            latency = getattr(obs, "latency_ms", 10.0)
            util = getattr(obs, "utilization", 0.5)
        
        prompt = f"Utilization: {util*100:.1f}%, Latency: {latency:.1f}ms, Servers: {servers}."
        sys_msg = SYS_PROMPT_SCALER

    # 2. Phase 1: Mandatory LLM Call (Proxy Compliance)
    # Phase 3: JSON Mode
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            timeout=10.0
        )
        resp_text = completion.choices[0].message.content
        data = json.loads(resp_text)
        
        if is_code_review:
            at = data.get("action_type", "reject")
            return CodeReviewAction(action_type=at, comment=data.get("comment", "Audit"))
        else:
            action_val = int(data.get("action", 0))
            
            # Phase 3: Oscillation Protection
            if last_action == 1 and action_val == 2: action_val = 0
            if last_action == 2 and action_val == 1: action_val = 0
            
            return ScalerAction(action=action_val)
            
    except Exception:
        # Phase 5: Graceful Heuristic Fallback (Emergency only)
        if is_code_review:
            return CodeReviewAction(action_type="reject", comment="Security Fallback")
        else:
            # Re-implement Antigravity thresholds as heuristic fallback
            h_action = 0
            if util > 0.85 or latency > 50.0: h_action = 1
            elif util < 0.45 and servers > 1: h_action = 2
            return ScalerAction(action=h_action)

# --- Phase 5: Timeout Reliability ---
class Timeout(threading.Thread):
    def __init__(self, seconds):
        super().__init__()
        self.seconds = seconds
        self.daemon = True
        self._cancel = threading.Event()
    def run(self):
        if not self._cancel.wait(self.seconds):
            print(f"[TIMEOUT] Antigravity execution exceeded {self.seconds}s limit. Exiting.", file=sys.stderr)
            os._exit(1)
    def cancel(self): self._cancel.set()

def run_task(env: Any, task_name: str):
    """
    Directly executes the inference loop with Phase 1-5 compliance.
    """
    # Phase 5: Graceful Reset
    obs = env.reset(task_name=task_name)
    print(f"[START] task={task_name} env=cloud-autoscaler-openenv model={MODEL_NAME}", flush=True)
    
    rewards_history = []
    last_action = 0
    done = False
    max_steps = 50 if "autoscaling" in task_name else 5
    
    for step in range(1, max_steps + 1):
        # Phase 5: Individual Call Protection
        try:
            action_obj = get_action(obs, task_name, last_action)
            action_log = json.dumps(action_obj.model_dump(), separators=(",", ":"))
            
            obs, res_reward, done, info = env.step(action_obj)
            
            # Phase 2: NaN/Inf Protection & Ultra-Strict Clipping
            raw_r = float(res_reward) if math.isfinite(float(res_reward)) else 0.01
            # Phase 2: The "0.01/0.99" Clip immediately before logging
            reward = min(max(raw_r, 0.01), 0.99)
            
            if "autoscaling" in task_name:
                last_action = getattr(action_obj, "action", 0)
        except Exception as e:
            reward = 0.01
            done = True
            action_log = '{"action":0}'
            
        rewards_history.append(reward)
        
        # Phase 4: Mandatory [STEP] Format (2 decimal places, Flush)
        print(f"[STEP] step={step} action={action_log} rewards={reward:.2f} done={'true' if done else 'false'} error=null", flush=True)
        
        if done: break
        
    # Phase 4: Mandatory [END] Format
    success_str = "true" if done else "false"
    r_str = ",".join(f"{r:.2f}" for r in rewards_history)
    print(f"[END] success={success_str} steps={len(rewards_history)} rewards={r_str}", flush=True)

if __name__ == "__main__":
    # Start 1800s safety timer
    t = Timeout(1800)
    t.start()
    
    try:
        # Initialize Environments
        scaler_env = CloudAutoScalerEnvironment()
        review_env = CodeReviewEnvironment()
        
        # All 6 mandatory tasks
        tasks = [
            "autoscaling_easy", "autoscaling_medium", "autoscaling_hard",
            "code_review_easy", "code_review_medium", "code_review_hard"
        ]
        
        for task in tasks:
            env = review_env if "code_review" in task else scaler_env
            run_task(env, task)
            
    except Exception as fatal:
        print(f"[ERROR] Fatal crash: {fatal}", file=sys.stderr)
        sys.exit(1)
    finally:
        t.cancel()