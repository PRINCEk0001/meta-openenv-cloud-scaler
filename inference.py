import json
import os
import sys
import threading
from typing import List

from openai import OpenAI

# Direct imports instead of network client
from models import ScalerAction
from server.environment import CloudAutoScalerEnvironment

# Load config from env or set defaults
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    # Fallback for local testing if not in environment
    HF_TOKEN = "dummy-token"

openai_client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# Prompt engineering to get valid JSON out of the LLM
SYS_PROMPT = "You're a cloud infra bot. Output ONLY a valid JSON object with key 'action' and value 0, 1, or 2. No markdown blocks."

def clamp_reward(r, eps=0.01):
    """Clamps reward to [0.01, 0.99] to avoid boundary rejection."""
    try:
        val = float(r)
    except (TypeError, ValueError):
        val = 0.01
    return max(eps, min(1.0 - eps, val))

def get_scaling_action(obs) -> ScalerAction:
    """Queries the LLM for the next scaling action based on our current traffic/utilization."""
    # Handle both object and dict observations
    traffic = getattr(obs, "current_traffic_load", 0.0)
    servers = getattr(obs, "active_servers", 10)
    latency = getattr(obs, "latency_ms", 10.0)
    capacity = getattr(obs, "total_capacity", 250.0)
    util = getattr(obs, "utilization", 0.5)
    step_num = getattr(obs, "step_number", 0)

    user_prompt = (
        f"Traffic: {traffic:.1f} req/s\n"
        f"Servers: {servers}\n"
        f"Latency: {latency:.1f} ms\n"
        f"Capacity: {capacity:.1f} req/s\n"
        f"Util: {util * 100:.1f}%\n"
        f"Step: {step_num}/50\n\n"
        "Options:\n"
        "0 = hold\n"
        "1 = add server (+25 req/s capacity)\n"
        "2 = remove server (-25 req/s capacity)\n\n"
        "Strategy:\n"
        "- Target 60-80% util\n"
        "- If util > 85% or latency > 50ms, add server (1)\n"
        "- If util < 55% and servers > 10, remove server (2)\n"
        "- Else hold (0)\n\n"
        "Reply with just JSON, e.g. {\"action\": 1}"
    )

    for attempt in range(3):
        try:
            resp = openai_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYS_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=1024,
                timeout=10,
            )

            content = resp.choices[0].message.content
            content = (content or '{"action": 0}').strip()
            content = content.replace("```json", "").replace("```", "").strip()

            parsed = json.loads(content)
            action_val = int(parsed.get("action", 0))

            if action_val not in (0, 1, 2):
                action_val = 0

            return ScalerAction(action=action_val)

        except (json.JSONDecodeError, Exception) as e:
            print(f"API/JSON error (attempt {attempt+1}): {e}", file=sys.stderr)

    return ScalerAction(action=0)


# Hacky cross-platform timeout since SIGALRM isn't a thing on Windows
class Timeout:
    def __init__(self, seconds):
        self.seconds = seconds
        self.timer = None

    def trigger(self):
        print("[ERROR] Timeout reached", file=sys.stderr)
        os._exit(1)

    def start(self):
        self.timer = threading.Timer(self.seconds, self.trigger)
        self.timer.daemon = True
        self.timer.start()

    def cancel(self):
        if self.timer:
            self.timer.cancel()


def run_task(env: CloudAutoScalerEnvironment, task_name: str):
    """Executes the standard inference loop directly against the environment class."""
    step = 0
    rewards_float = []
    done = False

    try:
        # Direct reset
        obs = env.reset(task_name=task_name)
        
        print(f"[START] task={task_name} env=cloud-autoscaler-openenv model={MODEL_NAME}", flush=True)

        for step in range(1, 51):
            action_obj = get_scaling_action(obs)
            # Action string for logging
            action_log_str = json.dumps(action_obj.model_dump(), separators=(",", ":")).replace("\n", " ")
            
            err = None
            try:
                # Direct step (returns obs, reward, done, info)
                obs, reward, done, info = env.step(action_obj)
                reward = float(reward)
            except Exception as ex:
                reward = 0.01
                done = True
                err = str(ex).replace("\n", " ")

            rewards_float.append(reward)
            
            err_log = err if err else "null"
            done_str = "true" if done else "false"

            # [STEP] - PLURAL rewards=, 2dp formatting, flush=True
            print(f"[STEP] step={step} action={action_log_str} rewards={clamp_reward(reward):.2f} done={done_str} error={err_log}", flush=True)
            
            if done:
                break

    finally:
        success_str = "true" if (len(rewards_float) > 0 and done) else "false"
        
        # [END] - Exactly 2 decimal places, comma-separated list
        r_str = ",".join(f"{clamp_reward(r):.2f}" for r in rewards_float) if rewards_float else "0.01"
        print(f"[END] success={success_str} steps={step} rewards={r_str}", flush=True)


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
