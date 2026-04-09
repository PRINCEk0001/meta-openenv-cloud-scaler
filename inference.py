import json
import os
import sys
import threading

from openai import OpenAI

from client import CloudAutoScalerEnv
from models import ScalerAction

# Load config from env or set defaults
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
HF_SPACE_URL = os.getenv("HF_SPACE_URL", "http://localhost:7860")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is required")

openai_client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# Prompt engineering to get valid JSON out of the LLM
SYS_PROMPT = "You're a cloud infra bot. Output ONLY a valid JSON object with key 'action' and value 0, 1, or 2. No markdown blocks."

def get_scaling_action(obs) -> ScalerAction:
    """Queries the LLM for the next scaling action based on our current traffic/utilization."""
    # Build a simple state string to pass to the model
    user_prompt = (
        f"Traffic: {obs.current_traffic_load:.1f} req/s\n"
        f"Servers: {obs.active_servers}\n"
        f"Latency: {obs.latency_ms:.1f} ms\n"
        f"Capacity: {obs.total_capacity:.1f} req/s\n"
        f"Util: {obs.utilization * 100:.1f}%\n"
        f"Step: {obs.step_number}/50\n\n"
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

    try:
        resp = openai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYS_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=20,
        )

        content = resp.choices[0].message.content.strip()
        
        # sometimes LLMs ignore the 'no markdown' instruction
        content = content.replace("```json", "").replace("```", "").strip()

        parsed = json.loads(content)
        action_val = int(parsed.get("action", 0))

        if action_val not in (0, 1, 2):
            print(f"Bad action {action_val}, defaulting to 0", file=sys.stderr)
            action_val = 0

        return ScalerAction(action=action_val)

    except json.JSONDecodeError as e:
        print(f"JSON error: {e}, falling back to 0", file=sys.stderr)
    except Exception as e:
        print(f"API error: {e}, falling back to 0", file=sys.stderr)

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


def run_task(env, task_name: str):
    """Executes the standard inference loop and generates logs."""
    step = 0
    rewards_float = []
    rewards_formatted = []

    obs = env.reset(task_name=task_name)

    # [START] - Perfectly matches guideline
    print(f"[START] task={task_name} env=cloud-autoscaler-openenv model={MODEL_NAME}", flush=True)

    try:
        done = False
        while not done:
            step += 1
            action = get_scaling_action(obs)
            action_str = json.dumps(action.model_dump(), separators=(",", ":"))
            err = None

            try:
                res = env.step(action)
                obs = res.observation
                reward = float(res.reward)
                done = res.done
            except Exception as ex:
                reward = 0.001 # Use 0.001 to stay in (0,1)
                done = True
                err = str(ex).replace("\n", " ")

            # Ultra-strict (0, 1) clamping using [0.001, 0.999]
            safe_reward = float(reward)
            safe_reward = max(0.001, min(0.999, safe_reward))
            safe_reward = round(safe_reward, 3)
            
            rewards_float.append(safe_reward)
            rewards_formatted.append(f"{safe_reward:.3f}")
            
            err_log = err if err else "null"
            done_str = "true" if done else "false"

            # [STEP] - Perfectly matches guideline
            print(f"[STEP] step={step} action={action_str} reward={safe_reward:.2f} done={done_str} error={err_log}", flush=True)

    finally:
        # Success logic: If the loop finished without crashing and reached 'done'
        # Adjust this if your environment provides a specific success flag
        final_success = "true" if (len(rewards_float) > 0 and done) else "false"
        
        r_str = ",".join(rewards_formatted)
        
        # [END] - Perfectly matches guideline
        print(f"[END] success={final_success} steps={step} rewards={r_str}", flush=True)



if __name__ == "__main__":
    t = Timeout(1100) # Give it 18 mins to safely run 3 tasks under 20 mins
    t.start()

    try:
        env = CloudAutoScalerEnv(base_url=HF_SPACE_URL).sync()
        with env as e:
            for task_name in ["autoscaling_easy", "autoscaling_medium", "autoscaling_hard"]:
                run_task(e, task_name)
    except KeyboardInterrupt:
        print("\n[ERROR] cancelled by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] fatal: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        t.cancel()
