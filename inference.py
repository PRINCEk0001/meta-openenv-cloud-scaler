````python
import json
import os
import sys
import threading
from typing import List

from openai import OpenAI

from models import ScalerAction
from server.environment import CloudAutoScalerEnvironment


API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
HF_TOKEN = os.getenv("HF_TOKEN")

openai_client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN or "dummy-token"
)

SYS_PROMPT = (
    "You're a cloud infra bot. Output ONLY a valid JSON object "
    "with key 'action' and value 0, 1, or 2. No markdown blocks."
)

# META STRICT RANGE
MIN_VALID_SCORE = 0.002
MAX_VALID_SCORE = 0.998


def clamp_reward(r):
    r = float(r)

    if r <= MIN_VALID_SCORE:
        return MIN_VALID_SCORE

    if r >= MAX_VALID_SCORE:
        return MAX_VALID_SCORE

    return r


def get_scaling_action(obs) -> ScalerAction:
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

    for _ in range(3):
        try:
            resp = openai_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYS_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=200,
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

        except Exception:
            pass

    return ScalerAction(action=0)


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
    rewards = []
    done = False

    try:
        obs = env.reset(task_name=task_name)

        print(
            f"[START] task={task_name} env=cloud-autoscaler-openenv model={MODEL_NAME}",
            flush=True,
        )

        for step in range(1, 51):

            action_obj = get_scaling_action(obs)
            action_log = json.dumps(
                action_obj.model_dump(),
                separators=(",", ":")
            )

            err = None

            try:
                obs, raw_reward, done, _ = env.step(action_obj)

                raw_reward = float(raw_reward or 0.0)

                reward = clamp_reward(raw_reward)
                reward = float(f"{reward:.3f}")

            except Exception as ex:
                reward = MIN_VALID_SCORE
                done = True
                err = str(ex).replace("\n", " ")

            rewards.append(reward)

            print(
                f"[STEP] step={step} "
                f"action={action_log} "
                f"rewards={reward:.3f} "
                f"done={'true' if done else 'false'} "
                f"error={err if err else 'null'}",
                flush=True,
            )

            if done:
                break

    finally:
        success = "true" if len(rewards) == 50 else "false"

        rewards_str = ",".join(f"{r:.3f}" for r in rewards)

        print(
            f"[END] success={success} "
            f"steps={len(rewards)} "
            f"rewards={rewards_str}",
            flush=True,
        )


if __name__ == "__main__":
    t = Timeout(1100)
    t.start()

    try:
        for task in [
            "autoscaling_easy",
            "autoscaling_medium",
            "autoscaling_hard",
        ]:
            env = CloudAutoScalerEnvironment()
            run_task(env, task)

    finally:
        t.cancel()
````
