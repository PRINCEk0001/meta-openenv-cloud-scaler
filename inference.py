import json
import os
import sys
import threading
import math
from typing import List, Union, Any

_root = os.path.dirname(os.path.abspath(__file__))
if _root not in sys.path:
    sys.path.insert(0, _root)

from openai import OpenAI

from server.models import ScalerAction, CodeReviewAction, WhyDidItFailAction
from server.environment import CloudAutoScalerEnvironment, CodeReviewEnvironment
from server.tasks import grade_task

# ── BUG 3 FIX: read HF_TOKEN (not API_KEY) — validator injects HF_TOKEN ─────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "llama-3.1-8b-instant")
HF_TOKEN     = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

def safe_score(raw) -> str:
    """Implement the strict [0.01, 0.99] safety clamp and 2dp formatting locally."""
    try:
        val = float(raw if raw is not None else 0.10)
    except (ValueError, TypeError):
        val = 0.10
    clamped = max(0.01, min(0.99, val))
    return f"{clamped:.2f}"

def clamp_reward(r, eps=0.01) -> float:
    """Numeric clamp returning float for internal history."""
    try:
        val = float(r)
    except (ValueError, TypeError):
        val = eps
    return max(eps, min(1.0 - eps, val))

SYS_PROMPT_SCALER = """
You are Anigrevity, an autonomous Cloud Infrastructure Controller for Meta's OpenEnv.

MISSION
Keep latency <50ms while using the fewest servers possible. Episode = 50 steps. Each server = 25 req/s capacity.

OBSERVATION FIELDS
- current_traffic_load: float (req/s)
- active_servers: int [1-50]
- latency_ms: float
- utilization: traffic / (servers*25)
- step_number: int

DECISION PROTOCOL (think silently, do not output reasoning):
1. Calculate projected utilization after action
2. CRITICAL: if utilization > 0.85 OR latency_ms > 50 → action=1 (SCALE UP)
3. DANGER: if utilization > 0.75 AND latency_ms > 40 → action=1
4. UNDER-UTILIZED: if utilization < 0.45 AND active_servers > 1 AND latency_ms < 30 → action=2 (SCALE DOWN)
5. STABLE: if 0.60 ≤ utilization ≤ 0.75 AND latency_ms < 50 → action=0 (HOLD)
6. ANTI-OSCILLATION: if last_action was 1 and utilization < 0.70 → prefer 0 over 2

SCORING AWARENESS
Meta grades 40% utilization closeness to 70%, 25% latency stability, 20% cost, 15% smoothness. Never flip-flop.

OUTPUT RULES — VIOLATION = VALIDATION ERROR
- Respond with EXACTLY one JSON object, no markdown, no text before/after
- Schema: {"action": 0} or {"action": 1} or {"action": 2}
- 0=hold, 1=add server, 2=remove server
- Integer only, not string

EXAMPLES
obs: utilization=0.92, latency=68 → {"action": 1}
obs: utilization=0.42, latency=22, servers=8 → {"action": 2}
obs: utilization=0.68, latency=38 → {"action": 0}
"""

SYS_PROMPT_REVIEW = """You are a senior security code reviewer for a production CI pipeline. Your job is to block vulnerabilities, request fixes for bugs, and approve only after a complete scan.

DECISION PROTOCOL:
1. Scan for critical patterns: SQL injection, XSS, command injection, path traversal, SSRF, hardcoded secrets or API keys, eval or exec, pickle.loads or unsafe deserialization, missing authentication, missing authorization, insecure cryptography, and logic errors that cause crashes.
2. If you find a critical security flaw that enables RCE, data theft, or auth bypass, choose action_type reject and severity critical.
3. If you find a bug, missing validation, or non-critical security issue, choose action_type request_changes and severity medium.
4. If you find a minor style issue or low risk concern, choose action_type comment and severity low.
5. Only choose action_type approve if you find zero issues after the full scan. For approve, set severity low.

OUTPUT RULES:
- Respond with exactly one JSON object. No markdown, no backticks, no extra text.
- Schema must be exactly: {"action_type": "approve"|"reject"|"request_changes"|"comment", "severity": "low"|"medium"|"high"|"critical", "comment": "<brief reason>", "reasoning": "<one sentence citing evidence>"}
- All four keys are required for every response.
- For reject and request_changes, comment must name the vulnerability pattern, reasoning must cite the specific function, variable, or line pattern.
- For approve, use comment "no issues found" and reasoning "full security scan passed".
- For comment, use severity low.
"""

SYS_PROMPT_WDIF = """
You are Anigrevity, an ML Failure Diagnosis Agent.

MISSION
Diagnose training failures in 3-4 inspections, then submit_diagnosis. Score 0.99 requires exact label + numeric evidence.

INSPECTION ORDER:
1. inspect_logs → check loss curves
2. inspect_config → check lr, optimizer, activation
3. inspect_gradients → if required (for vanishing/exploding)

FAILURE LABELS (use EXACT string):
- "underfitting"
- "overfitting"
- "learning rate too high"
- "learning rate too low"
- "exploding gradients"
- "vanishing gradients"
- "dying relu"
- "bad weight initialization"

DIAGNOSIS RULES:
- loss oscillates wildly (e.g., 2.31→0.45→3.12) + lr=5.0 → "learning rate too high"
- train_acc ~ val_acc ~ 0.10 for 20 epochs, linear model → "underfitting"
- train_loss ↓, val_loss ↑ after epoch 20 → "overfitting"
- loss NaN after epoch 3, RNN → "exploding gradients"
- gradient norms: 1.2e-1 → 3.4e-9 exponentially → "vanishing gradients"
- all hidden gradient norms = 0.0 with ReLU → "dying relu"

OUTPUT RULES
For inspections: {"action_type":"inspect_logs"} etc.
For final: {"action_type":"submit_diagnosis","diagnosis":"<exact label>","suggested_fix":"<concrete fix>","reasoning":"<cite numbers from logs/config>"}

EXAMPLE
→ {"action_type":"submit_diagnosis","diagnosis":"learning rate too high","suggested_fix":"reduce lr from 5.0 to 0.001 and add gradient clipping","reasoning":"train_loss oscillates 2.31 to 3.12 across epochs with lr=5.0 in config"}
"""


def get_action(obs: Any, task_name: str, last_action: int = 0) -> Union[ScalerAction, CodeReviewAction, WhyDidItFailAction]:
    task_name_l = task_name.lower()
    is_code_review = "code_review" in task_name_l or "codereview" in task_name_l
    is_wdif = "whydiditfail" in task_name_l or "wdif" in task_name_l

    if is_code_review:
        prompt  = f"File: {getattr(obs, 'file_content', '')}\nDiff: {getattr(obs, 'diff_summary', '')}"
        sys_msg = SYS_PROMPT_REVIEW
    elif is_wdif:
        prompt = f"Description: {getattr(obs, 'task_description', '')}\nFeedback: {getattr(obs, 'feedback', '')}\nData: {getattr(obs, 'visible_data', {})}"
        sys_msg = SYS_PROMPT_WDIF
    else:
        if isinstance(obs, (list, tuple)) or hasattr(obs, "shape"):
            traffic, servers, latency = obs[0], obs[1], obs[2]
            util = traffic / (servers * 25.0) if servers > 0 else 0.5
        else:
            servers = getattr(obs, "active_servers", 10)
            latency = getattr(obs, "latency_ms",     10.0)
            util    = getattr(obs, "utilization",    0.5)
        prompt  = f"Utilization: {util*100:.1f}%, Latency: {latency:.1f}ms, Servers: {servers}."
        sys_msg = SYS_PROMPT_SCALER

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user",   "content": prompt},
            ],
            response_format={"type": "json_object"},
            timeout=10.0,
        )
        data = json.loads(completion.choices[0].message.content)

        if is_code_review:
            return CodeReviewAction(
                action_type=data.get("action_type", "reject"),
                severity=data.get("severity", "low"),
                comment=data.get("comment", "Audit"),
                reasoning=data.get("reasoning", "Trajectory review")
            )
        elif is_wdif:
            return WhyDidItFailAction(
                action_type=data.get("action_type", "inspect_logs"),
                diagnosis=data.get("diagnosis"),
                suggested_fix=data.get("suggested_fix"),
                reasoning=data.get("reasoning")
            )
        else:
            action_val = int(data.get("action", 0))
            if last_action == 1 and action_val == 2: action_val = 0
            if last_action == 2 and action_val == 1: action_val = 0
            return ScalerAction(action=action_val)

    except Exception:
        if is_code_review:
            return CodeReviewAction(action_type="reject", comment="Security Fallback")
        else:
            h_action = 0
            if util > 0.85 or latency > 50.0: h_action = 1
            elif util < 0.45 and servers > 1:  h_action = 2
            return ScalerAction(action=h_action)


class Timeout(threading.Thread):
    def __init__(self, seconds):
        super().__init__()
        self.seconds = seconds
        self.daemon  = True
        self._cancel = threading.Event()

    def run(self):
        if not self._cancel.wait(self.seconds):
            print(f"[TIMEOUT] Exceeded {self.seconds}s limit.", file=sys.stderr)
            os._exit(1)

    def cancel(self):
        self._cancel.set()


def run_task(env: Any, task_name: str):
    rewards_history = []
    last_action     = 0
    done            = False
    step            = 0
    max_steps       = 50 if "autoscaling" in task_name else 5

    try:
        # BUG 5 FIX: env.reset() returns ResetResult — extract .observation
        reset_result = env.reset(task_name=task_name)
        obs = reset_result.observation if hasattr(reset_result, "observation") else reset_result

        print(
            f"[START] task={task_name} env=cloud-autoscaler-openenv model={MODEL_NAME}",
            flush=True,
        )

        for step in range(1, max_steps + 1):
            err = None
            try:
                action_obj = get_action(obs, task_name, last_action)
                action_log = json.dumps(
                    action_obj.model_dump(), separators=(",", ":")
                ).replace("\n", " ")

                # BUG 4 FIX: env.step() returns StepResult object, not a tuple
                step_result = env.step(action_obj)
                if hasattr(step_result, "observation"):
                    obs        = step_result.observation
                    res_reward = step_result.reward
                    done       = step_result.done
                else:
                    obs, res_reward, done, _ = step_result

                reward = clamp_reward(res_reward)

                if "autoscaling" in task_name:
                    last_action = getattr(action_obj, "action", 0)

            except Exception as ex:
                reward     = 0.01
                done       = True
                action_log = '{"action":0}'
                err        = str(ex).replace("\n", " ")

            rewards_history.append(reward)
            err_str  = err if err else "null"

            # BUG 1 FIX: [STEP] uses reward= (SINGULAR)
            print(
                f"[STEP] step={step} action={action_log} "
                f"reward={safe_score(reward)} "
                f"done={'true' if done else 'false'} "
                f"error={err_str}",
                flush=True,
            )

            if done:
                break

    finally:
        # BUG 2 FIX: [END] inside finally — always fires even on exception
        # OpenEnv Phase 2 Hardening: task score must be strictly in (0, 1)
        raw_score = grade_task(task_name, env._state)
        final_score = max(0.01, min(0.99, float(raw_score)))
        
        print(
            f"[END] task={task_name} score={final_score:.2f} steps={step}",
            flush=True,
        )


if __name__ == "__main__":
    t = Timeout(1800)
    t.start()

    try:
        scaler_env = CloudAutoScalerEnvironment()
        review_env = CodeReviewEnvironment()

        tasks = [
            "autoscaling_easy",  "autoscaling_medium",  "autoscaling_hard",
            "code_review_easy",  "code_review_medium",  "code_review_hard",
        ]

        for task in tasks:
            env = review_env if "code_review" in task else scaler_env
            run_task(env, task)

    except Exception as fatal:
        print(f"[ERROR] Fatal crash: {fatal}", file=sys.stderr)
        sys.exit(1)
    finally:
        t.cancel()