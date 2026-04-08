"""
main.py — Evaluator entry point for Meta OpenEnv / Hugging Face Spaces.

Two roles:
  1. Pre-flight self-test (run directly: python main.py)
  2. FastAPI server launcher (imported by uvicorn via Dockerfile CMD)

The pre-flight test mirrors the exact checks Meta's grader performs
during Round 1 automated evaluation.
"""

import sys
import numpy as np
import gymnasium as gym

# ── Import via the canonical HF filename ─────────────────────────────────────
try:
    from env import CloudScalerEnv
except ImportError as exc:
    print(f"[CRITICAL] Cannot import CloudScalerEnv from env.py: {exc}")
    sys.exit(1)


def final_deployment_check() -> bool:
    """
    Mimics exactly how the Meta Grader interacts with the environment.
    Returns True if all checks pass, False otherwise.
    """
    print("Starting Meta OpenEnv Pre-Deployment Test...")
    passed = True

    try:
        env = CloudScalerEnv()
        obs, info = env.reset()

        # ── 1. Observation type ───────────────────────────────────────────────
        if not isinstance(obs, np.ndarray):
            print("ERROR: Observation must be a NumPy array for HF compatibility.")
            return False
        print(f"  [OK] obs is np.ndarray  shape={obs.shape}  dtype={obs.dtype}")

        # ── 2. Action space ───────────────────────────────────────────────────
        if not isinstance(env.action_space, gym.spaces.Discrete):
            print("  [WARN] Discrete action spaces are preferred for this round.")
        else:
            print(f"  [OK] action_space is Discrete(n={env.action_space.n})")

        # ── 3. Required info keys from reset() ────────────────────────────────
        required_reset_keys = {"is_success", "latency_ms", "active_servers", "step_count"}
        missing = required_reset_keys - set(info.keys())
        if missing:
            print(f"  [WARN] reset() info missing keys: {missing}")
        else:
            print(f"  [OK] reset() info contains all required keys")

        # ── 4. 100-step burn-in ───────────────────────────────────────────────
        for s in range(100):
            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)

            # NaN / Inf guard
            if np.isnan(reward) or np.isinf(reward):
                print(f"  [ERROR] NaN or Inf reward at step {s}!")
                passed = False
                break

            # Reward in valid range
            if not (-2.0 <= reward <= 2.0):
                print(f"  [WARN] Reward {reward:.4f} outside [-2, 2] at step {s}")

            # Step returns 5 values (already guaranteed by tuple unpack above)

            # Required info keys on every step
            required_step_keys = {"is_success", "latency_ms", "active_servers", "step_count"}
            step_missing = required_step_keys - set(info.keys())
            if step_missing:
                print(f"  [WARN] step {s} info missing keys: {step_missing}")
                passed = False
                break

            if term or trunc:
                obs, info = env.reset()

        if passed:
            print("  [OK] Survived 100 random steps — 0 NaN/Inf rewards")
            print(f"  Final obs   : {obs}")
            print(f"  Final info  : {info}")

    except Exception as exc:
        import traceback
        print(f"  [CRITICAL] Uncaught exception: {exc}")
        traceback.print_exc()
        return False

    if passed:
        print("SUCCESS: Pre-deployment check passed. Safe to deploy.")
    else:
        print("FAILED: Fix the issues above before deploying.")

    return passed


# ── FastAPI server launch (called by Dockerfile CMD via uvicorn) ──────────────
def start_server():
    """Launch the FastAPI server on port 7860 (Hugging Face default)."""
    import os
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CloudScalerEnv launcher")
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start the FastAPI server instead of running the pre-flight test",
    )
    args = parser.parse_args()

    if args.serve:
        start_server()
    else:
        ok = final_deployment_check()
        sys.exit(0 if ok else 1)
