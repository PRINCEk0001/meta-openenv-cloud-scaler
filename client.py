"""
client.py — Reference client wrapper for Cloud AutoScaler.
Extends the openenv base client if available, else builds a simple shim.
"""

import argparse
import asyncio
import logging
import httpx

from models import ScalerAction, ScalerObservation, ResetResult, StepResult

log = logging.getLogger("openenv-client")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

try:
    from openenv.core.client import EnvClient as _EnvClient  # type: ignore
except ImportError:
    class _EnvClient:  # type: ignore
        """Quick shim to mock openenv.core.client.EnvClient"""
        def __init__(self, base_url: str = "http://localhost:7860"):
            self.base_url = base_url.rstrip("/")
            self._sync_mode = False

        def sync(self):
            self._sync_mode = True
            return self 

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def reset(self, task_name: str = "autoscaling_easy") -> ScalerObservation:
            r = httpx.post(f"{self.base_url}/reset", json={"task": task_name}, timeout=10)
            r.raise_for_status()
            return ResetResult.model_validate(r.json()).observation

        def step(self, action: ScalerAction) -> StepResult:
            r = httpx.post(f"{self.base_url}/step", json=action.model_dump(), timeout=10)
            r.raise_for_status()
            return StepResult.model_validate(r.json())


class CloudAutoScalerEnv(_EnvClient):
    """
    Client for the Cloud AutoScaler env.
    Supports both async and sync (context manager) execution.
    """
    action_type = ScalerAction
    observation_type = ScalerObservation

    def __init__(self, base_url: str = "http://localhost:7860"):
        super().__init__(base_url=base_url)

    async def areset(self, task_name: str = "autoscaling_easy") -> ScalerObservation:
        async with httpx.AsyncClient(base_url=self.base_url, timeout=10) as c:
            r = await c.post("/reset", json={"task": task_name})
            r.raise_for_status()
            return ResetResult.model_validate(r.json()).observation

    async def astep(self, action: ScalerAction) -> StepResult:
        async with httpx.AsyncClient(base_url=self.base_url, timeout=10) as c:
            r = await c.post("/step", json=action.model_dump())
            r.raise_for_status()
            return StepResult.model_validate(r.json())

    async def ahealth(self) -> dict:
        async with httpx.AsyncClient(base_url=self.base_url, timeout=10) as c:
            r = await c.get("/health")
            r.raise_for_status()
            return r.json()


def _heuristic(obs: ScalerObservation) -> int:
    """Basic agent that scales based on hard utilization thresholds."""
    if obs.utilization > 0.85 or obs.latency_ms > 50:
        return 1   # add
    elif obs.utilization < 0.55 and obs.active_servers > 10:
        return 2   # remove
    return 0       # hold

async def _run_async(host: str, n_steps: int):
    env = CloudAutoScalerEnv(base_url=host)
    await env.ahealth()
    obs = await env.areset()
    
    log.info(f"Reset: step={obs.step_number} servers={obs.active_servers} util={obs.utilization:.2f}")

    total = 0.0
    for _ in range(n_steps):
        action = ScalerAction(action=_heuristic(obs))
        result = await env.astep(action)
        obs = result.observation
        total += result.reward
        
        log.info(
            f"step={obs.step_number} | action={action.action} servers={obs.active_servers} "
            f"util={obs.utilization:.2f} latency={obs.latency_ms:.0f}ms reward={result.reward:+.3f}"
        )
        if result.done:
            break

    log.info(f"Done - total reward: {total:.4f}")

def _run_sync(host: str, n_steps: int):
    env = CloudAutoScalerEnv(base_url=host).sync()
    with env as e:
        obs = e.reset()
        log.info(f"Reset: step={obs.step_number} servers={obs.active_servers}")
        
        total = 0.0
        for _ in range(n_steps):
            action = ScalerAction(action=_heuristic(obs))
            result = e.step(action)
            obs = result.observation
            total += result.reward
            
            log.info(
                f"step={obs.step_number} | action={action.action} servers={obs.active_servers} "
                f"util={obs.utilization:.2f} latency={obs.latency_ms:.0f}ms reward={result.reward:+.3f}"
            )
            if result.done:
                break
                
        log.info(f"Done - total reward: {total:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cloud Autoscaler reference client")
    parser.add_argument("--host", default="http://localhost:7860")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--sync", action="store_true", help="Use sync client")
    args = parser.parse_args()

    if args.sync:
        _run_sync(args.host, args.steps)
    else:
        asyncio.run(_run_async(args.host, args.steps))
