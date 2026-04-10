"""
client.py — Reference client wrapper for Cloud AutoScaler.
Extends the openenv base client if available, else builds a simple shim.
"""

import argparse
import asyncio
import logging
import httpx
import json
import sys

from models import (
    ScalerAction, ScalerObservation, ResetResult, StepResult,
    WhyDidItFailAction, WhyDidItFailObservation, WhyDidItFailStepResult,
)

log = logging.getLogger("openenv-client")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stderr)

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
                f"util={obs.utilization:.2f} latency={obs.latency_ms:.0f}ms reward={result.reward:+.2f}"
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


# ── WhyDidItFail async WebSocket client ───────────────────────────────────────

class _WDIFResetResult:
    """Lightweight holder returned by WhyDidItFailEnv.reset()."""
    def __init__(self, observation: WhyDidItFailObservation):
        self.observation = observation
        self.done        = False
        self.reward      = 0.10


class _WDIFStepResult:
    """Lightweight holder returned by WhyDidItFailEnv.step()."""
    def __init__(self, observation: WhyDidItFailObservation, reward: float, done: bool):
        self.observation = observation
        self.reward      = reward
        self.done        = done


class WhyDidItFailEnv:
    """
    Async client for the WhyDidItFail environment server.

    Usage::

        env = WhyDidItFailEnv(base_url="http://localhost:8000")
        result = await env.reset(scenario_key="overfitting_easy")
        obs    = result.observation
        result = await env.step(action)
        await env.close()
    """

    def __init__(self, base_url: str = "http://localhost:8000"):
        self._base_url = base_url.rstrip("/")
        self._client: httpx.AsyncClient | None = None

    # -- Lifecycle ------------------------------------------------------------

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(base_url=self._base_url, timeout=30.0)
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    # -- Alternative constructor: spin up a Docker container first ------------

    @classmethod
    async def from_docker_image(cls, image_name: str, port: int = 8000) -> "WhyDidItFailEnv":
        """
        Pull & start the environment from a Docker image, then return a client.
        Requires 'docker' CLI on PATH. Falls back to localhost if image is empty.
        """
        import subprocess, time
        if not image_name:
            return cls(base_url=f"http://localhost:{port}")

        container_name = "wdif_env_auto"
        # Stop any previous instance
        subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)
        proc = subprocess.Popen(
            ["docker", "run", "--rm", "--name", container_name,
             "-p", f"{port}:{port}", image_name],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        # Give the container a moment to start
        time.sleep(3)
        return cls(base_url=f"http://localhost:{port}")

    # -- Core API -------------------------------------------------------------

    async def reset(self, scenario_key: str = "overfitting_easy") -> _WDIFResetResult:
        """
        Start a new episode for the given scenario.
        Returns a result whose .observation is a WhyDidItFailObservation.
        """
        client = await self._get_client()
        resp   = await client.post("/reset", json={"scenario_key": scenario_key})
        resp.raise_for_status()
        data = resp.json()

        obs = WhyDidItFailObservation(
            task_description=data.get("task_description",
                                      data.get("observation", {}).get("task_description", "")),
            feedback=data.get("feedback",
                              data.get("observation", {}).get("feedback", "")),
            visible_data=data.get("visible_data",
                                  data.get("observation", {}).get("visible_data")),
        )
        return _WDIFResetResult(observation=obs)

    async def step(self, action: WhyDidItFailAction) -> _WDIFStepResult:
        """
        Submit one action to the environment.
        Returns a result with .observation, .reward, .done.
        """
        client = await self._get_client()
        resp   = await client.post("/step", json=action.model_dump())
        resp.raise_for_status()
        data   = resp.json()

        obs_data = data.get("observation", data)
        obs = WhyDidItFailObservation(
            task_description=obs_data.get("task_description", ""),
            feedback=obs_data.get("feedback", ""),
            visible_data=obs_data.get("visible_data"),
        )
        reward = float(data.get("reward", 0.10))
        done   = bool(data.get("done", False))
        return _WDIFStepResult(observation=obs, reward=reward, done=done)
