"""
server/app.py — FastAPI app for the Cloud AutoScaler env.
Matches user 'Winning Snippet' logic with local safe_score for maximum compliance.
"""

import json
import logging
import os
import sys
from typing import Union, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import uvicorn
from pydantic import BaseModel

from .models import EnvInfo, ResetResult, ScalerAction, ScalerObservation, StepResult, GraderRequest, GraderResponse, CodeReviewAction, CodeReviewObservation
from server.environment import CloudAutoScalerEnvironment, CodeReviewEnvironment
from server.tasks import grade_task
from server.utils import clamp_reward

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stderr)
log = logging.getLogger("cloud-autoscaler")

def safe_score(raw) -> str:
    """Implement the strict [0.01, 0.99] safety clamp and 2dp formatting locally for Phase 2."""
    try:
        val = float(raw if raw is not None else 0.10)
    except (ValueError, TypeError):
        val = 0.10
    clamped = max(0.01, min(0.99, val))
    return f"{clamped:.2f}"

_env_instance = None

@asynccontextmanager
async def lifespan(application: FastAPI):
    global _env_instance
    log.info("Spinning up environment...")
    _env_instance = CloudAutoScalerEnvironment()
    _env_instance.reset()
    log.info("Environment ready to go")
    yield
    log.info("Shutting down")


app = FastAPI(
    title="Cloud Autoscaler OpenEnv",
    description="RL env for cloud infrastructure auto-scaling.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    # Serve the Kinetic Console as the landing page
    # Since app.py is run from the root or server directory, index.html is in the project root.
    # We'll use the absolute path or check availability.
    return FileResponse("index.html")


@app.get("/health")
async def health_check():
    return {"status": "healthy", "environment": "cloud-autoscaler"}


@app.get("/info", response_model=EnvInfo)
async def info():
    return EnvInfo()


class ResetRequest(BaseModel):
    task: str = "autoscaling_easy"


@app.post("/reset", response_model=ResetResult)
async def reset(req: ResetRequest = None):
    global _env_instance
    task_name = req.task if req else "autoscaling_easy"
    
    if "code_review" in task_name.lower() or "codereview" in task_name.lower():
        if not isinstance(_env_instance, CodeReviewEnvironment):
            _env_instance = CodeReviewEnvironment()
    else:
        if not isinstance(_env_instance, CloudAutoScalerEnvironment):
            _env_instance = CloudAutoScalerEnvironment()
            
    obs = _env_instance.reset(task_name=task_name)
    ep_id = getattr(_env_instance._state, "episode_id", "unknown")
    
    # [START] Mandatory Phase 1 log
    print(f"[START] task={task_name} env=cloud-autoscaler-openenv model={os.getenv('MODEL_NAME', 'llama-3.1-8b-instant')}", flush=True)
        
    return ResetResult(observation=obs, info={"episode_id": ep_id})


@app.post("/step", response_model=StepResult)
async def step(action: Union[ScalerAction, CodeReviewAction, Any]):
    global _env_instance
    if _env_instance is None:
        raise HTTPException(status_code=400, detail="Call /reset first")
        
    obs, reward, done, info = _env_instance.step(action)
    
    # [STEP] Mandatory Phase 1 log - Winning Snippet Format
    action_val = getattr(action, "action", 0)
    s_reward = safe_score(reward)
    print(f"[STEP] step={_env_instance._step_count} action={{\"action\":{action_val}}} reward={s_reward} done={'true' if done else 'false'} error=null", flush=True)
    return StepResult(observation=obs, reward=float(s_reward), done=done, info=info)
    

@app.post("/grader", response_model=GraderResponse)
async def grader(req: GraderRequest):
    global _env_instance
    try:
        if _env_instance is None or _env_instance._state is None:
            return GraderResponse(task=req.task, score=0.10, is_success=False)
        
        raw_score = grade_task(req.task, _env_instance._state)
        # OpenEnv Phase 2 Hardening: task score must be strictly in (0, 1)
        final_score = max(0.01, min(0.99, float(raw_score)))
        
        is_success = bool(final_score >= 0.5)
        rewards_list = getattr(_env_instance._state, "step_rewards", [])
        
        # Unified [END] format
        print(f"[END] task={req.task} score={final_score:.2f} steps={len(rewards_list)}", flush=True)

        return GraderResponse(task=req.task, score=final_score, is_success=is_success)
        
    except Exception as e:
        log.error(f"Grader error: {e}")
        return GraderResponse(task=req.task, score=0.10, is_success=False)



@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    ws_env = CloudAutoScalerEnvironment()
    obs = ws_env.reset()
    await websocket.send_json(ResetResult(observation=obs, info={"episode_id": ws_env._state.episode_id}).model_dump())
    
    try:
        while True:
            data = json.loads(await websocket.receive_text())
            if data.get("reset"):
                obs = ws_env.reset()
                await websocket.send_json(ResetResult(observation=obs, info={"episode_id": ws_env._state.episode_id}).model_dump())
                continue
            action = ScalerAction(action=int(data.get("action", 0)))
            obs, reward, done, info = ws_env.step(action)
            s_reward = safe_score(reward)
            await websocket.send_json(StepResult(observation=obs, reward=float(s_reward), done=done, info=info).model_dump())
            if done:
                obs = ws_env.reset()
    except Exception:
        pass
    finally:
        await websocket.close()

def main():
    """Entry point required for HF multi-mode deployment"""
    import os
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)

if __name__ == "__main__":
    main()
