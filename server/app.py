"""
server/app.py — FastAPI app for the Cloud AutoScaler env.

We route this locally using standard endpoints so we can inject 
"task" configs via the HTTP payloads exactly as needed.
"""

import json
import logging
import os
import sys
from typing import Union, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel

from .models import EnvInfo, ResetResult, ScalerAction, ScalerObservation, StepResult, GraderRequest, GraderResponse, CodeReviewAction, CodeReviewObservation
from server.environment import CloudAutoScalerEnvironment, CodeReviewEnvironment
from server.tasks import grade_task
from server.utils import safe_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stderr)
log = logging.getLogger("cloud-autoscaler")


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
    return {
        "name": "cloud-autoscaler-env",
        "version": "1.0.0",
        "status": "healthy",
        "docs": "/docs",
        "health": "/health",
        "reset": "POST /reset",
        "step": "POST /step",
    }


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
    
    # Environment Factory
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
    
    # [STEP] Mandatory Phase 1 log - Strict Clamp
    action_val = getattr(action, "action", 0)
    # Ensure even step rewards are clamped away from 0.0/1.0
    s_reward = safe_score(reward)
    print(f"[STEP] step={_env_instance._step_count} action={{\"action\":{action_val}}} rewards={s_reward} done={'true' if done else 'false'} error=null", flush=True)
    return StepResult(observation=obs, reward=float(s_reward), done=done, info=info)
    

@app.post("/grader", response_model=GraderResponse)
async def grader(req: GraderRequest):
    global _env_instance
    try:
        if _env_instance is None or _env_instance._state is None:
            log.warning(f"Grader called for {req.task} but no environment state is available.")
            return GraderResponse(task=req.task, score=0.10, is_success=False)
        
        raw_score = grade_task(req.task, _env_instance._state)
        score_str = safe_score(raw_score)
        
        is_success = bool(float(score_str) >= 0.5)
        
        # [END] Mandatory Phase 1 log - Strict Clamp every reward in the list
        rewards_list = getattr(_env_instance._state, "step_rewards", [])
        # Apply safe_score to EVERY item in the join to prevent 0.00/1.00 in the list
        r_str = ",".join(safe_score(r) for r in rewards_list) if rewards_list else "0.10"
        print(f"[END] success={'true' if is_success else 'false'} steps={len(rewards_list)} rewards={r_str}", flush=True)

        log.info(f"Grading ({req.task}) -> Raw={raw_score:.4f}, Final={score_str}, success={is_success}")
        return GraderResponse(task=req.task, score=float(score_str), is_success=is_success)
        
    except Exception as e:
        log.error(f"Critical error during grading of {req.task}: {e}")
        return GraderResponse(task=req.task, score=0.10, is_success=False)



@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    log.info("WS client connected")
    
    ws_env = CloudAutoScalerEnvironment()
    obs = ws_env.reset()
    await websocket.send_json(
        ResetResult(
            observation=obs,
            info={"episode_id": ws_env._state.episode_id},
        ).model_dump()
    )
    
    try:
        while True:
            data = json.loads(await websocket.receive_text())
            
            if data.get("reset"):
                obs = ws_env.reset()
                await websocket.send_json(
                    ResetResult(
                        observation=obs,
                        info={"episode_id": ws_env._state.episode_id},
                    ).model_dump()
                )
                continue
                
            action = ScalerAction(action=int(data.get("action", 0)))
            obs, reward, done, info = ws_env.step(action)
            s_reward = safe_score(reward)
            
            result = StepResult(observation=obs, reward=s_reward, done=done, info=info)
            await websocket.send_json(result.model_dump())
            
            # auto-reset on completion
            if done:
                obs = ws_env.reset()
                await websocket.send_json(
                    ResetResult(
                        observation=obs,
                        info={"episode_id": ws_env._state.episode_id},
                    ).model_dump()
                )
    except WebSocketDisconnect:
        log.info("WS client disconnected")
    except Exception as exc:
        log.exception(f"WS error: {exc}")
        await websocket.close(code=1011)


def main():
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)

if __name__ == "__main__":
    main()
