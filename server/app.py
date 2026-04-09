"""
server/app.py — FastAPI app for the Cloud AutoScaler env.

We route this locally using standard endpoints so we can inject 
"task" configs via the HTTP payloads exactly as needed.
"""

import json
import logging
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from models import EnvInfo, ResetResult, ScalerAction, ScalerObservation, StepResult, GraderRequest, GraderResponse, CodeReviewAction, CodeReviewObservation
from server.environment import CloudAutoScalerEnvironment, CodeReviewEnvironment
from server.tasks import grade_task

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
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
    
    # Logging based on env type
    if isinstance(obs, CodeReviewObservation):
        log.info(f"Resetting CodeReview ({task_name}) -> step={obs.step_number}")
    else:
        log.info(f"Resetting CloudScaler ({task_name}) -> servers={obs.active_servers}")
        
    return ResetResult(observation=obs, info={"episode_id": ep_id})


@app.post("/step", response_model=StepResult)
async def step(action: Union[ScalerAction, CodeReviewAction, Any]):
    global _env_instance
    if _env_instance is None:
        return {"error": "Call /reset first"}
        
    obs, reward, done, info = _env_instance.step(action)
    
    # Generic logging
    log.info(f"step={getattr(obs, 'step_number', '?')} reward={reward:+.3f} done={done}")
    
    return StepResult(observation=obs, reward=reward, done=done, info=info)
    

@app.post("/grader", response_model=GraderResponse)
async def grader(req: GraderRequest):
    global _env_instance
    try:
        if _env_instance is None or _env_instance._state is None:
            log.warning(f"Grader called for {req.task} but no environment state is available.")
            return GraderResponse(task=req.task, score=0.10, is_success=False)
        
        raw_score = grade_task(req.task, _env_instance._state)
        
        # Ultra-strict clamp as suggested by user, now with [0.001, 0.999] bounds
        score = float(raw_score)
        score = max(0.001, min(0.999, score))
        score = round(score, 3)
            
        is_success = bool(score >= 0.5)
        log.info(f"Grading ({req.task}) -> Raw={raw_score:.4f}, Final={score:.2f}, success={is_success}")
        return GraderResponse(task=req.task, score=score, is_success=is_success)
        
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
            
            result = StepResult(observation=obs, reward=reward, done=done, info=info)
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
