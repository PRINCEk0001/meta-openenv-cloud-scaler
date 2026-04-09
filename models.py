"""
Pydantic schemas for the Cloud AutoScaler env.
Falls back to base pydantic models if openenv-core isn't installed yet.
"""

from typing import Literal
from pydantic import BaseModel, Field

try:
    from openenv.core.models import Action as _Action
    from openenv.core.models import Observation as _Observation
    from openenv.core.models import State as _State
    _OPENENV_AVAILABLE = True
except ImportError:
    _Action = BaseModel        # type: ignore
    _Observation = BaseModel   # type: ignore
    _State = BaseModel         # type: ignore
    _OPENENV_AVAILABLE = False


class ScalerAction(_Action):
    action: Literal[0, 1, 2] = Field(..., description="0: hold, 1: add server, 2: remove server")

class ScalerObservation(_Observation):
    current_traffic_load: float = Field(..., ge=0, description="Current req/s")
    active_servers: int = Field(..., ge=0, le=50)
    latency_ms: float = Field(..., ge=0)
    step_number: int = Field(..., ge=0)
    total_capacity: float = Field(..., description="Max traffic capacity (req/s)")
    utilization: float = Field(..., ge=0)

class ScalerState(_State):
    episode_id: str
    step_count: int = 0
    total_reward: float = 0.0
    peak_traffic: float = 0.0
    avg_latency: float = 0.0

# API response wrappers for the FastAPI server
class StepResult(BaseModel):
    observation: ScalerObservation
    reward: float
    done: bool
    info: dict = Field(default_factory=dict)

class ResetResult(BaseModel):
    observation: ScalerObservation
    info: dict = Field(default_factory=dict)

class EnvInfo(BaseModel):
    name: str = "cloud-autoscaler-env"
    version: str = "1.0.0"
    description: str = "Cloud auto-scaling environment"
    action_space: dict = {"type": "discrete", "values": {0: "hold", 1: "add", 2: "remove"}}
    observation_space: dict = {
        "current_traffic_load": "float",
        "active_servers": "int [1, 50]",
        "latency_ms": "float",
        "step_number": "int",
        "total_capacity": "float",
        "utilization": "float",
    }
    max_steps: int = 50
    reward_range: tuple = (0.01, 0.99)
    openenv_core_available: bool = _OPENENV_AVAILABLE
