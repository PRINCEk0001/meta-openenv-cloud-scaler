"""
Pydantic schemas for the Cloud AutoScaler env AND the WhyDidItFail env.
Falls back to base pydantic models if openenv-core isn't installed yet.
"""

from typing import Literal, Optional, Any, Union
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
    observation: Union[ScalerObservation, CodeReviewObservation, Any]
    reward: float
    done: bool
    info: dict = Field(default_factory=dict)

class ResetResult(BaseModel):
    observation: Union[ScalerObservation, CodeReviewObservation, Any]
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

class GraderRequest(BaseModel):
    task: str = "autoscaling_easy"

class GraderResponse(BaseModel):
    task: str
    score: float
    is_success: bool


# ── WhyDidItFail env models ───────────────────────────────────────────────────

class WhyDidItFailAction(BaseModel):
    """
    Action submitted by the agent inside a WhyDidItFail episode.

    action_type options:
        "inspect_logs"       — examine training loss / accuracy curves
        "inspect_config"     — examine hyperparameter config
        "inspect_gradients"  — examine gradient norm statistics
        "submit_diagnosis"   — submit final diagnosis (ends episode)

    The three diagnosis fields are only required when action_type == "submit_diagnosis".
    """
    action_type: Literal[
        "inspect_logs",
        "inspect_config",
        "inspect_gradients",
        "submit_diagnosis",
    ]
    diagnosis:     Optional[str] = None
    suggested_fix: Optional[str] = None
    reasoning:     Optional[str] = None


class WhyDidItFailObservation(BaseModel):
    """Observation returned after each step in a WhyDidItFail episode."""
    task_description: str
    feedback:         str
    visible_data:     Optional[dict] = None


class WhyDidItFailStepResult(BaseModel):
    """Full step result returned by the WhyDidItFail server."""
    observation: WhyDidItFailObservation
    reward:      float
    done:        bool
    info:        dict = Field(default_factory=dict)
