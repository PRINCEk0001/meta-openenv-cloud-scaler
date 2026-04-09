---
title: Cloud AutoScaler Env
emoji: ☁️
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# ☁️ Cloud AutoScaler Environment

[![Gymnasium Compliant](https://img.shields.io/badge/Gymnasium-v0.29.1-blue.svg)](https://gymnasium.farama.org/)
[![FastAPI Server](https://img.shields.io/badge/FastAPI-0.110.0-teal.svg)](https://fastapi.tiangolo.com)
[![Meta OpenEnv](https://img.shields.io/badge/Track-Meta_OpenEnv-purple.svg)]()

Welcome to the **Cloud AutoScaler Environment**, a fully compliant Gymnasium simulation designed for the **Meta OpenEnv Track** hackathon. 

This environment simulates a core dev-ops challenge: managing a cloud server farm where an RL agent or heuristic controller must dynamically spin servers up and down in response to volatile web traffic. The agent must balance infrastructure **costs** against user **latency**, strictly avoiding critical service outages while minimizing over-provisioning.

---

## 🎯 The Challenge

The environment simulates a baseline sine wave of HTTP traffic, overlaid with randomized, high-volume traffic spikes. The agent receives continuous, real-time metrics (latency, current load, utilization) and must select one of three actions per step:

1. **Hold** (`0`) — Maintain the current server count.
2. **Scale Up** (`1`) — Provision an additional server.
3. **Scale Down** (`2`) — Deprovision a server.

**Optimization Target:** Maintain a strict 60-80% capacity utilization with latency strictly under 50ms, without overspending on idle infrastructure.

---

## 🏗️ Architecture & Requirements

This repository is strictly structured to support both local programmatic benchmarking and the **Meta OpenEnv Automated Evaluator** via Hugging Face Spaces.

### OpenEnv Compliance Features
* **Gymnasium v0.29.1 standard:** Implements a strict `CloudScalerEnv(gym.Env)` class.
* **Tuple returns:** Standardized `(obs, info)` for resets and `(obs, reward, terminated, truncated, info)` for steps.
* **Bounded execution:** Hard limits capped at `MAX_STEPS = 50` to prevent infinite loops (`truncated=True`).
* **Normalized Rewards:** Strictly clamped to the open interval `(0.001, 0.999)`.
* **Standardized Metadata:** The `info` dictionary strictly tracks `is_success`, `latency_ms`, `active_servers`, and `step_count`.

---

## 📊 Simulation Dynamics

* **Episode Horizon:** 50 steps
* **Initial State:** 10 active servers handling a baseline ~250 req/s load.
* **Capacity Limit:** Each server is capped at precisely 25 requests per second.
* **Traffic Volatility:** Traffic naturally oscillates between 250 and 750 req/s. Spike events (e.g. +400 req/s) occur randomly 20% of the time, generally every 5th step.

### Reward Function (Clamped `(0.001, 0.999)`)
* **Healthy (`0.97` base):** System latency < 50ms.
* **Degraded (`0.60` base):** System latency 50ms - 150ms.
* **Poor (`0.30` base):** System latency 150ms - 500ms.
* **Critical Outage (`0.02` base):** System latency > 500ms (Hard failure).
* *Note: Up to `-0.20` is dynamically deducted from the base score to penalize capacity over-provisioning (idle servers), with a final hard clamp at `0.001`.*

### Latency Modeling 
* **< 70% load:** 20-40ms (Optimal)
* **70-90% load:** 50-150ms (Acceptable)
* **90-100% load:** 200-400ms (Degraded)
* **> 100% load:** 600-1000ms (Outage)

---

## 🚀 Quick Start & Usage

This environment natively supports both headless Python invocation and headless REST/WebSocket execution via Docker.

### 1. Direct Python Execution (Evaluator Standard)
You can directly import the environment via the standard Hugging Face/OpenEnv topology:

```python
import env as open_env
import numpy as np

# Instantiate the standard Gymnasium environment
env = open_env.CloudScalerEnv()
obs, info = env.reset()

# Sample a random scaling action
action = env.action_space.sample()

# Execute timestep
obs, reward, terminated, truncated, info = env.step(action)
print(f"Latency: {info['latency_ms']}ms | Active Servers: {info['active_servers']}")
```

### 2. REST API / Docker Execution
When deployed on Hugging Face Spaces (or locally via Docker), the environment automatically serves a FastAPI wrapper on port `7860`.

```bash
# Build and run the container locally
docker build -t meta-openenv-cloud-scaler .
docker run -p 7860:7860 meta-openenv-cloud-scaler
```

**Testing the HTTP Endpoints:**
```bash
# 1. Reset the simulation episode
curl -X POST http://localhost:7860/reset

# 2. Scale up by provisioning one new server
curl -X POST http://localhost:7860/step \
     -H "Content-Type: application/json" \
     -d '{"action": 1}'

# 3. Verify server health
curl http://localhost:7860/health
```

### 3. Integrated Client Testing
A fully typed heuristic client (`client.py`) and a demo script (`demo.py`) are included to quickly test the live endpoints. 

```bash
# Run 10 steps using the live Hugging Face Space URL
pip install httpx pydantic
python demo.py
```

---

## 🤖 Agentic Evaluation & Variance Check

This environment is fully prepared for automated evaluation by Open LLM Agents (e.g., Nemotron 3 Super). 

### Baseline Performance (Heuristic)
Running the baseline standard agent (`client.py` using simple utiliziation threshold heuristics) over 50 steps typically yields a **Total Reward between ~35.0 and ~45.0**. Score variance strictly depends on the randomized +400 req/s traffic spikes.

### LLM Agent Variance & Readiness
- **Docstring Prompts**: The environment is equipped with rich `__doc__` strings in the `CloudScalerEnv` class, explicitly detailing the `[latency, cost, capacity]` trade-offs as expected by zero-shot LLM prompts.
- **Score Variance**: Agent evaluation pipelines should expect a baseline variance margin of `±10.0` points dynamically introduced by the pseudo-random sine-wave traffic injection points. Open LLM agents are evaluated against their ability to preempt these spikes while optimizing over-provisioning efficiency penalties.

---
*Developed for the Meta LLM OpenEnv Hackathon* ☁️
