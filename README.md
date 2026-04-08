---
title: Cloud AutoScaler Env
emoji: ☁️
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# Cloud Server Auto-Scaler Environment

Hey! Here is my submission for the OpenEnv track. It's a cloud server farm simulation where an RL agent has to quickly spin up or spin down servers in response to traffic spikes, trying to keep costs low without crashing the service.

## The Challenge

The environment simulates a base sine wave of traffic along with some randomized spikes. The agent gets continuous metrics (latency, current load, utilization) and has to pick one of three things every step:
- Do nothing (hold)
- Spin up a new server
- Terminate a server

It's basically a balancing act: if you over-provision, you get penalized for the cost. If you under-provision and let latency spike (or crash entirely), the penalty is huge.

* Target: 60-80% utilization with latency under 50ms.

---

## Quick Start

### Testing the REST API

```bash
# reset the episode
curl -X POST http://localhost:7860/reset

# scale up by one server
curl -X POST http://localhost:7860/step \
     -H "Content-Type: application/json" \
     -d '{"action": 1}'

# check if it's alive
curl http://localhost:7860/health
```

### Try the WebSocket

If you prefer websockets, connect to `/ws`. Here's a quick snippet:

```python
import asyncio, json, websockets

async def play():
    async with websockets.connect("ws://localhost:7860/ws") as ws:
        obs = json.loads(await ws.recv())
        print("Reset:", obs)
        await ws.send(json.dumps({"action": 1}))
        print("Step:", json.loads(await ws.recv()))

asyncio.run(play())
```

### Or just run the included client

```bash
# This uses a simple heuristic agent to run through 50 steps
python client.py --mode http --host http://localhost:7860 --steps 50
```

---

## Simulation Details

- **Episode length:** 200 steps
- **Start state:** 10 servers sitting around ~250 req/s load.
- **Server limit:** Each server caps at exactly 25 requests per second.
- **Traffic:** Oscillating mostly between 250 and 750, with random 400 req/s spikes hitting roughly 20% of the time every 5th step.

### Rewards (normalized to [-1.0, +1.0])
- **Healthy:** +1.0 for latency < 50ms (minus efficiency penalty)
- **Degraded:** +0.6 for latency < 150ms
- **Bad:** +0.3 for latency < 500ms
- **Outage:** -1.0 penalty if latency >= 500ms
- **Over-provisioning:** up to -0.2 deducted based on server count

### Latency Modeling
- **< 70% load:** 20-40ms (Optimal)
- **70-90% load:** 50-150ms (Acceptable)
- **90-100% load:** 200-400ms (Degraded)
- **> 100% load:** 600-1000ms (Outage)

---

## Running it locally

```bash
docker build -t cloud-env .
docker run -p 7860:7860 cloud-env
```

Or raw python:

```bash
pip install -r server/requirements.txt
python -m uvicorn server.app:app --host 0.0.0.0 --port 7860
```
