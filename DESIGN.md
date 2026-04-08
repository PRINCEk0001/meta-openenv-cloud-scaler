# Design Decisions

Here is a quick brain-dump of the architecture and tuning decisions behind this environment.

### 1. Episode Sizing
I went with 50 steps. Assuming a 6-second polling interval in a real autoscaler, this simulates roughly 5 minutes—enough time to observe the onset and decay of a massive traffic spike. I tried 100 steps initially but making agents iterate over that took too long during local testing, and 25 wasn't wide enough to capture the full sine wave logic anyway.

### 2. The Traffic & Math
The base traffic is a sine wave oscillating around 500 req/s, peaking at 750. I added random spikes of ~300 req/s happening probabilistically every 5 steps. This makes it impossible for the agent to just memorize a timeline. They have to actually monitor utilization and keep some buffering. The exact breakpoint is tuned so that a spike hitting exactly on a sine peak forces the agent into utilizing nearly all 50 possible servers to avoid a crash.

### 3. Server Capacities
To keep numbers whole and easy to watch, I pinned capacity at 25 req/s per server. 
I start the simulation with 10 servers active. Since the baseline traffic generally sits around 250 requests at step 0, starting with 10 gives the agent 100% exact capacity (250 / (10 * 25) = 1.0 utilization). It forces an immediate reaction.

### 4. Reward Shaping
I wanted to build an objective that actually mimics modern SLA constraints:
- You get +1 for keeping latency sub-50ms.
- You lose -0.04 continually for every server you boot up. (This forces scale-downs).
- A hard -10.0 penalty for outaging (latency > 500ms).

You break even around 25 servers. Anything more than that is costing you significantly, but if you skimp out during a traffic surge, the -10.0 penalty completely tanks your episode return.

### 5. Why FastAPI?
OpenEnv supports it out of the box, and the pydantic integration makes typing the inputs automatic. I used a custom shim wrapper so I can mock out the `openenv.core` package classes when running the build locally before fetching the core package. 

### What's next if I had more time?
- Startup delays (servers taking a few steps to come online).
- Mixed fleets, simulating Spot and On-Demand instances.
- Actually shedding requests to degrade gracefully rather than just spiking latency globally.
