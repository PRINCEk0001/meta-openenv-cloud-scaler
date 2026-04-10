import time
from client import CloudAutoScalerEnv
from models import ScalerAction

LIVE_SPACE_URL = "https://kkoriprince90-meta-openenv-cloud-scaler.hf.space"

def run_demo():
    print(f"🌍 Connecting to Live Space: {LIVE_SPACE_URL}")
    
    # 1. Initialize the client to hit your Hugging Face Space
    env = CloudAutoScalerEnv(base_url=LIVE_SPACE_URL).sync()
    
    # 2. Reset the environment and start a new episode
    print("\n🔄 Resetting Environment...")
    obs = env.reset(task_name="autoscaling_easy")
    
    print(f"📊 Starting State: {obs.active_servers} servers running. Traffic loop: {obs.current_traffic_load:.2f} req/s\n")
    print(f"{'Step':<6} | {'Action':<15} | {'Servers':<8} | {'Latency':<10} | {'Reward':<8}")
    print("-" * 65)

    # 3. Simple Heuristic Agent Loop
    total_reward = 0.0
    
    for _ in range(10):  # Just run 10 steps for a quick demo
        time.sleep(0.5)  # Pause purely for visual effect in the terminal
        
        # Decide action based on traffic/capacity utilization
        if obs.utilization > 0.80:
            act_idx = 1  # Add a server
            act_name = "Scale UP 📈"
        elif obs.utilization < 0.60 and obs.active_servers > 1:
            act_idx = 2  # Remove a server
            act_name = "Scale DOWN 📉"
        else:
            act_idx = 0  # Do nothing
            act_name = "Hold 🟢"

        # Apply the action
        res = env.step(ScalerAction(action=act_idx))
        
        # Log the result
        print(f"{res.observation.step_number:<6} | {act_name:<15} | {res.observation.active_servers:<8} | {res.observation.latency_ms:>6.1f} ms | {res.reward:>6.2f}")
        
        total_reward += float(res.reward)
        obs = res.observation

    print("-" * 65)
    print(f"🏁 Demo Complete. Total Reward after 10 steps: {total_reward:.2f}")

if __name__ == "__main__":
    run_demo()
