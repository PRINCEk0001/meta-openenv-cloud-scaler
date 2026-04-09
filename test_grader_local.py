import requests
import json
import time

def test_grader():
    base_url = "http://localhost:7860"
    
    print("Testing /grader endpoint...")
    
    # 1. Reset env
    requests.post(f"{base_url}/reset", json={"task": "autoscaling_easy"})
    
    # 2. Query grader (should be low/baseline initially)
    resp = requests.post(f"{base_url}/grader", json={"task": "autoscaling_easy"})
    data = resp.json()
    print(f"Initial Grade: {data}")
    
    score = data['score']
    assert 0.0 < score < 1.0, f"Score {score} out of range!"
    assert score != 0.0 and score != 1.0, f"Score {score} is at boundary!"

    # 3. Do some steps
    for _ in range(5):
        requests.post(f"{base_url}/step", json={"action": 1})
        
    # 4. Query grader again
    resp = requests.post(f"{base_url}/grader", json={"task": "autoscaling_easy"})
    data = resp.json()
    print(f"Mid-episode Grade: {data}")
    
    score = data['score']
    assert 0.001 <= score <= 0.999, f"Score {score} out of strict range!"
    
    print("\n[OK] /grader endpoint verification PASSED")

if __name__ == "__main__":
    test_grader()
