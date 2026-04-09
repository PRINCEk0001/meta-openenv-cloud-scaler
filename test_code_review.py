"""
test_code_review.py — Unit tests for the Code Review trajectory grader.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from server.code_review_logic import grade_code_review_trajectory

def test_graders():
    print("Running Code Review Trajectory Grader Tests...")
    
    # 1. Catastrophic Failure Test (Any 0.10)
    # Mean: (0.9*4 + 0.1) / 5 = 3.7 / 5 = 0.74
    # Penalty (Hard): -0.60
    # Expected: 0.74 - 0.60 = 0.14 -> Clamped to 0.14
    score = grade_code_review_trajectory([0.9, 0.9, 0.9, 0.9, 0.1], "hard")
    print(f"  Test Catastrophic (Hard): Expected ~0.14, Got {score}")
    assert abs(score - 0.14) < 0.05
    
    # 2. Perfect Consistency Test
    # Mean: 0.9
    # Consistency Bonus (Easy): +0.05
    # Explanation Bonus: +0.05
    # Expected: 0.90 + 0.05 + 0.05 = 1.00 -> Clamped to 0.999
    score = grade_code_review_trajectory([0.9, 0.9, 0.9, 0.9, 0.9], "easy")
    print(f"  Test Perfect (Easy): Expected 0.999, Got {score}")
    assert score == 0.999
    
    # 3. Partial Consistency Test (4/5 steps >= 0.70)
    # Sequence: [0.7, 0.7, 0.7, 0.7, 0.3]
    # Mean: (2.8+0.3)/5 = 0.62
    # Consistency Bonus (Med): +0.10
    # Expected: 0.62 + 0.10 = 0.72
    score = grade_code_review_trajectory([0.7, 0.7, 0.7, 0.7, 0.3], "medium")
    print(f"  Test Partial (Med): Expected 0.72, Got {score}")
    assert abs(score - 0.72) < 0.01

    print("\n[SUCCESS] All Code Review trajectory tests passed!")

if __name__ == "__main__":
    test_graders()
