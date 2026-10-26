from physics_engine import simulate_stabilization

def test_ai_stabilizes_faster_than_human():
    # Run AI simulation (2ms)
    ai_result = simulate_stabilization("Tesla", 2, 10)
    # Run Human simulation (200ms)
    human_result = simulate_stabilization("Tesla", 200, 10)
    
    # The AI's maximum tilt should be less than the Human's maximum tilt
    assert max(ai_result["tilt_array"]) < max(human_result["tilt_array"])
    print("Test passed: AI stabilization is mathematically superior.")

if __name__ == "__main__":
    test_ai_stabilizes_faster_than_human()