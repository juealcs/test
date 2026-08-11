from adaptive_agent.router import classify_complexity


def test_easy_requests_take_fast_path():
    assert classify_complexity("What is 12 * 18?") == "fast"
    assert classify_complexity("Rewrite this sentence politely") == "fast"


def test_complex_requests_take_deliberate_path():
    assert classify_complexity("Research and compare three databases, then recommend one") == "deliberate"
    assert classify_complexity("Debug this application") == "deliberate"


def test_explicit_override_wins():
    assert classify_complexity("Design a system", "fast") == "fast"

