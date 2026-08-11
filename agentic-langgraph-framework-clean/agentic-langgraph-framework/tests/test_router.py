from agentic_framework.router import classify, extract_math
from agentic_framework.tools import safe_calculate


def test_math_uses_deterministic_path():
    assert classify("What is (81 / 9) + 4?") == "deterministic"
    assert safe_calculate(extract_math("What is (81 / 9) + 4?")) == 13


def test_tool_request_uses_agent_path():
    assert classify("Read the file notes.txt") == "agent"


def test_code_execution_is_rejected():
    try:
        safe_calculate("__import__('os').system('whoami')")
    except ValueError:
        pass
    else:
        raise AssertionError("unsafe expression was accepted")
