import pytest

from multi_agent_framework.routing import classify, extract_math
from multi_agent_framework.tools import safe_calculate


def test_router_keeps_easy_math_fast():
    assert classify("What is (81 / 9) + 4?") == "fast"
    assert safe_calculate(extract_math("What is (81 / 9) + 4?") or "") == 13


def test_research_uses_planner():
    assert classify("Research current approaches and cite sources") == "planner"
    assert classify("hello", force_plan=True) == "planner"


def test_calculator_rejects_code_execution():
    with pytest.raises(ValueError):
        safe_calculate("__import__('os').system('whoami')")
