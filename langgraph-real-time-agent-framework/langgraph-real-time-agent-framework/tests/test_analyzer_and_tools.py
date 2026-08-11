import pytest

from real_agent_framework.services.problem_analyzer import ProblemAnalyzer
from real_agent_framework.tools import safe_calculate


def test_problem_analyzer_uses_fast_route_for_easy_work():
    analyzer = ProblemAnalyzer()
    assert analyzer.classify("Hello") == "simple"
    assert analyzer.classify("Research current evidence and cite sources") == "planned"
    assert analyzer.classify("Hello", force_plan=True) == "planned"


def test_calculator_is_allowlisted():
    assert safe_calculate("(81 / 9) + 4") == 13
    with pytest.raises(ValueError):
        safe_calculate("__import__('os').system('whoami')")
