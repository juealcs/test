import pytest

from adaptive_agent.tools import safe_calculate


def test_safe_calculate():
    assert safe_calculate("(12 + 3) * 2") == 30


def test_safe_calculate_rejects_code():
    with pytest.raises(ValueError):
        safe_calculate("__import__('os').getcwd()")

