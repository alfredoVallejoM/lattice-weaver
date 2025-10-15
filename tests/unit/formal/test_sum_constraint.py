import pytest
from lattice_weaver.core.csp_problem import SumConstraint


def test_sum_constraint_init():
    """Tests the initialization of the SumConstraint class."""
    sc = SumConstraint(scope=frozenset(["x", "y"]), target_sum=10)
    assert sc.target_sum == 10
    assert sc.scope == frozenset(["x", "y"])
    assert sc.name == "Sum({x, y}) == 10"
    assert callable(sc.relation)

def test_sum_constraint_relation_satisfied():
    """Tests the relation of SumConstraint when satisfied."""
    sc = SumConstraint(scope=frozenset(["x", "y"]), target_sum=10)
    assert sc.relation(3, 7) is True
    assert sc.relation(5, 5) is True

def test_sum_constraint_relation_not_satisfied():
    """Tests the relation of SumConstraint when not satisfied."""
    sc = SumConstraint(scope=frozenset(["x", "y"]), target_sum=10)
    assert sc.relation(3, 6) is False
    assert sc.relation(5, 6) is False

def test_sum_constraint_with_custom_name():
    """Tests SumConstraint with a custom name."""
    sc = SumConstraint(scope=frozenset(["a", "b", "c"]), target_sum=15, name="MySumConstraint")
    assert sc.name == "MySumConstraint"
    assert sc.relation(5, 5, 5) is True
    assert sc.relation(1, 2, 3) is False

def test_sum_constraint_metadata():
    """Tests SumConstraint with metadata."""
    metadata = {"priority": 1, "type": "arithmetic"}
    sc = SumConstraint(scope=frozenset(["v1", "v2"]), target_sum=0, metadata=metadata)
    assert sc.metadata == metadata

