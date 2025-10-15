import pytest
from typing import Dict, List, Any

from lattice_weaver.fibration.fibration_search_solver import FibrationSearchSolver
from lattice_weaver.fibration.constraint_hierarchy import ConstraintHierarchy, ConstraintLevel, Constraint, Hardness
from lattice_weaver.fibration.energy_landscape_optimized import EnergyLandscapeOptimized
from lattice_weaver.fibration.hacification_engine import HacificationEngine
from lattice_weaver.arc_engine.core import ArcEngine
from lattice_weaver.homotopy.rules import HomotopyRules

# Helper function for basic constraints
def ne_constraint(assignment: Dict[str, Any]):
    var1, var2 = list(assignment.keys())
    return assignment[var1] != assignment[var2]
def eq_constraint(assignment: Dict[str, Any]):
    var1, var2 = list(assignment.keys())
    return assignment[var1] == assignment[var2]
def unary_constraint_gt_0(assignment: Dict[str, Any]):
    var = list(assignment.keys())[0]
    return assignment[var] > 0

@pytest.fixture
def basic_hierarchy():
    hierarchy = ConstraintHierarchy()
    hierarchy.add_local_constraint(var1="Q0", var2="Q1", predicate=ne_constraint, hardness=Hardness.HARD, metadata={"name": "Q0_ne_Q1"})
    hierarchy.add_local_constraint(var1="Q1", var2="Q2", predicate=ne_constraint, hardness=Hardness.HARD, metadata={"name": "Q1_ne_Q2"})
    hierarchy.add_unary_constraint(variable="Q0", predicate=unary_constraint_gt_0, hardness=Hardness.HARD, metadata={"name": "Q0_gt_0"})
    return hierarchy

@pytest.fixture
def basic_domains():
    return {
        "Q0": [0, 1, 2],
        "Q1": [0, 1, 2],
        "Q2": [0, 1, 2]
    }

@pytest.fixture
def fibration_solver_with_homotopy(basic_domains, basic_hierarchy):
    return FibrationSearchSolver(
        variables=list(basic_domains.keys()),
        domains=basic_domains,
        hierarchy=basic_hierarchy,
        use_homotopy=True
    )

@pytest.fixture
def fibration_solver_without_homotopy(basic_domains, basic_hierarchy):
    return FibrationSearchSolver(
        variables=list(basic_domains.keys()),
        domains=basic_domains,
        hierarchy=basic_hierarchy,
        use_homotopy=False
    )

def test_fibration_solver_initialization_with_homotopy(fibration_solver_with_homotopy):
    assert fibration_solver_with_homotopy.homotopy_rules is not None
    assert isinstance(fibration_solver_with_homotopy.homotopy_rules, HomotopyRules)

def test_fibration_solver_initialization_without_homotopy(fibration_solver_without_homotopy):
    assert fibration_solver_without_homotopy.homotopy_rules is None

def test_homotopy_rules_precomputation_on_solve(fibration_solver_with_homotopy):
    # Call solve to trigger precomputation
    fibration_solver_with_homotopy.solve(time_limit_seconds=0.1)
    assert fibration_solver_with_homotopy.homotopy_rules._precomputed is True
    # Check if some rules were generated (e.g., dependency graph nodes)
    assert len(fibration_solver_with_homotopy.homotopy_rules.dependency_graph.nodes) > 0

def test_select_next_variable_with_homotopy(fibration_solver_with_homotopy, basic_domains):
    # Ensure homotopy rules are precomputed
    fibration_solver_with_homotopy.solve(time_limit_seconds=0.1)
    
    assignment = {}
    # With homotopy, the order should be Q0, Q1, Q2 based on frequency (Q0 and Q1 have 2 constraints, Q2 has 1)
    # If frequencies are equal, the order might depend on internal sorting, but Q0/Q1 should come before Q2
    next_var = fibration_solver_with_homotopy._select_next_variable(assignment)
    assert next_var in ["Q0", "Q1"]

    assignment = {"Q0": 0}
    next_var = fibration_solver_with_homotopy._select_next_variable(assignment)
    assert next_var in ["Q1", "Q2"]

def test_select_next_variable_without_homotopy(fibration_solver_without_homotopy, basic_domains):
    assignment = {}
    next_var = fibration_solver_without_homotopy._select_next_variable(assignment)
    # Without homotopy, it should fall back to default MRV, which for this simple case
    # might be Q0 or Q1 (both have 2 constraints, Q2 has 1)
    assert next_var in ["Q0", "Q1"]

    assignment = {"Q0": 0}
    next_var = fibration_solver_without_homotopy._select_next_variable(assignment)
    assert next_var in ["Q1", "Q2"]

def test_get_ordered_domain_values_with_homotopy(fibration_solver_with_homotopy, basic_domains):
    # Ensure homotopy rules are precomputed
    fibration_solver_with_homotopy.solve(time_limit_seconds=0.1)

    assignment = {"Q0": 1}
    ordered_values = fibration_solver_with_homotopy._get_ordered_domain_values("Q1", assignment)
    # Q1 != Q0 (1). So 1 should be filtered out. Remaining [0, 2].
    # Energy for Q1=0: Q0=1, Q1=0 -> total_energy = 0 (Q0!=Q1 satisfied)
    # Energy for Q1=2: Q0=1, Q1=2 -> total_energy = 0 (Q0!=Q1 satisfied)
    # The order between 0 and 2 might depend on internal sorting of equal energy values.
    assert 1 not in ordered_values
    assert len(ordered_values) == 2

def test_get_ordered_domain_values_without_homotopy(fibration_solver_without_homotopy, basic_domains):
    assignment = {"Q0": 1}
    ordered_values = fibration_solver_without_homotopy._get_ordered_domain_values("Q1", assignment)
    assert 1 not in ordered_values
    assert len(ordered_values) == 2

def test_fibration_solver_finds_solution(fibration_solver_with_homotopy):
    solution = fibration_solver_with_homotopy.solve(time_limit_seconds=1)
    assert solution is not None
    assert fibration_solver_with_homotopy.best_energy == 0.0 # For this simple problem, a 0-energy solution exists
    assert solution["Q0"] != solution["Q1"]
    assert solution["Q1"] != solution["Q2"]
    assert solution["Q0"] > 0

def test_fibration_solver_finds_solution_without_homotopy(fibration_solver_without_homotopy):
    solution = fibration_solver_without_homotopy.solve(time_limit_seconds=1)
    assert solution is not None
    assert fibration_solver_without_homotopy.best_energy == 0.0
    assert solution["Q0"] != solution["Q1"]
    assert solution["Q1"] != solution["Q2"]
    assert solution["Q0"] > 0

