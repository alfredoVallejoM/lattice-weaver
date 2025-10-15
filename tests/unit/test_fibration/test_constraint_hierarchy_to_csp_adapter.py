import pytest
from typing import Dict, Any, List, Tuple, Callable, FrozenSet
from lattice_weaver.core.csp_problem import CSP, Constraint as CSPConstraint
from lattice_weaver.fibration import (
    ConstraintHierarchy,
    Constraint,
    ConstraintLevel,
    Hardness,
    ConstraintHierarchyToCSPAdapter
)

class TestConstraintHierarchyToCSPAdapter:
    """Tests para la clase ConstraintHierarchyToCSPAdapter."""

    def test_convert_hierarchy_to_csp_simple(self):
        """
        Test: Convertir una ConstraintHierarchy simple a CSP.
        """
        hierarchy = ConstraintHierarchy()
        variables_domains = {
            "A": [1, 2],
            "B": [1, 2]
        }

        # Restricción HARD: A != B
        hierarchy.add_constraint(Constraint(
            level=ConstraintLevel.LOCAL,
            variables=("A", "B"),
            predicate=lambda assignment: (assignment["A"] != assignment["B"], 0.0 if assignment["A"] != assignment["B"] else 1.0),
            hardness=Hardness.HARD,
            metadata={"name": "A_neq_B"}
        ))

        # Restricción SOFT: A == 1 (debería ser ignorada por el adaptador CSP)
        hierarchy.add_constraint(Constraint(
            level=ConstraintLevel.LOCAL,
            variables=("A",),
            predicate=lambda assignment: (assignment["A"] == 1, 0.0 if assignment["A"] == 1 else 1.0),
            hardness=Hardness.SOFT,
            metadata={"name": "A_eq_1"}
        ))

        adapter = ConstraintHierarchyToCSPAdapter()
        csp, metadata = adapter.convert_hierarchy_to_csp(hierarchy, variables_domains)

        assert isinstance(csp, CSP)
        assert set(csp.variables) == {"A", "B"}
        assert csp.domains["A"] == frozenset({1, 2})
        assert csp.domains["B"] == frozenset({1, 2})

        # Solo la restricción HARD debería ser convertida
        assert len(csp.constraints) == 1
        csp_constraint = csp.constraints[0]

        assert frozenset(csp_constraint.scope) == frozenset({"A", "B"})
        assert csp_constraint.name == "A_neq_B"

        # Verificar la relación de la restricción CSP
        assert csp_constraint.relation(1, 2) is True
        assert csp_constraint.relation(2, 1) is True
        assert csp_constraint.relation(1, 1) is False

        # Verificar metadatos
        assert metadata["original_variables"] == ["A", "B"]

    def test_convert_hierarchy_solution_to_csp_solution_identity(self):
        """
        Test: La descompilación debe devolver la solución tal cual en este adaptador simple.
        """
        adapter = ConstraintHierarchyToCSPAdapter()
        csp_solution = {"var1": 10, "var2": "test"}
        metadata = {"original_variables": ["var1", "var2"]}
        hierarchy_solution = adapter.convert_csp_solution_to_hierarchy_solution(csp_solution, metadata)
        assert hierarchy_solution == csp_solution

    def test_empty_hierarchy(self):
        """
        Test: Convertir una ConstraintHierarchy vacía a CSP.
        """
        hierarchy = ConstraintHierarchy()
        variables_domains = {"X": [1, 2]}
        adapter = ConstraintHierarchyToCSPAdapter()
        csp, metadata = adapter.convert_hierarchy_to_csp(hierarchy, variables_domains)

        assert isinstance(csp, CSP)
        assert set(csp.variables) == {"X"}
        assert csp.domains["X"] == frozenset({1, 2})
        assert not csp.constraints

    def test_hierarchy_with_only_soft_constraints(self):
        """
        Test: Convertir una ConstraintHierarchy con solo restricciones SOFT a CSP.
        Debería resultar en un CSP sin restricciones.
        """
        hierarchy = ConstraintHierarchy()
        variables_domains = {"X": [1, 2]}
        hierarchy.add_constraint(Constraint(
            level=ConstraintLevel.LOCAL,
            variables=("X",),
            predicate=lambda assignment: assignment["X"] == 1,
            hardness=Hardness.SOFT,
            metadata={"name": "X_eq_1_soft"}
        ))

        adapter = ConstraintHierarchyToCSPAdapter()
        csp, metadata = adapter.convert_hierarchy_to_csp(hierarchy, variables_domains)

        assert isinstance(csp, CSP)
        assert set(csp.variables) == {"X"}
        assert csp.domains["X"] == frozenset({1, 2})
        assert not csp.constraints

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

