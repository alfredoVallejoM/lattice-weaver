import pytest
from typing import Dict, Any, List, Tuple, Callable, FrozenSet
from lattice_weaver.core.csp_problem import CSP, Constraint as CSPConstraint
from lattice_weaver.fibration import (
    ConstraintHierarchy,
    Constraint,
    ConstraintLevel,
    Hardness,
    CSPToConstraintHierarchyAdapter
)

class TestCSPToConstraintHierarchyAdapter:
    """Tests para la clase CSPToConstraintHierarchyAdapter."""

    def test_convert_simple_csp_to_hierarchy(self):
        """
        Test: Convertir un CSP simple (N-Reinas 2x2) a ConstraintHierarchy.
        Un problema 2x2 de N-Reinas no tiene solución, pero la conversión debe ser correcta.
        """
        # 1. Definir un CSP simple (N-Reinas 2x2)
        variables = {"Q0", "Q1"}
        domains = {"Q0": frozenset({0, 1}), "Q1": frozenset({0, 1})}

        constraints = []
        # Restricción de columna (implícita por el dominio)
        # Restricción de fila (implícita por el dominio)

        # Restricción de diagonal (Q0, Q1)
        constraints.append(CSPConstraint(
            scope=frozenset({"Q0", "Q1"}),
            relation=lambda q0, q1: abs(q0 - q1) != abs(0 - 1), # abs(0-1) = 1
            name="diag_Q0_Q1"
        ))
        # Restricción de columna (Q0, Q1) - no pueden estar en la misma columna
        constraints.append(CSPConstraint(
            scope=frozenset({"Q0", "Q1"}),
            relation=lambda q0, q1: q0 != q1,
            name="col_Q0_Q1"
        ))

        csp = CSP(variables=variables, domains=domains, constraints=constraints, name="NQueens2x2")

        # 2. Convertir CSP a ConstraintHierarchy
        adapter = CSPToConstraintHierarchyAdapter()
        hierarchy, fibration_domains, metadata = adapter.convert_csp_to_hierarchy(csp)

        # 3. Verificar la jerarquía y los dominios convertidos
        assert isinstance(hierarchy, ConstraintHierarchy)
        assert isinstance(fibration_domains, dict)
        assert isinstance(metadata, dict)

        assert set(fibration_domains.keys()) == variables
        assert fibration_domains["Q0"] == [0, 1]
        assert fibration_domains["Q1"] == [0, 1]

        # Debería haber 2 restricciones en el nivel LOCAL
        local_constraints = hierarchy.get_constraints_by_level(ConstraintLevel.LOCAL)
        assert len(local_constraints) == 2

        # Verificar una de las restricciones convertidas
        diag_constraint = next(c for c in local_constraints if c.metadata.get("original_constraint_name") == "diag_Q0_Q1")
        assert diag_constraint.level == ConstraintLevel.LOCAL
        assert diag_constraint.hardness == Hardness.HARD
        assert set(diag_constraint.variables) == {"Q0", "Q1"}

        # Verificar la evaluación de la restricción convertida
        # Asignación que viola la restricción diagonal (Q0=0, Q1=1) -> abs(0-1) = 1, abs(0-1) = 1. Son iguales, viola.
        satisfied, violation = diag_constraint.evaluate({"Q0": 0, "Q1": 1})
        assert not satisfied
        assert violation == 1.0

        # Asignación que satisface la restricción diagonal (Q0=0, Q1=0) -> abs(0-0) = 0, abs(0-1) = 1. No son iguales, satisface.
        # Pero esta viola la restricción de columna. La restricción diagonal en sí misma estaría satisfecha.
        satisfied, violation = diag_constraint.evaluate({"Q0": 0, "Q1": 0})
        assert satisfied
        assert violation == 0.0

        # 4. Simular una solución de Fibration Flow (que no existe para N-Reinas 2x2, pero para probar la descompilación)
        fibration_solution = {"Q0": 0, "Q1": 0}

        # 5. Descompilar la solución
        csp_solution = adapter.convert_hierarchy_solution_to_csp_solution(fibration_solution, metadata)

        # 6. Verificar la descompilación
        assert csp_solution == {"Q0": 0, "Q1": 0}

    def test_convert_csp_with_different_domains(self):
        """
        Test: Convertir un CSP con dominios de diferentes tipos.
        """
        variables = {"A", "B"}
        domains = {"A": frozenset({"red", "blue"}), "B": frozenset({10, 20})}
        constraints = [
            CSPConstraint(
                scope=frozenset({"A", "B"}),
                relation=lambda a, b: (a == "red" and b == 10) or (a == "blue" and b == 20),
                name="color_value_match"
            )
        ]
        csp = CSP(variables=variables, domains=domains, constraints=constraints)

        adapter = CSPToConstraintHierarchyAdapter()
        hierarchy, fibration_domains, metadata = adapter.convert_csp_to_hierarchy(csp)

        assert fibration_domains["A"] == ["blue", "red"]
        assert fibration_domains["B"] == [10, 20]

        local_constraints = hierarchy.get_constraints_by_level(ConstraintLevel.LOCAL)
        assert len(local_constraints) == 1
        constraint = local_constraints[0]

        # Test evaluation
        satisfied, _ = constraint.evaluate({"A": "red", "B": 10})
        assert satisfied
        satisfied, _ = constraint.evaluate({"A": "blue", "B": 20})
        assert satisfied
        satisfied, violation = constraint.evaluate({"A": "red", "B": 20})
        assert not satisfied
        assert violation == 1.0

    def test_empty_csp(self):
        """
        Test: Convertir un CSP vacío.
        """
        csp = CSP(variables=set(), domains={}, constraints=[])
        adapter = CSPToConstraintHierarchyAdapter()
        hierarchy, fibration_domains, metadata = adapter.convert_csp_to_hierarchy(csp)

        assert isinstance(hierarchy, ConstraintHierarchy)
        assert all(not constraints for constraints in hierarchy.get_all_constraints(only_non_empty=True).values())
        assert not fibration_domains
        assert not metadata["original_variables"]

    def test_csp_with_unary_constraint(self):
        """
        Test: Convertir un CSP con una restricción unaria.
        """
        variables = {"X"}
        domains = {"X": frozenset({1, 2, 3})}
        constraints = [
            CSPConstraint(
                scope=frozenset({"X"}),
                relation=lambda x: x > 1,
                name="X_greater_than_1"
            )
        ]
        csp = CSP(variables=variables, domains=domains, constraints=constraints)

        adapter = CSPToConstraintHierarchyAdapter()
        hierarchy, fibration_domains, metadata = adapter.convert_csp_to_hierarchy(csp)

        local_constraints = hierarchy.get_constraints_by_level(ConstraintLevel.LOCAL)
        assert len(local_constraints) == 1
        constraint = local_constraints[0]

        satisfied, _ = constraint.evaluate({"X": 2})
        assert satisfied
        satisfied, violation = constraint.evaluate({"X": 1})
        assert not satisfied
        assert violation == 1.0

    def test_decompile_solution_identity(self):
        """
        Test: La descompilación debe devolver la solución tal cual en este adaptador simple.
        """
        adapter = CSPToConstraintHierarchyAdapter()
        fibration_solution = {"var1": 10, "var2": "test"}
        metadata = {"original_variables": ["var1", "var2"]}
        csp_solution = adapter.convert_hierarchy_solution_to_csp_solution(fibration_solution, metadata)
        assert csp_solution == fibration_solution

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

