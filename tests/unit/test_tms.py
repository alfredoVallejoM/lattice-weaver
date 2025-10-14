import pytest
import time
from typing import Dict, Any, List, Set, Tuple

from lattice_weaver.core.csp_problem import CSP, Constraint
from lattice_weaver.core.csp_engine.solver import CSPSolver, CSPSolutionStats
from lattice_weaver.core.csp_engine.tms import TruthMaintenanceSystem, create_tms

# Helper function to create an N-Queens problem CSP
def create_nqueens_csp(n):
    variables = frozenset({f'Q{i}' for i in range(n)})
    domains = {var: frozenset(range(n)) for var in variables}
    constraints = []

    for i in range(n):
        for j in range(i + 1, n):
            constraints.append(Constraint(
                scope=frozenset({f'Q{i}', f'Q{j}'}),
                relation=lambda qi, qj, i=i, j=j: qi != qj and abs(qi - qj) != abs(i - j),
                name=f'neq_diag_Q{i}Q{j}'
            ))
    return CSP(variables=variables, domains=domains, constraints=frozenset(constraints), name=f"NQueens_{n}")


class TestTruthMaintenanceSystem:
    """Tests para Truth Maintenance System (TMS)."""

    def test_tms_basic(self):
        """Test: Funcionalidad básica del TMS."""
        tms = create_tms()

        # Registrar eliminaciones
        tms.record_removal("X", 1, "C1", {"Y": [2, 3]})
        tms.record_removal("X", 2, "C1", {"Y": [1]})
        tms.record_removal("Y", 3, "C2", {"X": [1, 2]})

        assert len(tms.justifications) == 3
        assert "X" in tms.dependency_graph
        assert "Y" in tms.dependency_graph

    def test_tms_explain_inconsistency(self):
        """Test: Explicación de inconsistencias."""
        tms = create_tms()

        # Simular eliminaciones que causan inconsistencia
        tms.record_removal("X", 1, "C1", {"Y": [2]})
        tms.record_removal("X", 2, "C2", {"Y": [1]})
        tms.record_removal("X", 3, "C1", {"Y": [3]})

        explanations = tms.explain_inconsistency("X")

        assert len(explanations) == 3
        assert any(exp.removed_value == 1 and exp.reason_constraint == "C1" for exp in explanations)

    def test_tms_suggest_constraint(self):
        """Test: Sugerencia de restricción a relajar."""
        tms = create_tms()

        # C1 causa 3 eliminaciones, C2 causa 1
        tms.record_removal("X", 1, "C1", {"Y": [2]})
        tms.record_removal("X", 2, "C1", {"Y": [1]})
        tms.record_removal("X", 3, "C1", {"Y": [3]})
        tms.record_removal("X", 4, "C2", {"Y": [1]})

        suggested = tms.suggest_constraint_to_relax("X")

        assert suggested == "C1"  # C1 causó más eliminaciones

    def test_tms_restorable_values(self):
        """Test: Identificación de valores restaurables."""
        tms = create_tms()

        # Registrar eliminaciones por C1
        tms.record_removal("X", 1, "C1", {"Y": [2]})
        tms.record_removal("X", 2, "C1", {"Y": [1]})
        tms.record_removal("Y", 3, "C1", {"X": [1]})

        # Registrar eliminaciones por C2
        tms.record_removal("Z", 5, "C2", {"W": [6]})

        # Obtener restaurables de C1
        restorable = tms.get_restorable_values("C1")

        assert "X" in restorable
        assert "Y" in restorable
        assert 1 in restorable["X"]
        assert 2 in restorable["X"]
        assert 3 in restorable["Y"]
        assert "Z" not in restorable # C2 removals should not be restored by C1

    def test_tms_with_csp_solver(self):
        """Test: Integración TMS con CSPSolver (indirecta)."""
        # El TMS es un componente separado que puede ser usado por el CSPSolver
        # o para analizar su comportamiento. Aquí, simplemente verificamos que
        # podemos crear un CSPSolver y un TMS de forma independiente.
        n = 4
        csp = create_nqueens_csp(n)
        solver = CSPSolver(csp)
        tms = create_tms()
        assert solver is not None
        assert tms is not None
        # Para una integración más profunda, se necesitaría modificar CSPSolver
        # para que use el TMS o para que el TMS pueda observar el CSPSolver.

    def test_tms_conflict_graph(self):
        """Test: Grafo de conflictos."""
        tms = create_tms()

        # Registrar eliminaciones
        tms.record_removal("X", 1, "C1", {"Y": [2], "Z": [3]})
        tms.record_removal("X", 2, "C2", {"Y": [1]})
        tms.record_removal("X", 3, "C1", {"Y": [3]})

        # Obtener grafo de conflictos
        conflict_graph = tms.get_conflict_graph("X")

        assert "C1" in conflict_graph
        assert "C2" in conflict_graph
        assert "Y" in conflict_graph["C1"]
        assert "Z" in conflict_graph["C1"]

    def test_tms_statistics(self):
        """Test: Estadísticas del TMS."""
        tms = create_tms()

        # Registrar datos
        tms.record_removal("X", 1, "C1", {"Y": [2]})
        tms.record_removal("X", 2, "C1", {"Y": [1]})
        tms.record_removal("Y", 3, "C2", {"X": [1]})
        tms.record_decision("X", 1)
        tms.record_decision("Y", 2)

        # Obtener estadísticas
        stats = tms.get_statistics()

        assert stats["total_justifications"] == 3
        assert stats["total_decisions"] == 2
        assert stats["variables_with_removals"] == 2
        assert stats["constraints_involved"] == 2

    def test_tms_clear(self):
        """Test: Limpieza del TMS."""
        tms = create_tms()

        # Agregar datos
        tms.record_removal("X", 1, "C1", {"Y": [2]})
        tms.record_decision("X", 1)

        # Limpiar
        tms.clear()

        assert len(tms.justifications) == 0
        assert len(tms.decisions) == 0
        assert len(tms.dependency_graph) == 0

