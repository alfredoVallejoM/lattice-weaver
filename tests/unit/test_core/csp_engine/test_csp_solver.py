import pytest
from unittest.mock import Mock
from lattice_weaver.core.csp_problem import CSP, Constraint as CSPConstraint
from lattice_weaver.core.csp_engine.solver import CSPSolver, CSPSolution
from lattice_weaver.fibration import FibrationSearchSolver

class TestCSPSolver:

    def test_solve_with_fibration_flow_finds_solution(self):
        """
        Test: CSPSolver.solve_with_fibration_flow encuentra una solución usando Fibration Flow.
        """
        # CSP simple: X en {1, 2}, Y en {1, 2}, X != Y
        variables = {"X", "Y"}
        domains = {"X": frozenset({1, 2}), "Y": frozenset({1, 2})}
        constraints = [
            CSPConstraint(
                scope=frozenset({"X", "Y"}),
                relation=lambda x, y: x != y,
                name="X_neq_Y"
            )
        ]
        csp = CSP(variables=variables, domains=domains, constraints=constraints)

        # Mock del solver de Fibration Flow
        mock_fibration_solver = Mock(spec=FibrationSearchSolver)
        mock_fibration_solver.solve.return_value = {"X": 1, "Y": 2}

        solver = CSPSolver(csp)
        # Inyectar el mock solver en el método para la prueba
        solver.fibration_solver = mock_fibration_solver

        stats = solver.solve_with_fibration_flow()
        
        assert len(stats.solutions) == 1
        assert stats.solutions[0].assignment == {"X": 1, "Y": 2}
        mock_fibration_solver.solve.assert_called_once()

    def test_solve_with_fibration_flow_no_solution(self):
        """
        Test: CSPSolver.solve_with_fibration_flow no encuentra solución si Fibration Flow no la encuentra.
        """
        # CSP insatisfacible: X en {1}, Y en {1}, X != Y
        variables = {"X", "Y"}
        domains = {"X": frozenset({1}), "Y": frozenset({1})}
        constraints = [
            CSPConstraint(
                scope=frozenset({"X", "Y"}),
                relation=lambda x, y: x != y,
                name="X_neq_Y"
            )
        ]
        csp = CSP(variables=variables, domains=domains, constraints=constraints)

        # Mock del solver de Fibration Flow que devuelve None
        mock_fibration_solver = Mock(spec=FibrationSearchSolver)
        mock_fibration_solver.solve.return_value = None

        solver = CSPSolver(csp)
        solver.fibration_solver = mock_fibration_solver

        stats = solver.solve_with_fibration_flow()
        
        assert len(stats.solutions) == 0
        mock_fibration_solver.solve.assert_called_once()

    def test_solve_with_fibration_flow_complex_csp(self):
        """
        Test: CSPSolver.solve_with_fibration_flow con un CSP más complejo (N-Reinas 3x3).
        """
        # N-Reinas 3x3 (tiene solución)
        variables = {"Q0", "Q1", "Q2"}
        domains = {"Q0": frozenset({0, 1, 2}), "Q1": frozenset({0, 1, 2}), "Q2": frozenset({0, 1, 2})}
        constraints = []
        for i in range(3):
            for j in range(i + 1, 3):
                qi = f"Q{i}"
                qj = f"Q{j}"
                # Restricción de fila/columna: val_i != val_j
                constraints.append(CSPConstraint(
                    scope=frozenset({qi, qj}),
                    relation=lambda val_i, val_j: val_i != val_j,
                    name=f"row_col_{qi}_{qj}"
                ))
                # Restricción diagonal: abs(val_i - val_j) != abs(i - j)
                constraints.append(CSPConstraint(
                    scope=frozenset({qi, qj}),
                    relation=(lambda i_val=i, j_val=j: lambda val_i, val_j: abs(val_i - val_j) != abs(i_val - j_val))(),
                    name=f"diag_{qi}_{qj}"
                ))
        csp = CSP(variables=variables, domains=domains, constraints=constraints, name="NQueens3x3")

        mock_fibration_solver = Mock(spec=FibrationSearchSolver)
        # Una solución para N-Reinas 3x3 (ej. Q0=0, Q1=2, Q2=1)
        mock_fibration_solver.solve.return_value = {"Q0": 0, "Q1": 2, "Q2": 1}

        solver = CSPSolver(csp)
        solver.fibration_solver = mock_fibration_solver

        stats = solver.solve_with_fibration_flow()
        assert len(stats.solutions) == 1
        assert stats.solutions[0].assignment == {"Q0": 0, "Q1": 2, "Q2": 1}
        mock_fibration_solver.solve.assert_called_once()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

