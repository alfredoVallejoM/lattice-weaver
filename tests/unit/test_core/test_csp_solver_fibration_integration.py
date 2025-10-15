import pytest
from unittest.mock import Mock
from lattice_weaver.core.csp_problem import CSP, Constraint as CSPConstraint
from lattice_weaver.core.csp_solver_fibration_integration import solve_csp_with_fibration_flow
from lattice_weaver.fibration import ConstraintHierarchy, ConstraintLevel, Hardness, FibrationSearchSolver

class TestCSPSolverFibrationIntegration:

    def test_solve_csp_with_fibration_flow_no_solver(self):
        """
        Test: solve_csp_with_fibration_flow devuelve None si no se proporciona un solver.
        """
        csp = CSP(variables=set(), domains={}, constraints=[])
        result = solve_csp_with_fibration_flow(csp)
        assert result is None

    def test_solve_csp_with_fibration_flow_solver_returns_none(self):
        """
        Test: solve_csp_with_fibration_flow devuelve None si el solver de Fibration Flow no encuentra solución.
        """
        csp = CSP(variables={"X"}, domains={"X": frozenset({1, 2})}, constraints=[])
        
        mock_solver = Mock(spec=FibrationSearchSolver)
        mock_solver.solve.return_value = None # Simula que no se encuentra solución

        result = solve_csp_with_fibration_flow(csp, solver=mock_solver)
        assert result is None
        mock_solver.solve.assert_called_once()

    def test_solve_csp_with_fibration_flow_solver_finds_solution(self):
        """
        Test: solve_csp_with_fibration_flow devuelve una solución si el solver de Fibration Flow la encuentra.
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
        mock_solver = Mock(spec=FibrationSearchSolver)
        # Simula que el solver de Fibration Flow encuentra una solución
        mock_solver.solve.return_value = {"X": 1, "Y": 2}

        result = solve_csp_with_fibration_flow(csp, solver=mock_solver)
        
        assert result == {"X": 1, "Y": 2}
        mock_solver.solve.assert_called_once()

        # Verificar que la jerarquía y los dominios pasados al solver son correctos
        args, kwargs = mock_solver.solve.call_args
        hierarchy_arg = args[0]
        domains_arg = args[1]

        assert isinstance(hierarchy_arg, ConstraintHierarchy)
        assert domains_arg == {"X": [1, 2], "Y": [1, 2]}

        # Verificar que la restricción CSP se convirtió correctamente en la jerarquía
        local_constraints = hierarchy_arg.get_constraints_by_level(ConstraintLevel.LOCAL)
        assert len(local_constraints) == 1
        fibration_constraint = local_constraints[0]
        assert fibration_constraint.hardness == Hardness.HARD
        assert set(fibration_constraint.variables) == {"X", "Y"}
        
        # Probar el predicado de la restricción convertida
        satisfied, _ = fibration_constraint.predicate({"X": 1, "Y": 2})
        assert satisfied
        satisfied, _ = fibration_constraint.predicate({"X": 1, "Y": 1})
        assert not satisfied

    def test_solve_csp_with_fibration_flow_complex_csp(self):
        """
        Test: solve_csp_with_fibration_flow con un CSP más complejo (N-Reinas 3x3).
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

        mock_solver = Mock(spec=FibrationSearchSolver)
        # Una solución para N-Reinas 3x3 (ej. Q0=0, Q1=2, Q2=1)
        mock_solver.solve.return_value = {"Q0": 0, "Q1": 2, "Q2": 1}

        result = solve_csp_with_fibration_flow(csp, solver=mock_solver)
        assert result == {"Q0": 0, "Q1": 2, "Q2": 1}
        mock_solver.solve.assert_called_once()

        # Verificar que la jerarquía y los dominios pasados al solver son correctos
        args, kwargs = mock_solver.solve.call_args
        hierarchy_arg = args[0]
        domains_arg = args[1]

        assert isinstance(hierarchy_arg, ConstraintHierarchy)
        assert domains_arg == {"Q0": [0, 1, 2], "Q1": [0, 1, 2], "Q2": [0, 1, 2]}

        # Verificar el número de restricciones convertidas (3 row_col + 3 diag = 6)
        local_constraints = hierarchy_arg.get_constraints_by_level(ConstraintLevel.LOCAL)
        assert len(local_constraints) == 6

        # Verificar una restricción diagonal convertida (ej. diag_Q0_Q1)
        diag_q0_q1_constraint = next(c for c in local_constraints if c.metadata.get("original_constraint_name") == "diag_Q0_Q1")
        # Q0=0, Q1=2. abs(0-2)=2. abs(0-1)=1. 2 != 1. Satisfecha.
        satisfied, _ = diag_q0_q1_constraint.predicate({"Q0": 0, "Q1": 2})
        assert satisfied
        # Q0=0, Q1=1. abs(0-1)=1. abs(0-1)=1. 1 != 1 es False. Violada.
        satisfied, _ = diag_q0_q1_constraint.predicate({"Q0": 0, "Q1": 1})
        assert not satisfied

        # Verificar una restricción de fila/columna convertida (ej. row_col_Q0_Q1)
        row_col_q0_q1_constraint = next(c for c in local_constraints if c.metadata.get("original_constraint_name") == "row_col_Q0_Q1")
        # Q0=0, Q1=1. 0 != 1. Satisfecha.
        satisfied, _ = row_col_q0_q1_constraint.predicate({"Q0": 0, "Q1": 1})
        assert satisfied
        # Q0=0, Q1=0. 0 != 0 es False. Violada.
        satisfied, _ = row_col_q0_q1_constraint.predicate({"Q0": 0, "Q1": 0})
        assert not satisfied

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

