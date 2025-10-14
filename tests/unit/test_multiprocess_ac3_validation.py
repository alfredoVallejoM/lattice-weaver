import pytest
from lattice_weaver.core.csp_problem import CSP, Constraint
from lattice_weaver.core.csp_engine.solver import CSPSolver
from multiprocessing import active_children

def create_nqueens_problem(n):
    variables = {f"Q{i}" for i in range(n)}
    domains = {f"Q{i}": frozenset(range(n)) for i in range(n)}
    constraints = []

    for i in range(n):
        for j in range(i + 1, n):
            qi = f"Q{i}"
            qj = f"Q{j}"
            constraints.append(Constraint(
                scope=frozenset({qi, qj}),
                relation=lambda val_i, val_j, captured_diff=abs(i - j): abs(val_i - val_j) != captured_diff,
                name=f"diag_{qi}_{qj}"
            ))
            constraints.append(Constraint(
                scope=frozenset({qi, qj}),
                relation=lambda val_i, val_j: val_i != val_j,
                name=f"row_col_{qi}_{qj}"
            ))
    return CSP(variables=variables, domains=domains, constraints=constraints)


def test_multiprocess_ac3_nqueens_4():
    """
    Valida la paralelización multiproceso de AC-3.1 para el problema de 4-Reinas.
    """
    n = 4
    csp_problem = create_nqueens_problem(n)

    solver = CSPSolver(csp_problem, parallel=True, parallel_mode='topological')
    solutions = solver.solve_all()

    # El problema de 4-Reinas tiene 2 soluciones únicas
    assert len(solutions) == 2
    for sol in solutions:
        assert len(sol) == n
        # Verificación de no-igualdad (simplificada)
        assigned_cols = list(sol.values())
        assert len(set(assigned_cols)) == n # Todas las reinas en columnas diferentes

def create_nqueens_problem_full(n):
    variables = {f"Q{i}" for i in range(n)}
    domains = {f"Q{i}": frozenset(range(n)) for i in range(n)}
    constraints = []

    for i in range(n):
        for j in range(i + 1, n):
            qi = f"Q{i}"
            qj = f"Q{j}"
            constraints.append(Constraint(
                scope=frozenset({qi, qj}),
                relation=lambda val_i, val_j, captured_diff=abs(i - j): abs(val_i - val_j) != captured_diff,
                name=f"diag_{qi}_{qj}"
            ))
            constraints.append(Constraint(
                scope=frozenset({qi, qj}),
                relation=lambda val_i, val_j: val_i != val_j,
                name=f"row_col_{qi}_{qj}"
            ))
    return CSP(variables=variables, domains=domains, constraints=constraints)


def test_multiprocess_ac3_nqueens_4_full():
    """
    Valida la paralelización multiproceso de AC-3.1 para el problema de 4-Reinas con restricciones de diagonal.
    """
    n = 4
    csp_problem = create_nqueens_problem_full(n)

    solver = CSPSolver(csp_problem, parallel=True, parallel_mode='topological')
    solutions = solver.solve_all()

    # El problema de 4-Reinas tiene 2 soluciones únicas
    assert len(solutions) == 2
    for sol in solutions:
        assert len(sol) == n
        # Verificar que las soluciones son válidas (no hay reinas en la misma fila, columna o diagonal)
        assigned_cols = list(sol.values())
        assert len(set(assigned_cols)) == n # Todas las reinas en columnas diferentes

        # Verificar restricciones de diagonal
        for i in range(n):
            for j in range(i + 1, n):
                val_i = sol[f"Q{i}"]
                val_j = sol[f"Q{j}"]
                assert abs(val_i - val_j) != abs(i - j), f"Fallo diagonal para Q{i}={val_i}, Q{j}={val_j}"

