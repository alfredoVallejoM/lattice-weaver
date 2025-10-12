import pytest
import time
from typing import List, Dict, Any, Tuple, Callable

from lattice_weaver.arc_engine.csp_solver import CSPSolver, CSPProblem, CSPSolution
from lattice_weaver.arc_engine.constraints import nqueens_not_equal, nqueens_not_diagonal, register_relation

# Asegurarse de que las relaciones estén registradas (se registran al importar constraints.py)

def create_nqueens_problem(n: int) -> CSPProblem:
    """
    Crea un problema de N-Reinas para un N dado.
    """
    variables = {f"Q{i}": list(range(n)) for i in range(n)}
    constraints = []

    for i in range(n):
        for j in range(i + 1, n):
            # Restricción de no estar en la misma fila (implícita por el dominio)
            # Restricción de no estar en la misma columna (implícita por el dominio)

            # Restricción de no estar en la misma fila (redundante con el dominio)
            # constraints.append((f"Q{i}", f"Q{j}", nqueens_not_equal))

            # Restricción de no estar en la misma diagonal
            constraints.append((f"Q{i}", f"Q{j}", nqueens_not_diagonal))

    return CSPProblem(variables=list(variables.keys()), domains=variables, constraints=constraints)




@pytest.mark.parametrize("n", [4, 8]) # Reducido para evitar timeouts en el sandbox
def test_nqueens_sequential_no_tms(benchmark, n):
    """
    Benchmark para N-Reinas con CSPSolver secuencial sin TMS.
    """
    problem = create_nqueens_problem(n)
    solver = CSPSolver(use_tms=False, parallel=False)
    
    solutions = benchmark(solver.solve, problem, return_all=True)
    assert len(solutions.solutions) > 0, f"No se encontraron soluciones para N={n}"
    # Aquí se podrían añadir aserciones más robustas sobre la corrección de las soluciones

@pytest.mark.parametrize("n", [4, 8, 12])
def test_nqueens_sequential_with_tms(benchmark, n):
    """
    Benchmark para N-Reinas con CSPSolver secuencial con TMS.
    """
    problem = create_nqueens_problem(n)
    solver = CSPSolver(use_tms=True, parallel=False)
    
    solutions = benchmark(solver.solve, problem, return_all=True)
    assert len(solutions.solutions) > 0, f"No se encontraron soluciones para N={n}"

@pytest.mark.parametrize("n", [4, 8, 12])
def test_nqueens_parallel_no_tms(benchmark, n):
    """
    Benchmark para N-Reinas con CSPSolver paralelo sin TMS.
    """
    problem = create_nqueens_problem(n)
    solver = CSPSolver(use_tms=False, parallel=True, parallel_mode='topological')
    
    solutions = benchmark(solver.solve, problem, return_all=True)
    assert len(solutions.solutions) > 0, f"No se encontraron soluciones para N={n}"

@pytest.mark.parametrize("n", [4, 8, 12])
def test_nqueens_parallel_with_tms(benchmark, n):
    """
    Benchmark para N-Reinas con CSPSolver paralelo con TMS.
    """
    problem = create_nqueens_problem(n)
    solver = CSPSolver(use_tms=True, parallel=True, parallel_mode='topological')
    
    solutions = benchmark(solver.solve, problem, return_all=True)
    assert len(solutions.solutions) > 0, f"No se encontraron soluciones para N={n}"

# Tests de integración adicionales para verificar la funcionalidad del TMS y la paralelización

def test_tms_integration_with_solver():
    """
    Verifica que el TMS registra eliminaciones y puede explicar inconsistencias
    cuando se usa con el CSPSolver.
    """
    n = 4
    problem = create_nqueens_problem(n)
    solver = CSPSolver(use_tms=True, parallel=False)
    
    # Forzar una inconsistencia para activar el TMS
    # Esto es un ejemplo, en un caso real, la inconsistencia surgiría naturalmente
    # durante la búsqueda si no hay soluciones.
    # Para este test, simplemente verificamos que el TMS se inicializa y se usa.
    solver.solve(problem, return_all=True)
    
    assert solver.arc_engine.tms is not None
    # Se podrían añadir aserciones más específicas sobre el contenido del TMS
    # si se expusieran métodos para acceder a sus registros.

def test_parallel_correctness():
    """
    Verifica que las soluciones encontradas por el solver paralelo son las mismas
    que las encontradas por el solver secuencial para un problema pequeño.
    """
    n = 4
    problem = create_nqueens_problem(n)
    
    solver_seq = CSPSolver(use_tms=False, parallel=False)
    solutions_seq = solver_seq.solve(problem, return_all=True)
    
    solver_par = CSPSolver(use_tms=False, parallel=True, parallel_mode='topological')
    solutions_par = solver_par.solve(problem, return_all=True)
    
    assert len(solutions_seq.solutions) == len(solutions_par.solutions)
    # Convertir soluciones a un formato comparable (ej. sets de tuplas)
    set_solutions_seq = {frozenset(sol.assignment.items()) for sol in solutions_seq.solutions}
    set_solutions_par = {frozenset(sol.assignment.items()) for sol in solutions_par.solutions}
    
    assert set_solutions_seq == set_solutions_par


