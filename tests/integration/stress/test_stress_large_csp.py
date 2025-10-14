import pytest
import signal
from contextlib import contextmanager
from typing import Dict, Any, List

from lattice_weaver.core.csp_problem import CSP, Constraint
from lattice_weaver.core.csp_engine.solver import CSPSolver, CSPSolutionStats
from tests.integration.helpers import create_nqueens_problem, create_map_coloring_problem


class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds):
    """Context manager para timeout."""
    def signal_handler(signum, frame):
        raise TimeoutException("Timeout alcanzado")
    
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


@pytest.fixture(scope="module")
def large_nqueens_csp_problem():
    """
    Fixture: Problema de N-Reinas grande (n=16) para tests de estrés.
    """
    return create_nqueens_problem(16)


@pytest.fixture(scope="module")
def stress_timeout():
    """
    Fixture: Timeout para tests de estrés (120 segundos).
    """
    return 120


@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.slow
def test_nqueens_large(large_nqueens_csp_problem: CSP, stress_timeout: int):
    """
    Test: Resolver N-Reinas n=16 con timeout.
    
    Validación: Debe encontrar al menos 1 solución en <120s.
    """
    solver = CSPSolver(large_nqueens_csp_problem)
    
    try:
        with time_limit(stress_timeout):
            stats: CSPSolutionStats = solver.solve(max_solutions=1)
            
            assert len(stats.solutions) > 0, \
                "Debe encontrar al menos una solución"
            
            print(f"✅ N-Reinas n=16: Solución encontrada")
            print(f"   Nodos explorados: {stats.nodes_explored}")
            print(f"   Backtracks: {stats.backtracks}")
            print(f"   Solución: {stats.solutions[0]}")
    
    except TimeoutException:
        pytest.fail(f"Timeout de {stress_timeout}s alcanzado")


@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.slow
def test_graph_coloring_large(stress_timeout: int):
    """
    Test: Colorear grafo grande (50 nodos) con timeout.
    
    Validación: Debe encontrar coloración válida en <120s.
    """
    # Crear grafo aleatorio de 50 nodos con densidad media
    n_nodes = 50
    n_colors = 5
    
    regions = [f'N{i}' for i in range(n_nodes)]
    borders = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if (i + j) % 10 < 3:  # Aproximadamente 30% de aristas
                borders.append((regions[i], regions[j]))
    
    csp_problem = create_map_coloring_problem(regions, borders, n_colors)
    
    solver = CSPSolver(csp_problem)
    
    try:
        with time_limit(stress_timeout):
            stats: CSPSolutionStats = solver.solve(max_solutions=1)
            
            assert len(stats.solutions) > 0, \
                "Debe encontrar al menos una coloración válida"
            
            print(f"✅ Graph Coloring (50 nodos): Solución encontrada")
            print(f"   Aristas: {len(borders)}")
            print(f"   Colores: {n_colors}")
            print(f"   Nodos explorados: {stats.nodes_explored}")
            print(f"   Backtracks: {stats.backtracks}")
    
    except TimeoutException:
        pytest.fail(f"Timeout de {stress_timeout}s alcanzado")

