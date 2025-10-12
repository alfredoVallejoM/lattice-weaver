"""
Tests de estrés: CSP Grandes

Valida que el sistema pueda manejar problemas CSP de gran escala.
"""

import pytest
import signal
from contextlib import contextmanager
from lattice_weaver.arc_engine.core import ArcEngine
from lattice_weaver.formal.csp_integration import CSPProblem


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


@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.slow
def test_nqueens_large(large_nqueens_problem, stress_timeout):
    """
    Test: Resolver N-Reinas n=16 con timeout.
    
    Validación: Debe encontrar al menos 1 solución en <120s.
    """
    solver = ArcEngine()
    
    try:
        with time_limit(stress_timeout):
            stats = solver.solve(large_nqueens_problem, max_solutions=1)
            
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
def test_graph_coloring_large(stress_timeout):
    """
    Test: Colorear grafo grande (50 nodos) con timeout.
    
    Validación: Debe encontrar coloración válida en <120s.
    """
    # Crear grafo aleatorio de 50 nodos con densidad media
    n_nodes = 50
    n_colors = 5
    
    variables = [f'N{i}' for i in range(n_nodes)]
    domains = {var: list(range(n_colors)) for var in variables}
    
    # Crear aristas (30% de densidad)
    constraints = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if (i + j) % 10 < 3:  # 30% de aristas
                constraints.append((
                    f'N{i}',
                    f'N{j}',
                    lambda c1, c2: c1 != c2
                ))
    
    problem = CSPProblem(
        variables=variables,
        domains=domains,
        constraints=constraints
    )
    
    solver = ArcEngine()
    
    try:
        with time_limit(stress_timeout):
            stats = solver.solve(problem, max_solutions=1)
            
            assert len(stats.solutions) > 0, \
                "Debe encontrar al menos una coloración válida"
            
            print(f"✅ Graph Coloring (50 nodos): Solución encontrada")
            print(f"   Aristas: {len(constraints)}")
            print(f"   Colores: {n_colors}")
            print(f"   Nodos explorados: {stats.nodes_explored}")
            print(f"   Backtracks: {stats.backtracks}")
    
    except TimeoutException:
        pytest.fail(f"Timeout de {stress_timeout}s alcanzado")

