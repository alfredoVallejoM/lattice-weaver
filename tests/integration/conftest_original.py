"""
Fixtures específicas para tests de integración.

Este módulo proporciona fixtures pre-configuradas de los motores principales
del sistema para facilitar los tests de integración.
"""
import pytest
from lattice_weaver.core.csp_problem import CSP
from lattice_weaver.formal.csp_integration import CSPProblem
from lattice_weaver.lattice_core import ParallelFCAEngine
from lattice_weaver.topology import create_tda_engine
from lattice_weaver.formal import CubicalEngine, TypeChecker

@pytest.fixture(scope="module")
def arc_engine():
    """
    Fixture: ArcEngine configurado.
    
    Retorna una instancia del motor de resolución CSP con configuración por defecto.
    Scope 'module' para reutilizar entre tests del mismo módulo.
    """
    return CSP(variables=set(), domains={}, constraints=[], name="TestCSP")

@pytest.fixture(scope="module")
def fca_engine():
    """
    Fixture: FCA Engine configurado.
    
    Retorna una instancia del motor de Análisis Formal de Conceptos
    con paralelización habilitada.
    """
    return ParallelFCAEngine()

@pytest.fixture(scope="module")
def tda_engine():
    """
    Fixture: TDA Engine configurado.
    
    Retorna una instancia del motor de Análisis Topológico de Datos.
    """
    return create_tda_engine()

@pytest.fixture(scope="module")
def cubical_engine():
    """
    Fixture: Cubical Engine configurado.
    
    Retorna una instancia del motor cúbico de HoTT con type checker.
    """
    return CubicalEngine(TypeChecker())

@pytest.fixture
def nqueens_problem():
    """
    Fixture: Problema N-Reinas (n=4).
    
    Retorna un problema CSP de N-Reinas con n=4.
    Este problema tiene exactamente 2 soluciones.
    """
    n = 4
    variables = [f"Q{i}" for i in range(n)]
    domains = {f"Q{i}": set(range(n)) for i in range(n)}
    
    # Restricciones: no atacarse
    constraints = []
    for i in range(n):
        for j in range(i + 1, n):
            # No misma fila
            constraints.append((f"Q{i}", f"Q{j}", lambda a, b: a != b))
            # No misma diagonal
            constraints.append((f"Q{i}", f"Q{j}", 
                              lambda a, b, i=i, j=j: abs(a - b) != abs(i - j)))
    
    return CSPProblem(
        variables=variables,
        domains=domains,
        constraints=constraints
    )

@pytest.fixture
def sudoku_problem():
    """
    Fixture: Problema Sudoku simple.
    
    Retorna un problema Sudoku 4x4 simplificado para tests rápidos.
    """
    n = 4  # Sudoku 4x4 (más simple que 9x9)
    variables = [f"C{i}{j}" for i in range(n) for j in range(n)]
    domains = {f"C{i}{j}": set(range(1, n + 1)) for i in range(n) for j in range(n)}
    
    constraints = []
    
    # Restricciones: filas únicas
    for i in range(n):
        for j1 in range(n):
            for j2 in range(j1 + 1, n):
                constraints.append((f"C{i}{j1}", f"C{i}{j2}", lambda a, b: a != b))
    
    # Restricciones: columnas únicas
    for j in range(n):
        for i1 in range(n):
            for i2 in range(i1 + 1, n):
                constraints.append((f"C{i1}{j}", f"C{i2}{j}", lambda a, b: a != b))
    
    # Restricciones: bloques 2x2 únicos
    for block_i in range(2):
        for block_j in range(2):
            cells = []
            for i in range(2):
                for j in range(2):
                    cells.append(f"C{block_i*2 + i}{block_j*2 + j}")
            
            for c1 in range(len(cells)):
                for c2 in range(c1 + 1, len(cells)):
                    constraints.append((cells[c1], cells[c2], lambda a, b: a != b))
    
    return CSPProblem(
        variables=variables,
        domains=domains,
        constraints=constraints
    )

@pytest.fixture
def graph_coloring_problem():
    """
    Fixture: Problema de coloreo de grafos.
    
    Retorna un problema de coloreo de grafos pequeño (5 nodos, 7 aristas, 3 colores).
    """
    nodes = 5
    colors = 3
    
    variables = [f"N{i}" for i in range(nodes)]
    domains = {f"N{i}": set(range(colors)) for i in range(nodes)}
    
    # Aristas (grafo ejemplo)
    edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4)]
    
    # Restricciones: nodos adyacentes con colores diferentes
    constraints = []
    for n1, n2 in edges:
        constraints.append((f"N{n1}", f"N{n2}", lambda a, b: a != b))
    
    return CSPProblem(
        variables=variables,
        domains=domains,
        constraints=constraints
    )

@pytest.fixture
def sample_lattice():
    """
    Fixture: Retículo de conceptos pre-computado.
    
    Retorna un retículo de conceptos pequeño para tests de topología.
    """
    from lattice_weaver.lattice_core.builder import FormalConcept
    
    # Conceptos de ejemplo
    concepts = [
        FormalConcept({"o1", "o2", "o3"}, set()),  # Top
        FormalConcept({"o1", "o2"}, {"a1"}),
        FormalConcept({"o2", "o3"}, {"a2"}),
        FormalConcept({"o1"}, {"a1", "a3"}),
        FormalConcept({"o2"}, {"a1", "a2"}),
        FormalConcept({"o3"}, {"a2", "a3"}),
        FormalConcept(set(), {"a1", "a2", "a3"}),  # Bottom
    ]
    
    return concepts

