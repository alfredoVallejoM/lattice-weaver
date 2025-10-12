"""
Problemas de referencia para benchmarking.

Este módulo define problemas CSP estándar usados para comparar
el rendimiento de diferentes algoritmos de resolución.
"""
from typing import Dict, Set, List, Tuple, Callable, Any
from dataclasses import dataclass
from lattice_weaver.formal.csp_integration import CSPProblem


@dataclass
class BenchmarkProblem:
    """
    Problema de benchmark con metadata.
    
    Attributes:
        name: Nombre del problema
        problem: Instancia del problema CSP
        expected_solutions: Número esperado de soluciones
        difficulty: Nivel de dificultad (easy, medium, hard)
        category: Categoría del problema
    """
    name: str
    problem: CSPProblem
    expected_solutions: int
    difficulty: str
    category: str


def create_nqueens(n: int) -> BenchmarkProblem:
    """
    Crea un problema de N-Reinas.
    
    Args:
        n: Tamaño del tablero (número de reinas)
    
    Returns:
        Problema de benchmark
    """
    variables = [f"Q{i}" for i in range(n)]
    domains = {f"Q{i}": set(range(n)) for i in range(n)}
    
    constraints = []
    for i in range(n):
        for j in range(i + 1, n):
            # No misma fila
            constraints.append((f"Q{i}", f"Q{j}", lambda a, b: a != b))
            # No misma diagonal
            constraints.append((f"Q{i}", f"Q{j}", 
                              lambda a, b, i=i, j=j: abs(a - b) != abs(i - j)))
    
    problem = CSPProblem(
        variables=variables,
        domains=domains,
        constraints=constraints
    )
    
    # Número de soluciones conocidas para N-Reinas
    solutions_count = {
        4: 2, 5: 10, 6: 4, 7: 40, 8: 92, 9: 352, 10: 724, 11: 2680, 12: 14200
    }
    
    difficulty = "easy" if n <= 6 else "medium" if n <= 8 else "hard"
    
    return BenchmarkProblem(
        name=f"N-Reinas (n={n})",
        problem=problem,
        expected_solutions=solutions_count.get(n, -1),
        difficulty=difficulty,
        category="constraint_satisfaction"
    )


def create_sudoku_4x4(givens: Dict[str, int] = None) -> BenchmarkProblem:
    """
    Crea un problema de Sudoku 4x4.
    
    Args:
        givens: Valores iniciales dados {celda: valor}
    
    Returns:
        Problema de benchmark
    """
    n = 4
    variables = [f"C{i}{j}" for i in range(n) for j in range(n)]
    domains = {f"C{i}{j}": set(range(1, n + 1)) for i in range(n) for j in range(n)}
    
    # Aplicar valores dados
    if givens:
        for cell, value in givens.items():
            domains[cell] = {value}
    
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
    
    problem = CSPProblem(
        variables=variables,
        domains=domains,
        constraints=constraints
    )
    
    givens_count = len(givens) if givens else 0
    difficulty = "easy" if givens_count >= 8 else "medium" if givens_count >= 6 else "hard"
    
    return BenchmarkProblem(
        name=f"Sudoku 4x4 ({givens_count} givens)",
        problem=problem,
        expected_solutions=1,  # Sudoku bien formado tiene 1 solución
        difficulty=difficulty,
        category="constraint_satisfaction"
    )


def create_graph_coloring(num_nodes: int, edges: List[Tuple[int, int]], num_colors: int) -> BenchmarkProblem:
    """
    Crea un problema de coloreo de grafos.
    
    Args:
        num_nodes: Número de nodos
        edges: Lista de aristas (pares de nodos)
        num_colors: Número de colores disponibles
    
    Returns:
        Problema de benchmark
    """
    variables = [f"N{i}" for i in range(num_nodes)]
    domains = {f"N{i}": set(range(num_colors)) for i in range(num_nodes)}
    
    constraints = []
    for n1, n2 in edges:
        constraints.append((f"N{n1}", f"N{n2}", lambda a, b: a != b))
    
    problem = CSPProblem(
        variables=variables,
        domains=domains,
        constraints=constraints
    )
    
    # Dificultad basada en densidad del grafo
    max_edges = num_nodes * (num_nodes - 1) // 2
    density = len(edges) / max_edges if max_edges > 0 else 0
    difficulty = "easy" if density < 0.3 else "medium" if density < 0.6 else "hard"
    
    return BenchmarkProblem(
        name=f"Graph Coloring ({num_nodes} nodes, {len(edges)} edges, {num_colors} colors)",
        problem=problem,
        expected_solutions=-1,  # Desconocido en general
        difficulty=difficulty,
        category="graph_coloring"
    )


def create_map_coloring() -> BenchmarkProblem:
    """
    Crea el problema clásico de coloreo de mapa de Australia.
    
    Returns:
        Problema de benchmark
    """
    # Estados de Australia
    states = ["WA", "NT", "SA", "Q", "NSW", "V", "T"]
    
    # Fronteras
    borders = [
        ("WA", "NT"), ("WA", "SA"),
        ("NT", "SA"), ("NT", "Q"),
        ("SA", "Q"), ("SA", "NSW"), ("SA", "V"),
        ("Q", "NSW"),
        ("NSW", "V")
    ]
    
    variables = states
    domains = {state: {"red", "green", "blue"} for state in states}
    
    constraints = []
    for s1, s2 in borders:
        constraints.append((s1, s2, lambda a, b: a != b))
    
    problem = CSPProblem(
        variables=variables,
        domains=domains,
        constraints=constraints
    )
    
    return BenchmarkProblem(
        name="Map Coloring (Australia)",
        problem=problem,
        expected_solutions=18,  # Número conocido de soluciones
        difficulty="easy",
        category="graph_coloring"
    )


def create_scheduling_problem(num_jobs: int, num_time_slots: int) -> BenchmarkProblem:
    """
    Crea un problema simple de scheduling.
    
    Args:
        num_jobs: Número de trabajos
        num_time_slots: Número de slots de tiempo
    
    Returns:
        Problema de benchmark
    """
    variables = [f"J{i}" for i in range(num_jobs)]
    domains = {f"J{i}": set(range(num_time_slots)) for i in range(num_jobs)}
    
    constraints = []
    
    # Restricción: algunos trabajos no pueden ser simultáneos
    # (simplificación: trabajos consecutivos no pueden estar en el mismo slot)
    for i in range(num_jobs - 1):
        constraints.append((f"J{i}", f"J{i+1}", lambda a, b: a != b))
    
    problem = CSPProblem(
        variables=variables,
        domains=domains,
        constraints=constraints
    )
    
    difficulty = "easy" if num_jobs <= 5 else "medium" if num_jobs <= 8 else "hard"
    
    return BenchmarkProblem(
        name=f"Scheduling ({num_jobs} jobs, {num_time_slots} slots)",
        problem=problem,
        expected_solutions=-1,
        difficulty=difficulty,
        category="scheduling"
    )


# Suite de problemas estándar
STANDARD_SUITE = [
    create_nqueens(4),
    create_nqueens(6),
    create_nqueens(8),
    create_sudoku_4x4({"C00": 1, "C01": 2, "C10": 3, "C11": 4}),
    create_graph_coloring(5, [(0,1), (0,2), (1,2), (1,3), (2,3), (2,4), (3,4)], 3),
    create_map_coloring(),
    create_scheduling_problem(5, 3),
]


def get_problem_by_name(name: str) -> BenchmarkProblem:
    """
    Obtiene un problema por nombre.
    
    Args:
        name: Nombre del problema
    
    Returns:
        Problema de benchmark
    
    Raises:
        ValueError: Si el problema no existe
    """
    for problem in STANDARD_SUITE:
        if problem.name == name:
            return problem
    raise ValueError(f"Problema '{name}' no encontrado")


def get_problems_by_difficulty(difficulty: str) -> List[BenchmarkProblem]:
    """
    Obtiene problemas por nivel de dificultad.
    
    Args:
        difficulty: Nivel de dificultad (easy, medium, hard)
    
    Returns:
        Lista de problemas
    """
    return [p for p in STANDARD_SUITE if p.difficulty == difficulty]


def get_problems_by_category(category: str) -> List[BenchmarkProblem]:
    """
    Obtiene problemas por categoría.
    
    Args:
        category: Categoría del problema
    
    Returns:
        Lista de problemas
    """
    return [p for p in STANDARD_SUITE if p.category == category]

