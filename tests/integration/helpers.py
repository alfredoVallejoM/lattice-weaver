from typing import Dict, List, Any, Optional, Set, Callable, Tuple
from dataclasses import dataclass

from lattice_weaver.core.csp_problem import CSP, Constraint
from lattice_weaver.core.csp_engine.solver import CSPSolver, CSPSolutionStats


@dataclass
class SolveStats:
    """
    Estadísticas de resolución de un problema CSP.
    
    Attributes:
        solutions: Lista de soluciones encontradas (diccionarios var -> valor)
        nodes_explored: Número de nodos explorados (estimado)
        backtracks: Número de backtracks (estimado)
        consistent: Si el problema tiene solución
    """
    solutions: List[Dict[str, Any]]
    nodes_explored: int
    backtracks: int
    consistent: bool


def solve_csp_problem(csp_problem: CSP, max_solutions: int = 1) -> SolveStats:
    """
    Resuelve un problema CSP usando CSPSolver.
    
    Args:
        csp_problem: Problema CSP a resolver (instancia de lattice_weaver.core.csp_problem.CSP)
        max_solutions: Número máximo de soluciones a encontrar
    
    Returns:
        Estadísticas de resolución con soluciones encontradas
    """
    solver = CSPSolver(csp_problem)
    stats: CSPSolutionStats = solver.solve(max_solutions=max_solutions)
    
    return SolveStats(
        solutions=stats.solutions,
        nodes_explored=stats.nodes_explored,
        backtracks=stats.backtracks,
        consistent=len(stats.solutions) > 0
    )


def create_map_coloring_problem(regions: List[str], 
                                borders: List[tuple], 
                                n_colors: int) -> CSP:
    """
    Crea un problema CSP para coloreo de mapas.
    
    Args:
        regions: Lista de regiones
        borders: Lista de pares de regiones adyacentes
        n_colors: Número de colores disponibles
    
    Returns:
        Problema CSP de coloreo de mapas
    """
    variables = frozenset(regions)
    domains = {region: frozenset(range(n_colors)) for region in regions}
    
    constraints = []
    for r1, r2 in borders:
        constraints.append(Constraint(
            scope=frozenset({r1, r2}),
            relation=lambda c1, c2: c1 != c2,
            name=f"neq_{r1}_{r2}"
        ))
    
    return CSP(
        variables=variables,
        domains=domains,
        constraints=frozenset(constraints),
        name="MapColoring"
    )


def create_nqueens_problem(n: int) -> CSP:
    """
    Crea un problema de N-Reinas.
    
    Args:
        n: Tamaño del tablero (n x n)
    
    Returns:
        Problema CSP de N-Reinas
    """
    variables = {f'Q{i}' for i in range(n)}
    domains = {f'Q{i}': frozenset(range(n)) for i in range(n)}
    constraints = []

    for i in range(n):
        for j in range(i + 1, n):
            constraints.append(Constraint(
                scope=frozenset({f'Q{i}', f'Q{j}'}),
                relation=lambda qi, qj, i=i, j=j: qi != qj and abs(qi - qj) != abs(i - j),
                name=f'diag_neq_Q{i}Q{j}'
            ))
    return CSP(variables=variables, domains=domains, constraints=frozenset(constraints), name=f"NQueens_{n}")


def create_sudoku_problem(puzzle: List[List[int]]) -> CSP:
    """
    Crea un problema CSP para Sudoku 4x4.
    
    Args:
        puzzle: Lista 4x4 con valores iniciales (0 = vacío)
    
    Returns:
        Problema CSP de Sudoku 4x4
    """
    variables = frozenset({f'C{i}{j}' for i in range(4) for j in range(4)})
    domains = {}
    for i in range(4):
        for j in range(4):
            var = f'C{i}{j}'
            if puzzle[i][j] == 0:
                domains[var] = frozenset({1, 2, 3, 4})
            else:
                domains[var] = frozenset({puzzle[i][j]})

    constraints = []

    # Row constraints
    for i in range(4):
        row_vars = [f'C{i}{j}' for j in range(4)]
        for k in range(4):
            for l in range(k + 1, 4):
                constraints.append(Constraint(
                    scope=frozenset({row_vars[k], row_vars[l]}),
                    relation=lambda v1, v2: v1 != v2,
                    name=f'row_neq_{row_vars[k]}_{row_vars[l]}'
                ))

    # Column constraints
    for j in range(4):
        col_vars = [f'C{i}{j}' for i in range(4)]
        for k in range(4):
            for l in range(k + 1, 4):
                constraints.append(Constraint(
                    scope=frozenset({col_vars[k], col_vars[l]}),
                    relation=lambda v1, v2: v1 != v2,
                    name=f'col_neq_{col_vars[k]}_{col_vars[l]}'
                ))

    # 2x2 block constraints
    for block_i in range(2):
        for block_j in range(2):
            block_vars = []
            for i in range(2):
                for j in range(2):
                    block_vars.append(f'C{block_i*2+i}{block_j*2+j}')
            
            for k in range(4):
                for l in range(k + 1, 4):
                    constraints.append(Constraint(
                        scope=frozenset({block_vars[k], block_vars[l]}),
                        relation=lambda v1, v2: v1 != v2,
                        name=f'block_neq_{block_vars[k]}_{block_vars[l]}'
                    ))
    
    return CSP(variables=variables, domains=domains, constraints=frozenset(constraints), name="Sudoku4x4")

