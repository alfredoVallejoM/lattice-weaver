"""
Wrappers y Helpers para Tests de Integración

Este módulo proporciona funciones wrapper que adaptan la API de bajo nivel
de LatticeWeaver a interfaces más convenientes para testing.

Basado en la API real documentada en:
- lattice_weaver/arc_engine/core.py
- lattice_weaver/formal/csp_integration.py

Autor: LatticeWeaver Team
Fecha: 12 de Octubre de 2025
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from lattice_weaver.arc_engine.core import ArcEngine
from lattice_weaver.formal.csp_integration import CSPProblem


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


def solve_csp_problem(problem: CSPProblem, max_solutions: int = 1) -> SolveStats:
    """
    Resuelve un problema CSP usando ArcEngine.
    
    Este wrapper adapta CSPProblem a la API de ArcEngine:
    1. Crea un ArcEngine
    2. Agrega variables con sus dominios
    3. Agrega restricciones
    4. Ejecuta arc consistency
    5. Extrae soluciones mediante backtracking
    
    Args:
        problem: Problema CSP a resolver
        max_solutions: Número máximo de soluciones a encontrar
    
    Returns:
        Estadísticas de resolución con soluciones encontradas
    
    Example:
        >>> problem = CSPProblem(
        ...     variables=['x', 'y'],
        ...     domains={'x': {1, 2}, 'y': {1, 2}},
        ...     constraints=[('x', 'y', lambda a, b: a != b)]
        ... )
        >>> stats = solve_csp_problem(problem, max_solutions=10)
        >>> print(f"Soluciones: {len(stats.solutions)}")
        Soluciones: 2
    """
    # 1. Crear engine
    engine = ArcEngine()
    
    # 2. Agregar variables con dominios
    for var in problem.variables:
        domain = list(problem.domains[var])
        engine.add_variable(var, domain)
    
    # 3. Agregar restricciones
    for i, (var1, var2, relation) in enumerate(problem.constraints):
        cid = f"c{i}_{var1}_{var2}"
        engine.add_constraint(var1, var2, relation, cid=cid)
    
    # 4. Ejecutar arc consistency
    consistent = engine.enforce_arc_consistency()
    
    if not consistent:
        # Problema inconsistente
        return SolveStats(
            solutions=[],
            nodes_explored=len(problem.variables),
            backtracks=0,
            consistent=False
        )
    
    # 5. Extraer soluciones mediante backtracking
    solutions = []
    nodes_explored = 0
    backtracks = 0
    
    def backtrack(assignment: Dict[str, Any], remaining_vars: List[str]) -> bool:
        """Backtracking recursivo para encontrar soluciones."""
        nonlocal nodes_explored, backtracks, solutions
        
        nodes_explored += 1
        
        # Caso base: todas las variables asignadas
        if not remaining_vars:
            solutions.append(assignment.copy())
            return len(solutions) >= max_solutions
        
        # Seleccionar siguiente variable
        var = remaining_vars[0]
        remaining = remaining_vars[1:]
        
        # Probar cada valor en el dominio
        for value in engine.variables[var].get_values():
            # Verificar consistencia con asignación actual
            is_consistent = True
            
            for assigned_var, assigned_val in assignment.items():
                # Buscar restricciones entre var y assigned_var
                for cid, constraint in engine.constraints.items():
                    if (constraint.var1 == var and constraint.var2 == assigned_var):
                        if not constraint.relation(value, assigned_val):
                            is_consistent = False
                            break
                    elif (constraint.var1 == assigned_var and constraint.var2 == var):
                        if not constraint.relation(assigned_val, value):
                            is_consistent = False
                            break
                
                if not is_consistent:
                    backtracks += 1
                    break
            
            if is_consistent:
                # Asignar y continuar
                assignment[var] = value
                if backtrack(assignment, remaining):
                    return True
                del assignment[var]
        
        return False
    
    # Iniciar backtracking
    backtrack({}, problem.variables)
    
    return SolveStats(
        solutions=solutions,
        nodes_explored=nodes_explored,
        backtracks=backtracks,
        consistent=True
    )


def create_nqueens_problem(n: int) -> CSPProblem:
    """
    Crea un problema de N-Reinas.
    
    Args:
        n: Tamaño del tablero (n x n)
    
    Returns:
        Problema CSP de N-Reinas
    
    Example:
        >>> problem = create_nqueens_problem(4)
        >>> stats = solve_csp_problem(problem)
        >>> print(f"Soluciones: {len(stats.solutions)}")
        Soluciones: 2
    """
    variables = [f'Q{i}' for i in range(n)]
    domains = {var: set(range(n)) for var in variables}
    
    constraints = []
    for i in range(n):
        for j in range(i + 1, n):
            # No en la misma fila
            constraints.append((
                f'Q{i}',
                f'Q{j}',
                lambda ri, rj: ri != rj
            ))
            # No en la misma diagonal
            constraints.append((
                f'Q{i}',
                f'Q{j}',
                lambda ri, rj, i=i, j=j: abs(ri - rj) != abs(i - j)
            ))
    
    return CSPProblem(
        variables=variables,
        domains=domains,
        constraints=constraints
    )


def create_sudoku_4x4(puzzle: List[List[int]]) -> CSPProblem:
    """
    Crea un problema CSP para Sudoku 4x4.
    
    Args:
        puzzle: Lista 4x4 con valores iniciales (0 = vacío)
    
    Returns:
        Problema CSP de Sudoku 4x4
    
    Example:
        >>> puzzle = [[1,0,0,4], [0,0,1,0], [0,1,0,0], [4,0,0,1]]
        >>> problem = create_sudoku_4x4(puzzle)
        >>> stats = solve_csp_problem(problem)
        >>> print(f"Solución única: {len(stats.solutions) == 1}")
        Solución única: True
    """
    variables = []
    domains = {}
    
    # Crear variables para cada celda
    for i in range(4):
        for j in range(4):
            var = f'C{i}{j}'
            variables.append(var)
            if puzzle[i][j] == 0:
                domains[var] = {1, 2, 3, 4}
            else:
                domains[var] = {puzzle[i][j]}
    
    constraints = []
    
    # Restricciones de fila
    for i in range(4):
        row_vars = [f'C{i}{j}' for j in range(4)]
        for k in range(4):
            for l in range(k + 1, 4):
                constraints.append((
                    row_vars[k],
                    row_vars[l],
                    lambda v1, v2: v1 != v2
                ))
    
    # Restricciones de columna
    for j in range(4):
        col_vars = [f'C{i}{j}' for i in range(4)]
        for k in range(4):
            for l in range(k + 1, 4):
                constraints.append((
                    col_vars[k],
                    col_vars[l],
                    lambda v1, v2: v1 != v2
                ))
    
    # Restricciones de bloque 2x2
    for block_i in range(2):
        for block_j in range(2):
            block_vars = []
            for i in range(2):
                for j in range(2):
                    block_vars.append(f'C{block_i*2+i}{block_j*2+j}')
            
            for k in range(4):
                for l in range(k + 1, 4):
                    constraints.append((
                        block_vars[k],
                        block_vars[l],
                        lambda v1, v2: v1 != v2
                    ))
    
    return CSPProblem(
        variables=variables,
        domains=domains,
        constraints=constraints
    )


def create_map_coloring_problem(regions: List[str], 
                                borders: List[tuple], 
                                n_colors: int) -> CSPProblem:
    """
    Crea un problema CSP para coloreo de mapas.
    
    Args:
        regions: Lista de regiones
        borders: Lista de pares de regiones adyacentes
        n_colors: Número de colores disponibles
    
    Returns:
        Problema CSP de coloreo de mapas
    
    Example:
        >>> regions = ['A', 'B', 'C']
        >>> borders = [('A', 'B'), ('B', 'C')]
        >>> problem = create_map_coloring_problem(regions, borders, 2)
        >>> stats = solve_csp_problem(problem)
        >>> print(f"Coloreable: {len(stats.solutions) > 0}")
        Coloreable: True
    """
    variables = regions
    domains = {region: set(range(n_colors)) for region in regions}
    
    constraints = []
    for r1, r2 in borders:
        constraints.append((
            r1,
            r2,
            lambda c1, c2: c1 != c2
        ))
    
    return CSPProblem(
        variables=variables,
        domains=domains,
        constraints=constraints
    )


# ============================================================================
# Factory de Wrappers
# ============================================================================

class ProblemFactory:
    """
    Factory para crear problemas CSP estándar.
    
    Proporciona métodos convenientes para crear problemas comunes
    sin necesidad de especificar manualmente variables, dominios y restricciones.
    
    Example:
        >>> factory = ProblemFactory()
        >>> problem = factory.nqueens(8)
        >>> stats = solve_csp_problem(problem)
        >>> print(f"Soluciones: {len(stats.solutions)}")
    """
    
    @staticmethod
    def nqueens(n: int) -> CSPProblem:
        """Crea problema N-Reinas."""
        return create_nqueens_problem(n)
    
    @staticmethod
    def sudoku_4x4(puzzle: List[List[int]]) -> CSPProblem:
        """Crea problema Sudoku 4x4."""
        return create_sudoku_4x4(puzzle)
    
    @staticmethod
    def map_coloring(regions: List[str], borders: List[tuple], 
                    n_colors: int) -> CSPProblem:
        """Crea problema de coloreo de mapas."""
        return create_map_coloring_problem(regions, borders, n_colors)
    
    @staticmethod
    def graph_coloring(n_nodes: int, edges: List[tuple], 
                      n_colors: int) -> CSPProblem:
        """
        Crea problema de coloreo de grafos.
        
        Args:
            n_nodes: Número de nodos
            edges: Lista de aristas (pares de índices)
            n_colors: Número de colores
        
        Returns:
            Problema CSP de coloreo de grafos
        """
        variables = [f'N{i}' for i in range(n_nodes)]
        domains = {var: set(range(n_colors)) for var in variables}
        
        constraints = []
        for i, j in edges:
            constraints.append((
                f'N{i}',
                f'N{j}',
                lambda c1, c2: c1 != c2
            ))
        
        return CSPProblem(
            variables=variables,
            domains=domains,
            constraints=constraints
        )
    
    @staticmethod
    def random_csp(n_vars: int, domain_size: int, 
                  constraint_density: float = 0.3) -> CSPProblem:
        """
        Crea problema CSP aleatorio.
        
        Args:
            n_vars: Número de variables
            domain_size: Tamaño de cada dominio
            constraint_density: Densidad de restricciones (0-1)
        
        Returns:
            Problema CSP aleatorio
        """
        import random
        
        variables = [f'V{i}' for i in range(n_vars)]
        domains = {var: set(range(domain_size)) for var in variables}
        
        constraints = []
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                if random.random() < constraint_density:
                    # Restricción aleatoria (por ejemplo, !=)
                    constraints.append((
                        variables[i],
                        variables[j],
                        lambda v1, v2: v1 != v2
                    ))
        
        return CSPProblem(
            variables=variables,
            domains=domains,
            constraints=constraints
        )


# ============================================================================
# Funciones de Utilidad
# ============================================================================

def print_solution(solution: Dict[str, Any], problem_name: str = "CSP"):
    """
    Imprime una solución de forma legible.
    
    Args:
        solution: Diccionario de asignaciones
        problem_name: Nombre del problema
    """
    print(f"\n✅ Solución {problem_name}:")
    for var, val in sorted(solution.items()):
        print(f"   {var} = {val}")


def print_stats(stats: SolveStats, problem_name: str = "CSP"):
    """
    Imprime estadísticas de resolución.
    
    Args:
        stats: Estadísticas de resolución
        problem_name: Nombre del problema
    """
    print(f"\n📊 Estadísticas {problem_name}:")
    print(f"   Soluciones encontradas: {len(stats.solutions)}")
    print(f"   Nodos explorados: {stats.nodes_explored}")
    print(f"   Backtracks: {stats.backtracks}")
    print(f"   Consistente: {stats.consistent}")

