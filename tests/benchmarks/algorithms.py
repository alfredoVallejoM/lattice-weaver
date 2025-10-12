"""
Implementaciones de algoritmos de resolución CSP del estado del arte.

Este módulo implementa algoritmos clásicos para comparación de rendimiento.
"""
from typing import Dict, Set, List, Tuple, Callable, Any, Optional
from dataclasses import dataclass
from copy import deepcopy
from lattice_weaver.formal.csp_integration import CSPProblem


@dataclass
class SolutionStats:
    """
    Estadísticas de resolución.
    
    Attributes:
        solutions: Lista de soluciones encontradas
        nodes_explored: Número de nodos explorados
        backtracks: Número de backtracks realizados
        time_ms: Tiempo de ejecución en milisegundos
    """
    solutions: List[Dict[str, Any]]
    nodes_explored: int
    backtracks: int
    time_ms: float = 0.0


class CSPSolver:
    """Clase base para solvers CSP."""
    
    def __init__(self):
        self.nodes_explored = 0
        self.backtracks = 0
    
    def solve(self, problem: CSPProblem, max_solutions: int = 1) -> SolutionStats:
        """
        Resuelve el problema CSP.
        
        Args:
            problem: Problema CSP
            max_solutions: Número máximo de soluciones a encontrar
        
        Returns:
            Estadísticas de resolución
        """
        raise NotImplementedError
    
    def is_consistent(self, var: str, value: Any, assignment: Dict[str, Any], problem: CSPProblem) -> bool:
        """
        Verifica si asignar value a var es consistente con assignment.
        
        Args:
            var: Variable a asignar
            value: Valor a asignar
            assignment: Asignación parcial actual
            problem: Problema CSP
        
        Returns:
            True si es consistente
        """
        # Verificar todas las restricciones que involucran var
        for v1, v2, predicate in problem.constraints:
            if v1 == var and v2 in assignment:
                if not predicate(value, assignment[v2]):
                    return False
            elif v2 == var and v1 in assignment:
                if not predicate(assignment[v1], value):
                    return False
        return True


class BacktrackingSolver(CSPSolver):
    """
    Solver con backtracking básico.
    
    Implementa el algoritmo de backtracking sin optimizaciones.
    """
    
    def solve(self, problem: CSPProblem, max_solutions: int = 1) -> SolutionStats:
        """Resuelve usando backtracking básico."""
        self.nodes_explored = 0
        self.backtracks = 0
        solutions = []
        
        def backtrack(assignment: Dict[str, Any]) -> bool:
            self.nodes_explored += 1
            
            # Solución completa
            if len(assignment) == len(problem.variables):
                solutions.append(dict(assignment))
                return len(solutions) >= max_solutions
            
            # Seleccionar variable no asignada
            var = next(v for v in problem.variables if v not in assignment)
            
            # Probar cada valor del dominio
            for value in problem.domains[var]:
                if self.is_consistent(var, value, assignment, problem):
                    assignment[var] = value
                    
                    if backtrack(assignment):
                        return True
                    
                    del assignment[var]
                    self.backtracks += 1
            
            return False
        
        backtrack({})
        
        return SolutionStats(
            solutions=solutions,
            nodes_explored=self.nodes_explored,
            backtracks=self.backtracks
        )


class ForwardCheckingSolver(CSPSolver):
    """
    Solver con Forward Checking.
    
    Implementa backtracking con propagación forward checking.
    """
    
    def solve(self, problem: CSPProblem, max_solutions: int = 1) -> SolutionStats:
        """Resuelve usando forward checking."""
        self.nodes_explored = 0
        self.backtracks = 0
        solutions = []
        
        # Inicializar dominios
        domains = {var: set(problem.domains[var]) for var in problem.variables}
        
        def forward_check(var: str, value: Any, domains: Dict[str, Set[Any]]) -> Optional[Dict[str, Set[Any]]]:
            """
            Realiza forward checking y retorna nuevos dominios o None si hay conflicto.
            """
            new_domains = {v: set(d) for v, d in domains.items()}
            
            # Para cada variable no asignada
            for other_var in problem.variables:
                if other_var == var or other_var not in new_domains:
                    continue
                
                # Filtrar valores inconsistentes
                values_to_remove = set()
                for other_value in new_domains[other_var]:
                    consistent = True
                    
                    # Verificar restricciones entre var y other_var
                    for v1, v2, predicate in problem.constraints:
                        if (v1 == var and v2 == other_var):
                            if not predicate(value, other_value):
                                consistent = False
                                break
                        elif (v2 == var and v1 == other_var):
                            if not predicate(other_value, value):
                                consistent = False
                                break
                    
                    if not consistent:
                        values_to_remove.add(other_value)
                
                new_domains[other_var] -= values_to_remove
                
                # Si un dominio queda vacío, hay conflicto
                if not new_domains[other_var]:
                    return None
            
            return new_domains
        
        def backtrack(assignment: Dict[str, Any], domains: Dict[str, Set[Any]]) -> bool:
            self.nodes_explored += 1
            
            # Solución completa
            if len(assignment) == len(problem.variables):
                solutions.append(dict(assignment))
                return len(solutions) >= max_solutions
            
            # Seleccionar variable no asignada con dominio más pequeño (MRV)
            unassigned = [v for v in problem.variables if v not in assignment]
            var = min(unassigned, key=lambda v: len(domains[v]))
            
            # Probar cada valor del dominio
            for value in list(domains[var]):
                if self.is_consistent(var, value, assignment, problem):
                    # Forward checking
                    new_domains = forward_check(var, value, domains)
                    
                    if new_domains is not None:
                        assignment[var] = value
                        del new_domains[var]
                        
                        if backtrack(assignment, new_domains):
                            return True
                        
                        del assignment[var]
                        self.backtracks += 1
            
            return False
        
        backtrack({}, domains)
        
        return SolutionStats(
            solutions=solutions,
            nodes_explored=self.nodes_explored,
            backtracks=self.backtracks
        )


class AC3Solver(CSPSolver):
    """
    Solver con AC-3 (Arc Consistency 3).
    
    Implementa el algoritmo AC-3 para establecer arc consistency.
    """
    
    def solve(self, problem: CSPProblem, max_solutions: int = 1) -> SolutionStats:
        """Resuelve usando AC-3 + backtracking."""
        self.nodes_explored = 0
        self.backtracks = 0
        solutions = []
        
        def revise(xi: str, xj: str, domains: Dict[str, Set[Any]]) -> bool:
            """
            Revisa el arco (xi, xj) y elimina valores inconsistentes de xi.
            
            Returns:
                True si se eliminó algún valor
            """
            revised = False
            values_to_remove = set()
            
            # Encontrar restricciones relevantes entre xi y xj
            relevant_constraints = []
            for v1, v2, predicate in problem.constraints:
                if (v1 == xi and v2 == xj):
                    relevant_constraints.append((v1, v2, predicate))
                elif (v2 == xi and v1 == xj):
                    relevant_constraints.append((v2, v1, predicate))
            
            # Si no hay restricciones entre xi y xj, no hay nada que revisar
            if not relevant_constraints:
                return False
            
            for vi in list(domains[xi]):
                # Verificar si existe algún vj que satisfaga TODAS las restricciones
                satisfies = False
                for vj in domains[xj]:
                    consistent = True
                    
                    # Verificar todas las restricciones relevantes
                    for v1, v2, predicate in relevant_constraints:
                        if v1 == xi and v2 == xj:
                            if not predicate(vi, vj):
                                consistent = False
                                break
                        elif v1 == xj and v2 == xi:
                            if not predicate(vj, vi):
                                consistent = False
                                break
                    
                    if consistent:
                        satisfies = True
                        break
                
                if not satisfies:
                    values_to_remove.add(vi)
                    revised = True
            
            domains[xi] -= values_to_remove
            return revised
        
        def ac3(domains: Dict[str, Set[Any]]) -> bool:
            """
            Establece arc consistency.
            
            Returns:
                True si es consistente, False si hay conflicto
            """
            # Cola de arcos (solo variables no asignadas)
            queue = []
            for v1, v2, _ in problem.constraints:
                if v1 in domains and v2 in domains:
                    queue.append((v1, v2))
                    queue.append((v2, v1))
            
            while queue:
                xi, xj = queue.pop(0)
                
                # Verificar que ambas variables aún están en dominios
                if xi not in domains or xj not in domains:
                    continue
                
                if revise(xi, xj, domains):
                    if not domains[xi]:
                        return False
                    
                    # Agregar arcos vecinos a la cola
                    for v1, v2, _ in problem.constraints:
                        if v1 in domains and v2 in domains:
                            if v2 == xi and v1 != xj:
                                queue.append((v1, xi))
                            elif v1 == xi and v2 != xj:
                                queue.append((v2, xi))
            
            return True
        
        def backtrack(assignment: Dict[str, Any], domains: Dict[str, Set[Any]]) -> bool:
            self.nodes_explored += 1
            
            # Solución completa
            if len(assignment) == len(problem.variables):
                solutions.append(dict(assignment))
                return len(solutions) >= max_solutions
            
            # Seleccionar variable no asignada
            unassigned = [v for v in problem.variables if v not in assignment]
            var = min(unassigned, key=lambda v: len(domains[v]))
            
            # Probar cada valor del dominio
            for value in list(domains[var]):
                new_domains = {v: set(d) for v, d in domains.items()}
                new_domains[var] = {value}
                
                # Aplicar AC-3
                if ac3(new_domains):
                    assignment[var] = value
                    del new_domains[var]
                    
                    if backtrack(assignment, new_domains):
                        return True
                    
                    del assignment[var]
                    self.backtracks += 1
            
            return False
        
        # Inicializar dominios
        domains = {var: set(problem.domains[var]) for var in problem.variables}
        
        # Aplicar AC-3 inicial
        if ac3(domains):
            backtrack({}, domains)
        
        return SolutionStats(
            solutions=solutions,
            nodes_explored=self.nodes_explored,
            backtracks=self.backtracks
        )


# Diccionario de algoritmos disponibles
ALGORITHMS = {
    "backtracking": BacktrackingSolver,
    "forward_checking": ForwardCheckingSolver,
    "ac3": AC3Solver,
}


def get_solver(algorithm: str) -> CSPSolver:
    """
    Obtiene un solver por nombre.
    
    Args:
        algorithm: Nombre del algoritmo
    
    Returns:
        Instancia del solver
    
    Raises:
        ValueError: Si el algoritmo no existe
    """
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Algoritmo '{algorithm}' no encontrado. Disponibles: {list(ALGORITHMS.keys())}")
    
    return ALGORITHMS[algorithm]()

