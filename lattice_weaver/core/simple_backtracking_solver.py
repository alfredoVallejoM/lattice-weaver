# lattice_weaver/core/simple_backtracking_solver.py

"""
Solver de Backtracking Simple para CSPs

Este módulo implementa un algoritmo de backtracking básico para resolver CSPs.
Se utiliza principalmente como un placeholder para `is_satisfiable` y para
generar soluciones en los tests de renormalización.
"""

from typing import Dict, Any, Optional, List
from collections import deque

from .csp_problem import CSP, Constraint

def solve_csp_backtracking(csp: CSP) -> Optional[Dict[str, Any]]:
    """
    Intenta encontrar una solución para el CSP usando backtracking.
    
    Args:
        csp: El CSP a resolver.
    
    Returns:
        Una asignación de variables a valores si se encuentra una solución,
        o None si el CSP no es satisfacible.
    """
    variables = list(csp.variables)
    domains = {var: list(dom) for var, dom in csp.domains.items()} # Convertir a lista para poder pop
    
    assignment: Dict[str, Any] = {}

    def select_unassigned_variable(current_assignment: Dict[str, Any]) -> Optional[str]:
        unassigned_vars = [v for v in variables if v not in current_assignment]
        if not unassigned_vars:
            return None
        # MRV heuristic: choose the variable with the fewest legal values
        return min(unassigned_vars, key=lambda var: len(domains[var]))

    def backtrack() -> bool:
        current_var = select_unassigned_variable(assignment)
        if current_var is None:
            return True  # Todas las variables asignadas, solución encontrada

        # Si el dominio de la variable está vacío, no hay solución
        if not domains[current_var]:
            return False

        for value in domains[current_var]:
            assignment[current_var] = value

            # Verificar consistencia con las restricciones que involucran la variable actual
            if is_consistent(assignment, csp.constraints):
                # Si es consistente, intentar asignar la siguiente variable
                if backtrack():
                    return True
            
            # Si no es consistente o el subproblema no tiene solución, deshacer asignación
            del assignment[current_var]
            
        return False # No se encontró ningún valor para la variable actual

    # Iniciar el backtracking desde la primera variable
    if backtrack():
        return assignment
    else:
        return None

def is_consistent(assignment: Dict[str, Any], constraints: List[Constraint]) -> bool:
    """
    Verifica si la asignación parcial es consistente con las restricciones.
    
    Solo verifica las restricciones donde todas sus variables ya están asignadas.
    """
    for constraint in constraints:
        # Verificar solo si todas las variables del scope de la restricción están en la asignación
        if all(var in assignment for var in constraint.scope):
            values_in_scope = [assignment[var] for var in constraint.scope]
            if not constraint.relation(*values_in_scope):
                return False
    return True


def generate_solutions_backtracking(csp: CSP, num_solutions: int = 1) -> List[Dict[str, Any]]:
    """
    Genera un número específico de soluciones para el CSP usando backtracking.
    
    Args:
        csp: El CSP a resolver.
        num_solutions: El número máximo de soluciones a encontrar.
    
    Returns:
        Una lista de asignaciones de variables a valores.
    """
    solutions: List[Dict[str, Any]] = []
    variables = list(csp.variables)
    domains = {var: list(dom) for var, dom in csp.domains.items()}
    
    assignment: Dict[str, Any] = {}

    def select_unassigned_variable_all(current_assignment: Dict[str, Any]) -> Optional[str]:
        unassigned_vars = [v for v in variables if v not in current_assignment]
        if not unassigned_vars:
            return None
        # MRV heuristic: choose the variable with the fewest legal values
        return min(unassigned_vars, key=lambda var: len(domains[var]))

    def backtrack_all() -> None:
        nonlocal solutions
        if len(solutions) >= num_solutions:
            return

        current_var = select_unassigned_variable_all(assignment)
        if current_var is None:
            solutions.append(assignment.copy()) # Encontró una solución
            return

        if not domains[current_var]:
            return

        # Make a copy of the domain to iterate over, as it might be modified by propagation
        for value in list(domains[current_var]):
            assignment[current_var] = value
            if is_consistent(assignment, csp.constraints):
                backtrack_all()
            del assignment[current_var]

    backtrack_all()
    return solutions

