# lattice_weaver/arc_engine/csp_problem.py

"""
Definición Base de Problemas CSP

Este módulo define las estructuras de datos fundamentales para representar
un Problema de Satisfacción de Restricciones (CSP), incluyendo variables,
dominios y restricciones.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Set, Tuple, Callable, Optional
import itertools


@dataclass(frozen=True)
class Constraint:
    """
    Representa una restricción en un CSP.
    
    Attributes:
        scope: Un frozenset de nombres de variables involucradas en la restricción.
        relation: Una función booleana que toma los valores de las variables
                  en el orden de `scope` y retorna True si la restricción se satisface.
        name: Nombre opcional de la restricción para depuración o trazabilidad.
    """
    scope: FrozenSet[str]
    relation: Callable[..., bool]
    name: Optional[str] = None

    def __repr__(self) -> str:
        return f"Constraint(scope={self.scope}, name={self.name or 'anonymous'})"


@dataclass
class CSP:
    """
    Representa un Problema de Satisfacción de Restricciones (CSP).
    
    Attributes:
        variables: Un set de nombres de variables.
        domains: Un diccionario que mapea cada nombre de variable a su dominio
                 (un frozenset de valores posibles).
        constraints: Una lista de objetos Constraint que definen las relaciones
                     entre las variables.
    """
    variables: Set[str]
    domains: Dict[str, FrozenSet[Any]]
    constraints: List[Constraint]
    name: Optional[str] = None

    def __post_init__(self):
        # Asegurar que todas las variables tienen un dominio
        for var in self.variables:
            if var not in self.domains:
                raise ValueError(f"Variable {var} has no defined domain.")
        # Asegurar que todas las restricciones referencian variables existentes
        for constraint in self.constraints:
            for var in constraint.scope:
                if var not in self.variables:
                    raise ValueError(f"Constraint references unknown variable {var}.")

    def __repr__(self) -> str:
        return (
            f"CSP(name={self.name or 'anonymous'},\n"
            f"  variables={self.variables},\n"
            f"  domains={self.domains},\n"
            f"  constraints={self.constraints}\n"
            f")"
        )


from .simple_backtracking_solver import solve_csp_backtracking

def is_satisfiable(csp: CSP) -> bool:
    """
    Verifica si un CSP es satisfacible utilizando el solver de backtracking.
    """
    return solve_csp_backtracking(csp) is not None


def verify_solution(csp: CSP, assignment: Dict[str, Any]) -> bool:
    """
    Verifica si una asignación dada es una solución válida para el CSP.
    
    Args:
        csp: El CSP a verificar.
        assignment: Un diccionario que mapea variables a valores.
    
    Returns:
        True si la asignación satisface todas las restricciones del CSP, False en caso contrario.
    """
    # 1. Verificar que todas las variables del CSP están en la asignación
    if not csp.variables.issubset(assignment.keys()):
        return False

    # 2. Verificar que los valores asignados están dentro de los dominios
    for var, value in assignment.items():
        if var in csp.domains and value not in csp.domains[var]:
            return False

    # 3. Verificar que todas las restricciones se satisfacen
    for constraint in csp.constraints:
        # Obtener los valores de las variables en el scope de la restricción
        try:
            values_in_scope = [assignment[var] for var in constraint.scope]
        except KeyError:
            # Si alguna variable en el scope de la restricción no está en la asignación,
            # la asignación no es completa para esta restricción.
            return False
        
        # Evaluar la restricción
        if not constraint.relation(*values_in_scope):
            return False
            
    return True


def generate_nqueens(n: int, name: Optional[str] = None) -> CSP:
    """
    Genera un problema CSP para el problema de las N-Reinas.
    
    Args:
        n: El número de reinas (y el tamaño del tablero).
    
    Returns:
        Un objeto CSP que representa el problema de las N-Reinas.
    """
    variables = {f"Q{i}" for i in range(n)}
    domains = {f"Q{i}": frozenset(range(n)) for i in range(n)}
    constraints = []

    # Restricciones de fila y columna (implícitas por el dominio y la asignación)
    # Restricciones de diagonal
    for i in range(n):
        for j in range(i + 1, n):
            qi = f"Q{i}"
            qj = f"Q{j}"

            # No en la misma fila (ya cubierto por dominio)
            # No en la misma columna (ya cubierto por asignación de valores distintos)

            # No en la misma diagonal
            constraints.append(Constraint(
                scope=frozenset({qi, qj}),
                relation=lambda val_i, val_j, captured_diff=abs(i - j): abs(val_i - val_j) != captured_diff,
                name=f"diag_{qi}_{qj}"
            ))
            
            # No en la misma fila (explícitamente para el solver)
            constraints.append(Constraint(
                scope=frozenset({qi, qj}),
                relation=lambda val_i, val_j: val_i != val_j,
                name=f"row_col_{qi}_{qj}"
            ))

    return CSP(variables=variables, domains=domains, constraints=constraints, name=name)


def generate_random_csp(num_vars: int, domain_size: int, num_constraints: int, name: Optional[str] = None) -> CSP:
    """
    Genera un CSP aleatorio para pruebas.
    
    Args:
        num_vars: Número de variables.
        domain_size: Tamaño del dominio para cada variable.
        num_constraints: Número de restricciones binarias aleatorias.
    
    Returns:
        Un objeto CSP aleatorio.
    """
    import random

    variables = {f"v{i}" for i in range(num_vars)}
    domains = {var: frozenset(range(domain_size)) for var in variables}
    constraints = []

    all_pairs = list(itertools.combinations(variables, 2))
    random.shuffle(all_pairs)

    for i in range(min(num_constraints, len(all_pairs))):
        v1, v2 = all_pairs[i]
        
        # Crear una restricción binaria aleatoria (ej. v1 != v2, v1 < v2, etc.)
        # Para simplificar, usaremos solo v1 != v2
        constraints.append(Constraint(
            scope=frozenset({v1, v2}),
            relation=lambda x, y: x != y,
            name=f"neq_{v1}_{v2}"
        ))

    return CSP(variables=variables, domains=domains, constraints=constraints, name=name)


def solve_subproblem_exhaustive(subproblem: Dict) -> FrozenSet[Tuple]:
    """
    Resuelve un subproblema de forma exhaustiva.
    
    Args:
        subproblem: Diccionario con 'variables', 'domains', 'constraints'.
    
    Returns:
        Un frozenset de tuplas, donde cada tupla es una configuración válida.
    """
    variables = subproblem["variables"]
    domains = subproblem["domains"]
    constraints = subproblem["constraints"]

    domain_lists = [list(domains[var]) for var in variables]
    all_configs = itertools.product(*domain_lists)

    valid_configs = []
    for config in all_configs:
        assignment = dict(zip(variables, config))
        is_valid = True
        for constraint in constraints:
            try:
                values_in_scope = [assignment[var] for var in constraint.scope]
                if not constraint.relation(*values_in_scope):
                    is_valid = False
                    break
            except KeyError:
                is_valid = False # Should not happen if subproblem is well-formed
                break
        if is_valid:
            valid_configs.append(config)
    
    return frozenset(valid_configs)

