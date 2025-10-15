
# lattice_weaver/core/csp_problem.py

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
        metadata: Diccionario para almacenar metadatos adicionales sobre la restricción.
    """
    scope: FrozenSet[str]
    relation: Callable[..., bool]
    name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"Constraint(scope={self.scope}, name={self.name or 'anonymous'})"


@dataclass(frozen=True)
class AllDifferentConstraint(Constraint):
    """
    Representa una restricción AllDifferent en un CSP.
    Asegura que todas las variables en su scope tengan valores distintos.
    """
    def __post_init__(self):
        def alldiff_relation(*values) -> bool:
            return len(values) == len(set(values))

        object.__setattr__(self, 'relation', alldiff_relation)
        if self.name is None:
            object.__setattr__(self, 'name', f"AllDifferent({' '.join(sorted(list(self.scope)))})")

    def __repr__(self) -> str:
        return f"AllDifferentConstraint(scope={self.scope}, name={self.name})"


@dataclass(frozen=True)
class SumConstraint:
    """
    Representa una restricción de suma en un CSP.
    Asegura que la suma de los valores de las variables en su scope sea igual a un target_sum.
    Este dataclass contiene un objeto Constraint para manejar la lógica de la restricción.
    """
    scope: FrozenSet[str]
    target_sum: int
    name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    _constraint: Constraint = field(init=False, repr=False)

    def __post_init__(self):
        def sum_relation(*values) -> bool:
            return sum(values) == self.target_sum

        # Determine the name for the internal Constraint object
        # If name is not provided, generate a default name and set it for both SumConstraint and its internal Constraint
        if self.name is None:
            generated_name = f"Sum({{{', '.join(sorted(list(self.scope)))}}}) == {self.target_sum}"
            object.__setattr__(self, 'name', generated_name)
            constraint_name = generated_name
        else:
            constraint_name = self.name

        # Create the internal Constraint object
        object.__setattr__(self, '_constraint', Constraint(
            scope=self.scope,
            relation=sum_relation,
            name=constraint_name,
            metadata=self.metadata
        ))

    def __getattr__(self, name: str) -> Any:
        # Delegate attribute access to the internal Constraint object
        if name in ['scope', 'relation', 'name', 'metadata']:
            return getattr(self._constraint, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __repr__(self) -> str:
        return f"SumConstraint(scope={self.scope}, target_sum={self.target_sum}, name={self.name})"


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
        name: Nombre opcional del CSP.
        metadata: Diccionario para almacenar metadatos adicionales sobre el CSP.
    """
    variables: Set[str] = field(default_factory=set)
    domains: Dict[str, FrozenSet[Any]] = field(default_factory=dict)
    constraints: List[Constraint] = field(default_factory=list)
    name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_variable(self, name: str, domain: List[Any]):
        if name in self.variables:
            raise ValueError(f"Variable {name} already exists.")
        self.variables.add(name)
        self.domains[name] = frozenset(domain)

    def add_constraint(self, constraint: Constraint):
        # If the constraint is a SumConstraint, add its internal Constraint object
        if isinstance(constraint, SumConstraint):
            actual_constraint = constraint._constraint
        else:
            actual_constraint = constraint

        for var in actual_constraint.scope:
            if var not in self.variables:
                raise ValueError(f"Constraint references unknown variable {var}.")
        self.constraints.append(actual_constraint)

    def __post_init__(self):
        for var in self.variables:
            if var not in self.domains:
                raise ValueError(f"Variable {var} has no defined domain.")
        for constraint in self.constraints:
            for var in constraint.scope:
                if var not in self.variables:
                    raise ValueError(f"Constraint references unknown variable {var}.")

    def __repr__(self) -> str:
        return (
            f"CSP(name={self.name or 'anonymous'}"
            f",\n  variables={self.variables}"
            f",\n  domains={self.domains}"
            f",\n  constraints={self.constraints}\n"
            f")"
        )


from .simple_backtracking_solver import solve_csp_backtracking

def is_satisfiable(csp: CSP) -> bool:
    """
    Verifica si un CSP es satisfacible utilizando el solver de backtracking
    implementado en `simple_backtracking_solver`.
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
    if not csp.variables.issubset(assignment.keys()):
        return False

    for var, value in assignment.items():
        if var in csp.domains and value not in csp.domains[var]:
            return False

    for constraint in csp.constraints:
        try:
            values_in_scope = [assignment[var] for var in constraint.scope]
            if not constraint.relation(*values_in_scope):
                return False
        except KeyError:
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

    for i in range(n):
        for j in range(i + 1, n):
            qi = f"Q{i}"
            qj = f"Q{j}"

            constraints.append(Constraint(
                scope=frozenset({qi, qj}),
                relation=lambda val_i, val_j, captured_diff=abs(i - j): abs(val_i - val_j) != captured_diff,
                name=f"diag_{qi}_{qj}"
            ))
            
            constraints.append(Constraint(
                scope=frozenset({qi, qj}),
                relation=lambda val_i, val_j: val_i != val_j,
                name=f"row_col_{qi}_{qj}"
            ))

    return CSP(variables=variables, domains=domains, constraints=constraints, name=name, metadata={'abstraction_level': 0})


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
        
        constraints.append(Constraint(
            scope=frozenset({v1, v2}),
            relation=lambda x, y: x != y,
            name=f"neq_{v1}_{v2}"
        ))

    return CSP(variables=variables, domains=domains, constraints=constraints, name=name, metadata={'abstraction_level': 0})


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

