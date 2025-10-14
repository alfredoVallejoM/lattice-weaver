from constraint import Problem, AllDifferentConstraint, Constraint
from typing import Dict, List, Any, Tuple

from ..fibration.constraint_hierarchy import ConstraintHierarchy, ConstraintLevel, Hardness

class PythonConstraintAdapter:
    """
    Adaptador para el solver `python-constraint`.
    Convierte un problema definido con `ConstraintHierarchy` a un formato compatible con `python-constraint`.
    `python-constraint` solo maneja restricciones HARD.
    """
    def __init__(self, variables: List[str], domains: Dict[str, List[Any]], hierarchy: ConstraintHierarchy):
        self.variables = variables
        self.domains = domains
        self.hierarchy = hierarchy
        self.problem = Problem()

        # Añadir variables y dominios
        for var in self.variables:
            self.problem.addVariable(var, self.domains[var])

        # Añadir restricciones HARD de la jerarquía
        self._add_hard_constraints()

    def _add_hard_constraints(self):
        """
        Añade las restricciones HARD de la ConstraintHierarchy al problema de python-constraint.
        """
        for level in ConstraintLevel:
            for constraint in self.hierarchy.get_constraints_at_level(level):
                if constraint.hardness == Hardness.HARD:
                    if constraint.metadata.get("name") == "all_different" and len(constraint.variables) > 1:
                        self.problem.addConstraint(AllDifferentConstraint(), constraint.variables)
                    else:
                        def make_wrapper(predicate, variables):
                            def wrapper(*args):
                                assignment = {var: val for var, val in zip(variables, args)}
                                return predicate(assignment)
                            return wrapper
                        
                        self.problem.addConstraint(make_wrapper(constraint.predicate, constraint.variables), constraint.variables)

    def solve(self) -> List[Dict[str, Any]]:
        """
        Resuelve el problema y retorna todas las soluciones encontradas.
        """
        return self.problem.getSolutions()

