from typing import Dict, List, Any, Tuple, Callable, FrozenSet
from ..core.csp_problem import CSP, Constraint as CSPConstraint
from .constraint_hierarchy import ConstraintHierarchy, Constraint, ConstraintLevel, Hardness

class CSPToConstraintHierarchyAdapter:
    """
    Adaptador bidireccional entre la representación de Problemas de Satisfacción de Restricciones (CSP)
    y la ConstraintHierarchy de Fibration Flow.

    Permite convertir un objeto CSP en una ConstraintHierarchy para ser procesado por Fibration Flow,
    y reconstruir soluciones de Fibration Flow en el formato original de CSP.
    """

    def __init__(self):
        """
        Inicializa el adaptador.
        """
        pass

    def convert_csp_to_hierarchy(self, csp: CSP) -> Tuple[ConstraintHierarchy, Dict[str, List[Any]], Dict[str, Any]]:
        """
        Convierte un objeto CSP en una ConstraintHierarchy compatible con Fibration Flow.

        Args:
            csp (CSP): El Problema de Satisfacción de Restricciones original.

        Returns:
            Tuple[ConstraintHierarchy, Dict[str, List[Any]], Dict[str, Any]]: Una tupla que contiene:
                - `ConstraintHierarchy`: La jerarquía de restricciones convertida.
                - `Dict[str, List[Any]]`: Los dominios de las variables en formato de lista.
                - `Dict[str, Any]`: Metadatos de mapeo para la descompilación.
        """
        fibration_hierarchy = ConstraintHierarchy()
        fibration_domains: Dict[str, List[Any]] = {}
        
        # Convertir dominios de variables
        for var, domain_set in csp.domains.items():
            fibration_domains[var] = sorted(list(domain_set))

        # Convertir restricciones
        for csp_constraint in csp.constraints:
            # Envolver la relación del CSP en un predicado compatible con ConstraintHierarchy
            def wrapped_predicate(assignment: Dict[str, Any], original_relation: Callable[..., bool] = csp_constraint.relation, scope: FrozenSet[str] = csp_constraint.scope) -> Tuple[bool, float]:
                # Asegurarse de que todas las variables del scope estén en la asignación
                if not scope.issubset(assignment.keys()):
                    # Si no todas las variables están asignadas, no se puede evaluar completamente.
                    # Se asume que no hay violación por el momento (comportamiento para asignaciones parciales).
                    return True, 0.0
                
                # Asegurarse de que los valores se pasan a la relación en el orden correcto del scope
                # El orden de los argumentos para la relación original debe coincidir con el orden en csp_constraint.scope
                # El orden de los argumentos para la relación original debe coincidir con el orden en csp_constraint.scope.
                # Dado que `csp_constraint.scope` es un `frozenset`, su iteración no garantiza un orden consistente.
                # Para asegurar un orden predecible, ordenamos las variables del scope.
                ordered_scope = sorted(list(csp_constraint.scope))
                values_in_scope = [assignment[var] for var in ordered_scope]
                return original_relation(*values_in_scope)

            # Crear la Constraint de Fibration Flow
            fibration_constraint = Constraint(
                level=ConstraintLevel.LOCAL,  # Por defecto, todas las restricciones CSP son LOCAL
                variables=tuple(sorted(list(csp_constraint.scope))), # Asegurar orden consistente
                predicate=wrapped_predicate,
                hardness=Hardness.HARD,       # Las restricciones CSP son inherentemente HARD
                metadata={
                    "original_csp_name": csp.name,
                    "original_constraint_name": csp_constraint.name,
                    **csp_constraint.metadata
                }
            )
            fibration_hierarchy.add_constraint(fibration_constraint)

        # Los metadatos de mapeo son simples en este caso, ya que las variables no se transforman.
        compilation_metadata = {
            "original_variables": list(csp.variables),
            "original_domains": {var: list(domain_set) for var, domain_set in csp.domains.items()}
        }

        return fibration_hierarchy, fibration_domains, compilation_metadata

    def convert_hierarchy_solution_to_csp_solution(self, fibration_solution: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convierte una solución de Fibration Flow a un formato de solución CSP.
        Dado que este adaptador no realiza transformaciones complejas de variables,
        la conversión es un mapeo directo.

        Args:
            fibration_solution (Dict[str, Any]): La solución obtenida de Fibration Flow.
            metadata (Dict[str, Any]): Metadatos de mapeo generados durante la compilación.

        Returns:
            Dict[str, Any]: La solución en el formato original de CSP.
        """
        # En este adaptador simple, la solución de Fibration Flow ya está en el formato de CSP,
        # ya que las variables no se agrupan ni se transforman.
        # Los metadatos podrían usarse para validación si fuera necesario.
        return fibration_solution

