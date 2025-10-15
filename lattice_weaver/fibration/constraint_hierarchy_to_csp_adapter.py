from typing import Dict, List, Any, Tuple, Callable, FrozenSet, Set
from ..core.csp_problem import CSP, Constraint as CSPConstraint
from .constraint_hierarchy import ConstraintHierarchy, Constraint, Hardness

class ConstraintHierarchyToCSPAdapter:
    """
    Adaptador bidireccional entre la ConstraintHierarchy de Fibration Flow y la representación
    de Problemas de Satisfacción de Restricciones (CSP).

    Permite convertir una ConstraintHierarchy en un objeto CSP para ser resuelto por solvers CSP,
    y reconstruir soluciones CSP en el formato de Fibration Flow.
    """

    def __init__(self):
        """
        Inicializa el adaptador.
        """
        pass

    def convert_hierarchy_to_csp(self, hierarchy: ConstraintHierarchy, variables_domains: Dict[str, List[Any]]) -> Tuple[CSP, Dict[str, Any]]:
        """
        Convierte una ConstraintHierarchy en un objeto CSP.

        Solo las restricciones HARD de la jerarquía se convertirán en restricciones CSP.
        Las restricciones SOFT serán ignoradas.

        Args:
            hierarchy (ConstraintHierarchy): La jerarquía de restricciones de Fibration Flow.
            variables_domains (Dict[str, List[Any]]): Los dominios de las variables en formato de lista.

        Returns:
            Tuple[CSP, Dict[str, Any]]: Una tupla que contiene:
                - `CSP`: El Problema de Satisfacción de Restricciones convertido.
                - `Dict[str, Any]`: Metadatos de mapeo para la descompilación.
        """
        csp_variables: Set[str] = set(variables_domains.keys())
        csp_domains: Dict[str, FrozenSet[Any]] = {var: frozenset(domains) for var, domains in variables_domains.items()}
        csp_constraints: List[CSPConstraint] = []

        # Recopilar todas las restricciones HARD de la jerarquía
        for level_constraints in hierarchy.get_all_constraints().values():
            for fibration_constraint in level_constraints:
                if fibration_constraint.hardness == Hardness.HARD:
                    # Envolver el predicado de Fibration Flow en una relación compatible con CSP
                    def wrapped_relation(original_predicate: Callable[[Dict[str, Any]], Tuple[bool, float]], scope_vars: Tuple[str, ...], *values) -> bool:
                        assignment = dict(zip(scope_vars, values))
                        satisfied, _ = original_predicate(assignment)
                        return satisfied

                    csp_constraint = CSPConstraint(
                        scope=frozenset(fibration_constraint.variables),
                        relation=lambda *args, pred=fibration_constraint.predicate, scope=fibration_constraint.variables: wrapped_relation(pred, scope, *args),
                        name=fibration_constraint.metadata.get("name", fibration_constraint.expression or "anonymous_fibration_constraint"),
                        metadata={
                            "fibration_level": fibration_constraint.level.value,
                            "fibration_hardness": fibration_constraint.hardness.value,
                            **fibration_constraint.metadata
                        }
                    )
                    csp_constraints.append(csp_constraint)

        csp = CSP(
            variables=csp_variables,
            domains=csp_domains,
            constraints=csp_constraints,
            name="ConvertedFromFibrationHierarchy",
            metadata={
                "original_hierarchy_name": hierarchy.name if hasattr(hierarchy, 'name') else "anonymous_hierarchy"
            }
        )

        # Los metadatos de mapeo son simples en este caso, ya que las variables no se transforman.
        compilation_metadata = {
            "original_variables": sorted(list(csp_variables)),
            "original_domains": {var: list(domains) for var, domains in csp_domains.items()}
        }

        return csp, compilation_metadata

    def convert_csp_solution_to_hierarchy_solution(self, csp_solution: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convierte una solución CSP a un formato de solución de Fibration Flow.
        Dado que este adaptador no realiza transformaciones complejas de variables,
        la conversión es un mapeo directo.

        Args:
            csp_solution (Dict[str, Any]): La solución obtenida de un solver CSP.
            metadata (Dict[str, Any]): Metadatos de mapeo generados durante la compilación.

        Returns:
            Dict[str, Any]: La solución en el formato original de Fibration Flow.
        """
        # En este adaptador simple, la solución CSP ya está en el formato de Fibration Flow,
        # ya que las variables no se agrupan ni se transforman.
        # Los metadatos podrían usarse para validación si fuera necesario.
        return csp_solution

