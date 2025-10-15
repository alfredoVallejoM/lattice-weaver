"""
Refactorización del Puente CSP-Cúbico para la Arquitectura v8.0
"""

from typing import Dict, Any, List, Tuple
from ..core.csp_problem import CSP, Constraint, AllDifferentConstraint
from .cubical_types import (
    CubicalType, CubicalSubtype, CubicalSigmaType, 
    CubicalFiniteType, CubicalPredicate, CubicalTerm, 
    VariableTerm, ValueTerm, CubicalNegation, CubicalAnd, CubicalPath
)

class CSPToCubicalBridge:
    """Traduce un CSP a su representación cúbica genérica."""

    def to_cubical(self, csp: CSP) -> CubicalSubtype:
        """Convierte un CSP en un CubicalSubtype."""
        search_space = self._translate_search_space(csp)
        predicate = self._translate_constraints(csp)
        return CubicalSubtype(search_space, predicate)

    def _translate_search_space(self, csp: CSP) -> CubicalSigmaType:
        components = []
        for var_name in sorted(csp.variables):
            domain = csp.domains[var_name]
            components.append((var_name, CubicalFiniteType(len(domain))))
        return CubicalSigmaType(components)

    def _translate_constraints(self, csp: CSP) -> CubicalPredicate:
        # Lógica para combinar todas las restricciones en un único predicado.
        # Por ahora, un placeholder.
        predicates = []
        for constraint in csp.constraints:
            predicates.append(self._translate_constraint(constraint))
        
        # Combina todos los predicados en un único CubicalAnd.
        # Si no hay predicados, retorna un predicado que siempre es verdadero.
        if not predicates:
            return CubicalPath(ValueTerm(True), ValueTerm(True))
        elif len(predicates) == 1:
            return predicates[0]
        else:
            return CubicalAnd(frozenset(predicates))

    def _translate_constraint(self, constraint: Constraint) -> CubicalPredicate:
        if isinstance(constraint, AllDifferentConstraint):
            return self._translate_alldifferent_constraint(constraint)
        else:
            # Placeholder para otros tipos de restricciones
            return CubicalPath(ValueTerm(True), ValueTerm(True))

    def _translate_alldifferent_constraint(self, constraint: AllDifferentConstraint) -> CubicalPredicate:
        # Para AllDifferent, necesitamos que todos los pares de variables sean diferentes.
        # Esto se traduce en una conjunción de negaciones de predicados de igualdad.
        # Creamos una lista de predicados de desigualdad para cada par de variables.
        
        scope_list = sorted(list(constraint.scope))
        inequality_predicates = []
        for i in range(len(scope_list)):
            for j in range(i + 1, len(scope_list)):
                var1 = VariableTerm(scope_list[i])
                var2 = VariableTerm(scope_list[j])
                equality_predicate = CubicalPath(var1, var2)
                inequality_predicates.append(CubicalNegation(equality_predicate))
        
        # Si no hay predicados (menos de 2 variables), retornamos un predicado verdadero
        if not inequality_predicates:
            return CubicalPath(ValueTerm(True), ValueTerm(True))

        # Combinamos todos los predicados de desigualdad con CubicalAnd
        return CubicalAnd(frozenset(inequality_predicates))

