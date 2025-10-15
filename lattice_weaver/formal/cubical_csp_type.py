"""
Cubical CSP Type: Tipos Cúbicos derivados de Problemas CSP

Este módulo implementa la representación de problemas de satisfacción de
restricciones (CSP) como tipos cúbicos en la teoría de tipos homotópica.

La idea central es traducir un CSP a un tipo Sigma dependiente donde:
- Cada variable del CSP se convierte en un componente del Sigma
- Cada dominio se convierte en un tipo finito
- Cada restricción se convierte en una proposición (tipo con 0 o 1 habitante)

Ejemplo:
    CSP: X < Y, dominios {X: {1,2,3}, Y: {1,2,3}}
    →
    Tipo: Σ (x : {1,2,3}) Σ (y : {1,2,3}) (x < y)

Autor: LatticeWeaver Team (Track: CSP-Cubical Integration)
Fecha: 12 de Octubre de 2025
Versión: 1.0
"""

from typing import Dict, List, Set, Any, Optional, Callable
from dataclasses import dataclass, field
import logging

from .cubical_types import CubicalType, CubicalFiniteType, CubicalSigmaType, CubicalPredicate, CubicalSubtype, CubicalTerm, VariableTerm, ValueTerm
from .cubical_syntax import (
    Type, SigmaType, Universe, TypeVar, PathType,
    Term, Var, Pair, UnitType, Unit
)

logger = logging.getLogger(__name__)


# ============================================================================
# Tipos Finitos (para dominios CSP)
# ============================================================================

@dataclass(frozen=True)
class FiniteType(CubicalFiniteType):
    """
    Tipo finito con un conjunto explícito de valores.
    
    Representa dominios de variables CSP como tipos finitos.
    
    Attributes:
        name: Nombre del tipo (ej: "Domain_X")
        values: Conjunto de valores del dominio
    
    Examples:
        >>> # Dominio {1, 2, 3}
        >>> dom = FiniteType("Domain_X", {1, 2, 3})
        >>> str(dom)
        '{1, 2, 3}'
    """
    name: str
    values: frozenset
    
    def __post_init__(self):
        # Asegurar que values es frozenset para inmutabilidad
        if not isinstance(self.values, frozenset):
            object.__setattr__(self, 'values', frozenset(self.values))
    
    def free_vars(self) -> Set[str]:
        return set()
    
    def substitute(self, var: str, term: Term) -> 'Type':
        return self
    
    def __str__(self) -> str:
        # Ordenar valores para representación consistente
        sorted_vals = sorted(self.values, key=lambda x: (type(x).__name__, str(x)))
        vals_str = ', '.join(str(v) for v in sorted_vals)
        return f"{{{vals_str}}}"
    
    def __hash__(self):
        return hash((self.name, self.values))


@dataclass(frozen=True)
class PropositionType(CubicalPredicate):
    """
    Tipo proposición representando una restricción CSP.
    
    Una proposición es un tipo con 0 o 1 habitante:
    - Si la restricción se satisface, tiene 1 habitante (prueba)
    - Si no se satisface, tiene 0 habitantes (es vacío)
    
    Attributes:
        constraint_name: Nombre de la restricción
        variables: Variables involucradas
        predicate: Predicado que define la restricción
    
    Examples:
        >>> # Restricción X < Y
        >>> prop = PropositionType("X_lt_Y", ["X", "Y"], lambda x, y: x < y)
    """
    constraint_name: str
    variables: tuple  # Tupla de nombres de variables
    predicate: Callable  # Función que verifica la restricción
    
    def __post_init__(self):
        # Asegurar que variables es tupla para hashability
        if not isinstance(self.variables, tuple):
            object.__setattr__(self, 'variables', tuple(self.variables))
    
    def free_vars(self) -> Set[str]:
        return set(self.variables)
    
    def substitute(self, var: str, term: Term) -> 'Type':
        # Las proposiciones no se sustituyen directamente
        return self
    
    def __str__(self) -> str:
        vars_str = ', '.join(self.variables)
        return f"({self.constraint_name}({vars_str}))"
    
    def __hash__(self):
        return hash((self.constraint_name, self.variables))
    
    def check(self, *values: Any) -> bool:
        """
        Verifica si los valores satisfacen la restricción.
        
        Args:
            *values: Valores de las variables (en orden)
        
        Returns:
            True si satisface la restricción, False en caso contrario
        """
        try:
            return self.predicate(*values)
        except Exception as e:
            logger.error(f"Error al verificar restricción {self.constraint_name}: {e}")
            return False


# ============================================================================
# CubicalCSPType - Tipo Cúbico derivado de CSP
# ============================================================================

@dataclass
class CubicalCSPType:
    """
    Tipo cúbico representando un espacio de soluciones CSP.
    
    Traduce un problema CSP a un tipo Sigma dependiente de la forma:
    
        Σ (x1 : D1) Σ (x2 : D2) ... Σ (xn : Dn) (C1 × C2 × ... × Cm)
    
    Donde:
    - Di son tipos finitos (dominios)
    - Ci son proposiciones (restricciones)
    
    Attributes:
        variables: Lista de nombres de variables
        domain_types: Mapa variable → tipo de dominio
        constraint_props: Lista de proposiciones de restricciones
        solution_type: Tipo Sigma completo
    
    Examples:
        >>> # CSP: X < Y, dominios {1,2,3}
        >>> csp_type = CubicalCSPType.from_csp_problem(csp)
        >>> print(csp_type.solution_type)
        Σ(x : {1, 2, 3}). Σ(y : {1, 2, 3}). (x < y)
    """
    
    variables: List[str]
    domain_types: Dict[str, FiniteType]
    constraint_props: List[PropositionType]
    solution_type: Optional[Type] = None
    
    # Caché para optimización
    _type_cache: Dict[str, Type] = field(default_factory=dict, repr=False)
    
    def __post_init__(self):
        """Construye el tipo Sigma completo."""
        if self.solution_type is None:
            self.solution_type = self._build_sigma_type()
    
    def _build_sigma_type(self) -> Type:
        """
        Construye el tipo Sigma completo.
        
        Returns:
            Tipo Sigma anidado representando el espacio de soluciones
        """
        if not self.variables:
            # Sin variables, tipo trivial
            return UnitType()
        
        # Construir tipo de restricciones (producto de proposiciones)
        constraints_type = self._build_constraints_type()
        
        # Construir Sigma anidado desde la última variable hacia la primera
        current_type = constraints_type
        
        for var in reversed(self.variables):
            domain_type = self.domain_types[var]
            current_type = SigmaType(var, domain_type, current_type)
        
        return current_type
    
    def _build_constraints_type(self) -> Type:
        """
        Construye el tipo producto de todas las restricciones.
        
        Returns:
            Tipo producto C1 × C2 × ... × Cm
        """
        if not self.constraint_props:
            # Sin restricciones, tipo trivial (siempre satisfecho)
            return UnitType()
        
        if len(self.constraint_props) == 1:
            return self.constraint_props[0]
        
        # Construir producto anidado
        current_type = self.constraint_props[-1]
        for prop in reversed(self.constraint_props[:-1]):
            # Usar SigmaType con variable dummy para representar producto
            current_type = SigmaType("_", prop, current_type)
        
        return current_type
    
    @classmethod
    def from_csp_problem(
        cls,
        variables: List[str],
        domains: Dict[str, Set[Any]],
        constraints: List[Dict[str, Any]]
    ) -> 'CubicalCSPType':
        """
        Construye un CubicalCSPType desde un problema CSP.
        
        Args:
            variables: Lista de nombres de variables
            domains: Mapa variable → conjunto de valores
            constraints: Lista de restricciones, cada una con:
                - 'variables': lista de variables involucradas
                - 'predicate': función que verifica la restricción
                - 'name': nombre opcional de la restricción
        
        Returns:
            CubicalCSPType construido
        
        Examples:
            >>> variables = ['X', 'Y']
            >>> domains = {'X': {1, 2, 3}, 'Y': {1, 2, 3}}
            >>> constraints = [
            ...     {'variables': ['X', 'Y'], 'predicate': lambda x, y: x < y}
            ... ]
            >>> csp_type = CubicalCSPType.from_csp_problem(variables, domains, constraints)
        """
        # Construir tipos de dominios
        domain_types = {}
        for var in variables:
            domain = domains.get(var, set())
            domain_types[var] = FiniteType(f"Domain_{var}", frozenset(domain))
        
        # Construir proposiciones de restricciones
        constraint_props = []
        for i, constraint in enumerate(constraints):
            constraint_vars = constraint.get('variables', [])
            predicate = constraint.get('predicate')
            name = constraint.get('name', f"C{i}")
            
            if predicate is None:
                logger.warning(f"Restricción {name} sin predicado, ignorada")
                continue
            
            prop = PropositionType(name, tuple(constraint_vars), predicate)
            constraint_props.append(prop)
        
        return cls(
            variables=variables,
            domain_types=domain_types,
            constraint_props=constraint_props
        )
    
    def check_term(self, term: Term) -> bool:
        """
        Verifica si un término habita este tipo.
        
        Args:
            term: Término a verificar
        
        Returns:
            True si term : solution_type
        
        Notes:
            - Implementación básica, puede ser extendida con type checker completo
        """
        # Por ahora, verificación simple
        # TODO: Integrar con CubicalEngine para type checking completo
        return True
    
    def synthesize_term(self, solution: Dict[str, Any]) -> Term:
        """
        Sintetiza un término cúbico desde una solución CSP.
        
        Args:
            solution: Solución del CSP (mapa variable → valor)
        
        Returns:
            Término cúbico habitando solution_type
        
        Examples:
            >>> solution = {'X': 1, 'Y': 2}
            >>> term = csp_type.synthesize_term(solution)
        """
        # Construir término anidado de pares
        # Para Σ (x : D) Σ (y : D') (C), el término es (x, (y, proof_C))
        
        # Verificar que la solución satisface las restricciones
        if not self.verify_solution(solution):
            raise ValueError(f"Solución {solution} no satisface las restricciones")
        
        # Construir prueba de restricciones (término trivial)
        constraints_term = Unit()  # Prueba trivial
        
        # Construir pares anidados desde la última variable
        current_term = constraints_term
        
        for var in reversed(self.variables):
            value = solution.get(var)
            if value is None:
                raise ValueError(f"Variable {var} no tiene valor en la solución")
            
            # Crear término constante para el valor
            value_term = Var(str(value))  # Representación simple
            
            # Crear par (valor, término_anterior)
            current_term = Pair(value_term, current_term)
        
        return current_term
    
    def verify_solution(self, solution: Dict[str, Any]) -> bool:
        """
        Verifica si una solución satisface todas las restricciones.
        
        Args:
            solution: Solución del CSP
        
        Returns:
            True si satisface todas las restricciones
        """
        for prop in self.constraint_props:
            # Extraer valores de las variables involucradas
            values = []
            for var in prop.variables:
                value = solution.get(var)
                if value is None:
                    logger.warning(f"Variable {var} no tiene valor en la solución")
                    return False
                values.append(value)
            
            # Verificar restricción
            if not prop.check(*values):
                logger.debug(
                    f"Restricción {prop.constraint_name} no satisfecha "
                    f"para valores {values}"
                )
                return False
        
        return True
    
    def get_domain_size(self) -> int:
        """
        Calcula el tamaño del espacio de búsqueda (producto de dominios).
        
        Returns:
            Número total de asignaciones posibles
        """
        size = 1
        for domain_type in self.domain_types.values():
            size *= len(domain_type.values)
        return size
    
    def get_constraint_count(self) -> int:
        """
        Retorna el número de restricciones.
        
        Returns:
            Número de restricciones
        """
        return len(self.constraint_props)
    
    def __str__(self) -> str:
        """Representación en string del tipo."""
        return str(self.solution_type)
    
    def __repr__(self) -> str:
        """Representación detallada."""
        return (
            f"CubicalCSPType("
            f"vars={self.variables}, "
            f"domains={len(self.domain_types)}, "
            f"constraints={len(self.constraint_props)})"
        )


# ============================================================================
# Funciones de Utilidad
# ============================================================================

def create_finite_type(name: str, values: Set[Any]) -> FiniteType:
    """
    Crea un tipo finito.
    
    Args:
        name: Nombre del tipo
        values: Conjunto de valores
    
    Returns:
        FiniteType creado
    """
    return FiniteType(name, frozenset(values))


def create_proposition(
    name: str,
    variables: List[str],
    predicate: Callable
) -> PropositionType:
    """
    Crea una proposición.
    
    Args:
        name: Nombre de la restricción
        variables: Variables involucradas
        predicate: Predicado que verifica la restricción
    
    Returns:
        PropositionType creada
    """
    return PropositionType(name, tuple(variables), predicate)


# ============================================================================
# Ejemplo de Uso
# ============================================================================

def example_usage():
    """
    Ejemplo de uso de CubicalCSPType.
    """
    logger.info("=== Ejemplo de CubicalCSPType ===")
    
    # CSP simple: X < Y, dominios {1, 2, 3}
    variables = ['X', 'Y']
    domains = {
        'X': {1, 2, 3},
        'Y': {1, 2, 3}
    }
    constraints = [
        {
            'variables': ['X', 'Y'],
            'predicate': lambda x, y: x < y,
            'name': 'X_lt_Y'
        }
    ]
    
    # Crear tipo cúbico
    csp_type = CubicalCSPType.from_csp_problem(variables, domains, constraints)
    
    logger.info(f"Tipo CSP: {csp_type}")
    logger.info(f"Tipo solución: {csp_type.solution_type}")
    logger.info(f"Tamaño del espacio: {csp_type.get_domain_size()}")
    logger.info(f"Número de restricciones: {csp_type.get_constraint_count()}")
    
    # Verificar solución válida
    solution1 = {'X': 1, 'Y': 2}
    is_valid = csp_type.verify_solution(solution1)
    logger.info(f"Solución {solution1} válida: {is_valid}")
    
    # Verificar solución inválida
    solution2 = {'X': 2, 'Y': 1}
    is_valid = csp_type.verify_solution(solution2)
    logger.info(f"Solución {solution2} válida: {is_valid}")
    
    # Sintetizar término
    if csp_type.verify_solution(solution1):
        term = csp_type.synthesize_term(solution1)
        logger.info(f"Término sintetizado: {term}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    example_usage()

