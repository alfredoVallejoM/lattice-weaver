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
        super().__post_init__() # Llama al __post_init__ de CubicalFiniteType para validar size
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
        predicate: Predicado que verifica la restricción
    
    Examples:
        >>> # Restricción X < Y
        >>> prop = PropositionType("X_lt_Y", ["X", "Y"], lambda x, y: x < y)
    """
    constraint_name: str
    variables: tuple  # Tupla de nombres de variables
    predicate: Callable  # Función que verifica la restricción

    def _compute_hash(self) -> int:
        return hash((self.constraint_name, self.variables))

    def _compute_string(self) -> str:
        vars_str = ", ".join(self.variables)
        return f"({self.constraint_name}({vars_str}))"

    def __post_init__(self):
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
            domain_types[var] = FiniteType(name=f"Domain_{var}", values=frozenset(domain), size=len(domain))
        
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
        # Implementación de check_term (asumiendo que solution_type es un SigmaType)
        current_type = self.solution_type
        current_term = term
        
        # Desempaquetar el término y verificar cada componente
        for var_name in self.variables:
            if not isinstance(current_type, SigmaType):
                logger.error(f"Tipo esperado SigmaType, encontrado {type(current_type)}")
                return False
            
            if not isinstance(current_term, Pair):
                logger.error(f"Término esperado Pair, encontrado {type(current_term)}")
                return False
            
            # Verificar el tipo del valor actual
            value = current_term.fst
            domain_type = self.domain_types[var_name]
            if value not in domain_type.values:
                logger.error(f"Valor {value} no en el dominio de {var_name}")
                return False
            
            current_type = current_type.snd_type.substitute(var_name, ValueTerm(value))
            current_term = current_term.snd
        
        # Verificar las proposiciones (restricciones)
        if isinstance(current_type, UnitType):
            # Si no hay restricciones, o todas se han reducido a UnitType
            return True
        elif isinstance(current_type, PropositionType):
            # Si queda una proposición, verificarla
            # Esto asume que PropositionType tiene un método check(values)
            # y que 'current_term' es Unit() si la proposición es satisfecha
            return isinstance(current_term, Unit) and current_type.check(*[term.value for term in term.get_values_in_order()])
        elif isinstance(current_type, SigmaType) and current_type.fst_type == UnitType():
            # Esto es un producto de proposiciones, donde cada una se reduce a UnitType
            # Si el término final es una serie de Unit(), entonces es válido
            return self._check_nested_unit_terms(current_term)
        else:
            logger.error(f"Tipo final inesperado: {type(current_type)}")
            return False

    def _check_nested_unit_terms(self, term: Term) -> bool:
        """
        Verifica si un término es una serie anidada de Unit().
        """
        if isinstance(term, Unit):
            return True
        elif isinstance(term, Pair):
            return self._check_nested_unit_terms(term.fst) and self._check_nested_unit_terms(term.snd)
        return False

    def synthesize_term(self, solution: Dict[str, Any]) -> Term:
        """
        Sintetiza un término cúbico a partir de una solución CSP.
        
        Args:
            solution: Solución del CSP (mapa variable → valor)
            
        Returns:
            Término cúbico habitando el tipo de soluciones
        """
        if not self.variables:
            return Unit()
        
        # Construir el término desde la última variable hacia la primera
        # Las proposiciones se asumen satisfechas, por lo que se representan con Unit()
        current_term = Unit()
        
        # Invertir el orden de las variables para construir el término correctamente
        # en el orden de anidamiento del SigmaType
        for var in reversed(self.variables):
            value = solution.get(var)
            if value is None:
                raise ValueError(f"Solución incompleta: falta valor para {var}")
            current_term = Pair(ValueTerm(value), current_term)
        
        # El término final debe ser un Pair anidado que termina en Unit()
        # para representar la satisfacción de todas las proposiciones
        return current_term

    def verify_solution(self, solution: Dict[str, Any]) -> bool:
        """
        Verifica si una solución CSP satisface todas las restricciones.
        
        Args:
            solution: Solución a verificar
            
        Returns:
            True si la solución es válida, False en caso contrario
        """
        for prop in self.constraint_props:
            # Obtener los valores de las variables de la restricción en el orden correcto
            values_for_prop = [solution.get(var) for var in prop.variables]
            
            # Si algún valor falta, la solución es inválida para esta restricción
            if any(v is None for v in values_for_prop):
                logger.warning(f"Solución incompleta para restricción {prop.constraint_name}")
                return False
            
            if not prop.check(*values_for_prop):
                logger.debug(f"Restricción {prop.constraint_name} no satisfecha por {solution}")
                return False
        return True

    def get_domain_size(self) -> int:
        """
        Calcula el tamaño total del espacio de búsqueda (producto de los tamaños de dominio).
        """
        size = 1
        for dt in self.domain_types.values():
            size *= dt.size
        return size

    def get_constraint_count(self) -> int:
        """
        Retorna el número de restricciones.
        """
        return len(self.constraint_props)

    def __str__(self) -> str:
        if self.solution_type:
            return str(self.solution_type)
        return "CubicalCSPType(uninitialized)"

    def __repr__(self) -> str:
        return (
            f"CubicalCSPType("
            f"variables={len(self.variables)}, "
            f"domains={len(self.domain_types)}, "
            f"constraints={len(self.constraint_props)})"
        )

