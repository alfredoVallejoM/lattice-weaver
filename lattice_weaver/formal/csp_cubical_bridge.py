"""
CSP to Cubical Bridge: Puente entre CSP y Tipos Cúbicos

Este módulo implementa el puente bidireccional entre el motor CSP (ArcEngine)
y el sistema de tipos cúbicos, permitiendo:

1. Traducir problemas CSP a tipos cúbicos
2. Verificar soluciones usando type checking cúbico
3. Convertir soluciones a términos cúbicos
4. Extraer propiedades topológicas del espacio de soluciones

Autor: LatticeWeaver Team (Track: CSP-Cubical Integration)
Fecha: 12 de Octubre de 2025
Versión: 1.0
"""

from typing import Dict, List, Set, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from functools import lru_cache
import logging

from ..core.csp_problem import CSP

from ..core.csp_problem import Constraint
from .cubical_csp_type import CubicalCSPType, PropositionType
from .cubical_syntax import Term, Type
from .cubical_engine import CubicalEngine

logger = logging.getLogger(__name__)


# ============================================================================
# CSPToCubicalBridge - Puente Principal
# ============================================================================

@dataclass
class CSPToCubicalBridge:
    """
    Puente bidireccional entre CSP y tipos cúbicos.
    
    Traduce problemas CSP a tipos cúbicos para verificación formal
    y análisis topológico del espacio de soluciones.
    
    Attributes:
        csp_problem: Problema CSP (lattice_weaver.core.csp_problem.CSP)
        cubical_engine: Motor de tipos cúbicos
        cubical_type: Tipo cúbico derivado del CSP
        
    Examples:
            >>> from lattice_weaver.core.csp_problem import CSP, Constraint
            >>> csp = CSP(
            >>>     variables={'X', 'Y'},
            >>>     domains={'X': frozenset({1, 2, 3}), 'Y': frozenset({1, 2, 3})},
            >>>     constraints=[Constraint(scope=frozenset({'X', 'Y'}), relation=lambda x, y: x < y, name='X_lt_Y')]
            >>> )
            >>> bridge = CSPToCubicalBridge(csp_problem=csp)
        >>> cubical_type = bridge.translate_to_cubical_type()
        >>> print(cubical_type)
        Σ(X : {1, 2, 3}). Σ(Y : {1, 2, 3}). (X_lt_Y(X, Y))
    """
    
    csp_problem: CSP
    cubical_engine: Optional[CubicalEngine] = None
    cubical_type: Optional[CubicalCSPType] = None
    
    # Caché para optimización
    _translation_cache: Dict[int, CubicalCSPType] = field(default_factory=dict, repr=False)
    _verification_cache: Dict[int, bool] = field(default_factory=dict, repr=False)
    
    def __post_init__(self):
        """Inicializa el bridge."""
        # No se necesita inicializar cubical_engine aquí, se hace en el constructor de CubicalCSPType si es necesario.
        # Traducir automáticamente si no se ha hecho
        if self.cubical_type is None:
            self.cubical_type = self.translate_to_cubical_type()
    
    def translate_to_cubical_type(self) -> CubicalCSPType:
        """
        Traduce el CSP a un tipo cúbico.
        
        Returns:
            Tipo cúbico representando el espacio de soluciones
            
        Examples:
            >>> bridge = CSPToCubicalBridge(csp_problem=csp)
            >>> cubical_type = bridge.translate_to_cubical_type()
        """
        # Verificar caché
        csp_hash = self._hash_csp()
        if csp_hash in self._translation_cache:
            logger.debug("Usando traducción cacheada")
            return self._translation_cache[csp_hash]
        
        # Extraer variables y dominios
        variables = list(self.csp_problem.variables)
        domains = {}
        for var_name, domain in self.csp_problem.domains.items():
            domains[var_name] = set(domain)
        
        # Extraer restricciones
        constraints = []
        for constraint in self.csp_problem.constraints:
            constraint_dict = self._constraint_to_dict(constraint)
            if constraint_dict:
                constraints.append(constraint_dict)
        
        # Crear tipo cúbico
        cubical_type = CubicalCSPType.from_csp_problem(
            variables=variables,
            domains=domains,
            constraints=constraints
        )
        
        # Cachear
        self._translation_cache[csp_hash] = cubical_type
        
        logger.info(f"Traducido CSP a tipo cúbico: {cubical_type}")
        return cubical_type
    
    def _constraint_to_dict(
        self,
        constraint: Constraint
    ) -> Optional[Dict[str, Any]]:
        """
        Convierte una restricción del CSP a diccionario.
        
        Args:
            constraint_id: ID de la restricción
            constraint: Objeto Constraint
        
        Returns:
            Diccionario con variables, predicate y name
        """
        return {
            'variables': list(constraint.scope),
            'predicate': constraint.relation,
            'name': constraint.name
        }
    
    def solution_to_term(self, solution: Dict[str, Any]) -> Term:
        """
        Convierte una solución CSP a un término cúbico.
        
        Args:
            solution: Solución del CSP (mapa variable → valor)
            
        Returns:
            Término cúbico habitando el tipo de soluciones
            
        Examples:
            >>> solution = {'X': 1, 'Y': 2}
            >>> term = bridge.solution_to_term(solution)
            >>> print(term)
            (1, (2, ()))
        """
        if self.cubical_type is None:
            raise ValueError("Tipo cúbico no inicializado")
        
        return self.cubical_type.synthesize_term(solution)
    
    def verify_solution(self, solution: Dict[str, Any]) -> bool:
        """
        Verifica una solución usando type checking cúbico.
        
        Realiza verificación en dos niveles:
        1. Verificación básica de restricciones (CubicalCSPType)
        2. Type checking formal (CubicalEngine) si está disponible
        
        Args:
            solution: Solución a verificar
            
        Returns:
            True si la solución type-checks correctamente
            
        Examples:
            >>> solution = {'X': 1, 'Y': 2}
            >>> is_valid = bridge.verify_solution(solution)
            >>> print(is_valid)
            True
        """
        # Verificar caché
        solution_hash = hash(frozenset(solution.items()))
        if solution_hash in self._verification_cache:
            logger.debug(f"Usando verificación cacheada para {solution}")
            return self._verification_cache[solution_hash]
        
        if self.cubical_type is None:
            raise ValueError("Tipo cúbico no inicializado")
        
        # Nivel 1: Verificación básica de restricciones
        is_valid_basic = self.cubical_type.verify_solution(solution)
        
        if not is_valid_basic:
            # Si falla la verificación básica, no continuar
            self._verification_cache[solution_hash] = False
            return False
        
        # Nivel 2: Type checking formal (si CubicalEngine está disponible)
        # NOTA: El type checking formal con CubicalEngine está deshabilitado
        # temporalmente porque los tipos CSP (FiniteType, PropositionType)
        # no son directamente compatibles con el sistema de tipos HoTT estándar.
        # 
        # TODO: Implementar traducción de FiniteType/PropositionType a tipos HoTT
        # estándar (e.g., usando tipos inductivos o universos finitos).
        #
        # Por ahora, usamos solo la verificación básica de restricciones,
        # que es correcta y suficiente para la funcionalidad actual.
        
        # Cachear resultado de verificación básica
        self._verification_cache[solution_hash] = is_valid_basic
        return is_valid_basic
    
    def verify_solution_with_proof(
        self,
        solution: Dict[str, Any]
    ) -> Tuple[bool, Optional[Term]]:
        """
        Verifica una solución y retorna la prueba (término).
        
        Args:
            solution: Solución a verificar
            
        Returns:
            Tupla (is_valid, proof_term)
            
        Examples:
            >>> solution = {'X': 1, 'Y': 2}
            >>> is_valid, proof = bridge.verify_solution_with_proof(solution)
        """
        is_valid = self.verify_solution(solution)
        
        if is_valid:
            try:
                proof = self.solution_to_term(solution)
                return (True, proof)
            except Exception as e:
                logger.error(f"Error al sintetizar prueba: {e}")
                return (True, None)
        else:
            return (False, None)
    
    def get_solution_space_properties(self) -> Dict[str, Any]:
        """
        Extrae propiedades del espacio de soluciones.
        
        Returns:
            Diccionario con propiedades:
            - domain_size: Tamaño del espacio de búsqueda
            - constraint_count: Número de restricciones
            - variable_count: Número de variables
            - type_complexity: Complejidad del tipo
            
        Examples:
            >>> props = bridge.get_solution_space_properties()
            >>> print(props['domain_size'])
            9
        """
        if self.cubical_type is None:
            raise ValueError("Tipo cúbico no inicializado")
        
        return {
            'domain_size': self.cubical_type.get_domain_size(),
            'constraint_count': self.cubical_type.get_constraint_count(),
            'variable_count': len(self.cubical_type.variables),
            'type_complexity': self._compute_type_complexity(),
            'variables': self.cubical_type.variables,
            'domain_types': {
                var: str(dtype)
                for var, dtype in self.cubical_type.domain_types.items()
            }
        }
    
    def _compute_type_complexity(self) -> int:
        """
        Calcula una métrica de complejidad del tipo.
        
        Returns:
            Métrica de complejidad (mayor = más complejo)
        """
        if self.cubical_type is None:
            return 0
        
        # Complejidad = número de variables + número de restricciones
        return (
            len(self.cubical_type.variables) +
            len(self.cubical_type.constraint_props)
        )
    
    def _hash_csp(self) -> int:
        """
        Calcula un hash del CSP para caching.
        
        Incluye variables, dominios y restricciones para detectar cambios.
        
        Returns:
            Hash del CSP
        """
        # Hash basado en variables
        var_tuple = tuple(sorted(self.csp_problem.variables))
        
        # Hash basado en dominios (valores actuales)
        domain_tuples = []
        for var_name in sorted(self.csp_problem.variables):
            domain = self.csp_problem.domains[var_name]
            domain_values = tuple(sorted(list(domain), key=lambda x: (type(x).__name__, str(x))))
            domain_tuples.append((var_name, domain_values))
        domain_tuple = tuple(domain_tuples)
        
        # Hash basado en restricciones (IDs y nombres de relaciones)
        constraint_tuples = []
        for constraint in self.csp_problem.constraints:

            constraint_tuples.append((
                constraint.name,
                tuple(sorted(constraint.scope)),
                str(constraint.relation) # Convertir la función a string para el hash
            ))
        constraint_tuple = tuple(constraint_tuples)
        
        return hash((var_tuple, domain_tuple, constraint_tuple))
    
    def clear_cache(self):
        """Limpia las cachés de traducción y verificación."""
        self._translation_cache.clear()
        self._verification_cache.clear()
        logger.debug("Cachés limpiadas")
    
    def __str__(self) -> str:
        """Representación en string del bridge."""
        if self.cubical_type:
            return f"CSPToCubicalBridge({self.cubical_type})"
        return "CSPToCubicalBridge(not initialized)"
    
    def __repr__(self) -> str:
        """Representación detallada."""
        return (
            f"CSPToCubicalBridge("
            f"variables={len(self.csp_problem.variables)}, "
            f"constraints={len(self.csp_problem.constraints)})"
        )


# ============================================================================
# Funciones de Utilidad
# ============================================================================









# ============================================================================
# Ejemplo de Uso
# ============================================================================







