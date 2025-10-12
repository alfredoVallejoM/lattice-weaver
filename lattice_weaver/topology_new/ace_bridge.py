"""
ACE-Locale Bridge: Integración entre Adaptive Consistency Engine y Locales

Este módulo proporciona la integración entre el motor de consistencia adaptativa (ACE)
y la teoría de Locales, permitiendo razonamiento topológico sobre problemas CSP.

Funcionalidades:
- Conversión de problemas CSP a Locales
- Interpretación topológica de consistencia
- Operadores modales sobre espacios de soluciones
- Análisis de conectividad de regiones factibles

Autor: LatticeWeaver Team (Track B)
Fecha: 12 de Octubre de 2025
Versión: 1.0
"""

from typing import Set, Dict, Optional, List, Tuple, Any
from dataclasses import dataclass
import logging

from .locale import Frame, Locale, FrameBuilder, LocaleBuilder, Hashable
from .operations import ModalOperators, ConnectivityAnalyzer

logger = logging.getLogger(__name__)


# ============================================================================
# CSPLocale - Locale derivado de un problema CSP
# ============================================================================

@dataclass
class CSPLocale:
    """
    Locale derivado de un problema CSP.
    
    Interpreta un problema de satisfacción de restricciones (CSP) como un
    espacio topológico donde:
    - Los abiertos representan regiones del espacio de soluciones
    - La consistencia corresponde a densidad topológica
    - Los operadores modales capturan propagación de restricciones
    
    Attributes:
        locale: Locale subyacente
        variables: Variables del CSP
        domains: Dominios de las variables
        constraints: Restricciones del CSP
        _solution_space: Espacio de soluciones (caché)
    
    Examples:
        >>> # CSP: X < Y, dominios {1, 2, 3}
        >>> csp_locale = CSPLocale.from_csp(variables, domains, constraints)
        >>> 
        >>> # Analizar conectividad del espacio de soluciones
        >>> analyzer = ConnectivityAnalyzer(csp_locale.locale)
        >>> is_connected = analyzer.is_connected()
    
    Notes:
        - La construcción del Locale puede ser costosa para CSPs grandes
        - Se recomienda usar para análisis cualitativo, no resolución directa
    """
    
    locale: Locale
    variables: List[str]
    domains: Dict[str, Set[Any]]
    constraints: List[Any]  # Restricciones del CSP original
    _solution_space: Optional[Set[Tuple]] = None
    
    @classmethod
    def from_csp(
        cls,
        variables: List[str],
        domains: Dict[str, Set[Any]],
        constraints: List[Any]
    ) -> 'CSPLocale':
        """
        Construye un CSPLocale desde un problema CSP.
        
        Args:
            variables: Lista de variables
            domains: Dominios de las variables
            constraints: Restricciones del CSP
        
        Returns:
            CSPLocale construido
        
        Notes:
            - Para CSPs pequeños (< 10 variables, dominios < 5)
            - Construcción explícita del espacio de soluciones
        """
        # Construir espacio de soluciones
        solution_space = cls._build_solution_space(variables, domains, constraints)
        
        # Construir Locale sobre el espacio de soluciones
        # Usamos powerset (espacio discreto) como primera aproximación
        frame = FrameBuilder.from_powerset(solution_space)
        locale = LocaleBuilder.from_frame(
            frame,
            name=f"CSP({','.join(variables)})"
        )
        
        return cls(
            locale=locale,
            variables=variables,
            domains=domains,
            constraints=constraints,
            _solution_space=solution_space
        )
    
    @staticmethod
    def _build_solution_space(
        variables: List[str],
        domains: Dict[str, Set[Any]],
        constraints: List[Any]
    ) -> Set[Tuple]:
        """
        Construye el espacio de soluciones del CSP.
        
        Args:
            variables: Lista de variables
            domains: Dominios de las variables
            constraints: Restricciones
        
        Returns:
            Conjunto de asignaciones que satisfacen todas las restricciones
        
        Notes:
            - Enumeración exhaustiva (exponencial)
            - Solo para CSPs pequeños
        """
        import itertools
        
        # Generar todas las asignaciones posibles
        domain_lists = [list(domains[var]) for var in variables]
        all_assignments = itertools.product(*domain_lists)
        
        # Filtrar por restricciones
        solutions = set()
        
        for assignment_tuple in all_assignments:
            assignment = dict(zip(variables, assignment_tuple))
            
            # Verificar todas las restricciones
            satisfies_all = True
            for constraint in constraints:
                if not CSPLocale._check_constraint(constraint, assignment):
                    satisfies_all = False
                    break
            
            if satisfies_all:
                solutions.add(assignment_tuple)
        
        logger.info(
            f"Espacio de soluciones construido: "
            f"{len(solutions)} soluciones de {len(list(itertools.product(*domain_lists)))} asignaciones"
        )
        
        return frozenset(solutions)
    
    @staticmethod
    def _check_constraint(constraint: Any, assignment: Dict[str, Any]) -> bool:
        """
        Verifica si una asignación satisface una restricción.
        
        Args:
            constraint: Restricción (puede ser función, tupla, etc.)
            assignment: Asignación de variables
        
        Returns:
            True si satisface, False en caso contrario
        
        Notes:
            - Implementación simplificada
            - Asume constraints son callables
        """
        try:
            if callable(constraint):
                return constraint(assignment)
            else:
                # Formato (var1, op, var2) o similar
                var1, op, var2 = constraint
                val1 = assignment.get(var1)
                val2 = assignment.get(var2)
                
                if val1 is None or val2 is None:
                    return True  # Variables no asignadas
                
                if op == '<':
                    return val1 < val2
                elif op == '<=':
                    return val1 <= val2
                elif op == '>':
                    return val1 > val2
                elif op == '>=':
                    return val1 >= val2
                elif op == '==':
                    return val1 == val2
                elif op == '!=':
                    return val1 != val2
                else:
                    logger.warning(f"Operador desconocido: {op}")
                    return True
        except Exception as e:
            logger.error(f"Error al verificar restricción: {e}")
            return False
    
    def get_solution_space(self) -> Set[Tuple]:
        """
        Retorna el espacio de soluciones.
        
        Returns:
            Conjunto de soluciones
        """
        return self._solution_space
    
    def is_consistent(self) -> bool:
        """
        Verifica si el CSP es consistente (tiene soluciones).
        
        Returns:
            True si tiene soluciones, False en caso contrario
        """
        return len(self._solution_space) > 0
    
    def analyze_topology(self) -> Dict[str, Any]:
        """
        Analiza propiedades topológicas del espacio de soluciones.
        
        Returns:
            Diccionario con análisis topológico
        """
        modal = ModalOperators(self.locale)
        connectivity = ConnectivityAnalyzer(self.locale)
        
        analysis = {
            'num_solutions': len(self._solution_space),
            'is_consistent': self.is_consistent(),
            'is_connected': connectivity.is_connected(),
            'is_compact': connectivity.is_compact(),
            's4_axioms_valid': modal.verify_s4_axioms(),
            'num_components': len(connectivity.connected_components())
        }
        
        return analysis


# ============================================================================
# ACELocaleBridge - Puente entre ACE y Locales
# ============================================================================

class ACELocaleBridge:
    """
    Puente entre Adaptive Consistency Engine y Locales.
    
    Proporciona utilidades para:
    - Convertir estados de ACE a Locales
    - Interpretar operaciones de ACE en términos topológicos
    - Analizar la estructura topológica del espacio de búsqueda
    
    Attributes:
        ace_engine: Instancia del motor ACE (opcional)
    
    Examples:
        >>> bridge = ACELocaleBridge()
        >>> 
        >>> # Convertir problema CSP a Locale
        >>> csp_locale = bridge.csp_to_locale(variables, domains, constraints)
        >>> 
        >>> # Analizar topología
        >>> analysis = csp_locale.analyze_topology()
    """
    
    def __init__(self, ace_engine: Optional[Any] = None):
        """
        Inicializa el bridge.
        
        Args:
            ace_engine: Instancia del motor ACE (opcional)
        """
        self.ace_engine = ace_engine
        logger.debug("ACELocaleBridge inicializado")
    
    def csp_to_locale(
        self,
        variables: List[str],
        domains: Dict[str, Set[Any]],
        constraints: List[Any]
    ) -> CSPLocale:
        """
        Convierte un problema CSP a un Locale.
        
        Args:
            variables: Variables del CSP
            domains: Dominios de las variables
            constraints: Restricciones del CSP
        
        Returns:
            CSPLocale construido
        """
        return CSPLocale.from_csp(variables, domains, constraints)
    
    def analyze_consistency_topology(
        self,
        csp_locale: CSPLocale
    ) -> Dict[str, Any]:
        """
        Analiza la topología de la consistencia.
        
        Interpreta la consistencia del CSP en términos topológicos:
        - Conectividad → espacio de soluciones conexo
        - Densidad → distribución de soluciones
        - Componentes → clusters de soluciones
        
        Args:
            csp_locale: Locale del CSP
        
        Returns:
            Análisis topológico de la consistencia
        """
        return csp_locale.analyze_topology()
    
    def modal_propagation(
        self,
        csp_locale: CSPLocale,
        region: Hashable
    ) -> Dict[str, Hashable]:
        """
        Propaga una región usando operadores modales.
        
        Interpreta la propagación de restricciones como operadores modales:
        - ◇ (interior): región "definitivamente factible"
        - □ (clausura): región "posiblemente factible"
        
        Args:
            csp_locale: Locale del CSP
            region: Región del espacio de soluciones
        
        Returns:
            Diccionario con interior y clausura
        """
        modal = ModalOperators(csp_locale.locale)
        
        return {
            'interior': modal.diamond(region),
            'closure': modal.box(region),
            'boundary': csp_locale.locale.boundary(region)
        }


# ============================================================================
# Utilidades de Integración
# ============================================================================

def create_simple_csp_locale(
    num_variables: int,
    domain_size: int,
    constraint_type: str = 'alldiff'
) -> CSPLocale:
    """
    Crea un CSPLocale simple para pruebas.
    
    Args:
        num_variables: Número de variables
        domain_size: Tamaño del dominio
        constraint_type: Tipo de restricción ('alldiff', 'ordered', 'none')
    
    Returns:
        CSPLocale simple
    
    Examples:
        >>> # CSP con 3 variables, dominio {1,2,3}, todas diferentes
        >>> locale = create_simple_csp_locale(3, 3, 'alldiff')
    """
    variables = [f'X{i}' for i in range(num_variables)]
    domain = set(range(domain_size))
    domains = {var: domain for var in variables}
    
    # Construir restricciones
    constraints = []
    
    if constraint_type == 'alldiff':
        # Todas las variables diferentes
        for i in range(num_variables):
            for j in range(i + 1, num_variables):
                constraints.append((variables[i], '!=', variables[j]))
    
    elif constraint_type == 'ordered':
        # Variables en orden creciente
        for i in range(num_variables - 1):
            constraints.append((variables[i], '<', variables[i + 1]))
    
    elif constraint_type == 'none':
        # Sin restricciones
        pass
    
    else:
        raise ValueError(f"Tipo de restricción desconocido: {constraint_type}")
    
    return CSPLocale.from_csp(variables, domains, constraints)


# ============================================================================
# Ejemplo de Uso
# ============================================================================

def example_usage():
    """
    Ejemplo de uso del ACE-Locale Bridge.
    """
    logger.info("=== Ejemplo de ACE-Locale Bridge ===")
    
    # Crear CSP simple: 3 variables, dominio {0,1,2}, todas diferentes
    csp_locale = create_simple_csp_locale(3, 3, 'alldiff')
    
    logger.info(f"CSP Locale creado: {csp_locale.locale}")
    logger.info(f"Número de soluciones: {len(csp_locale.get_solution_space())}")
    
    # Analizar topología
    analysis = csp_locale.analyze_topology()
    logger.info(f"Análisis topológico: {analysis}")
    
    # Usar bridge
    bridge = ACELocaleBridge()
    
    # Propagar una región
    if csp_locale.is_consistent():
        # Tomar una solución como región
        solution = next(iter(csp_locale.get_solution_space()))
        region = frozenset({solution})
        
        propagation = bridge.modal_propagation(csp_locale, region)
        logger.info(f"Propagación modal: {propagation}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    example_usage()

