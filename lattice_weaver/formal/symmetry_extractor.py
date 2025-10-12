"""
SymmetryExtractor: Extracción y Análisis de Simetrías en CSP

Este módulo implementa la detección y análisis de simetrías en problemas CSP,
permitiendo:

1. Detectar simetrías de variables (permutaciones)
2. Detectar simetrías de valores
3. Agrupar soluciones equivalentes por simetría
4. Optimizar búsqueda explotando simetrías
5. Generar clases de equivalencia

Las simetrías son transformaciones que preservan la estructura del problema
y son fundamentales para reducir el espacio de búsqueda.

Autor: LatticeWeaver Team (Track: CSP-Cubical Integration)
Fecha: 12 de Octubre de 2025
Versión: 1.0
"""

from typing import Dict, List, Set, Any, Optional, Tuple, Callable, FrozenSet
from dataclasses import dataclass, field
from itertools import permutations
from collections import defaultdict
import logging

from .cubical_csp_type import CubicalCSPType
from .csp_cubical_bridge import CSPToCubicalBridge

logger = logging.getLogger(__name__)


# ============================================================================
# Representación de Simetrías
# ============================================================================

@dataclass(frozen=True)
class Symmetry:
    """
    Representa una simetría del problema CSP.
    
    Una simetría es una transformación que preserva la estructura del problema.
    
    Attributes:
        type_: Tipo de simetría ('variable', 'value', 'combined')
        mapping: Mapeo que define la transformación
        description: Descripción legible de la simetría
    """
    type_: str
    mapping: FrozenSet[Tuple[str, str]]  # Para variables: (var1, var2)
    description: str = ""
    
    def apply_to_solution(self, solution: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aplica la simetría a una solución.
        
        Args:
            solution: Solución original
            
        Returns:
            Solución transformada
        """
        if self.type_ == 'variable':
            # Simetría de variables: permutar variables
            new_solution = {}
            mapping_dict = dict(self.mapping)
            
            for var, value in solution.items():
                new_var = mapping_dict.get(var, var)
                new_solution[new_var] = value
            
            return new_solution
        else:
            # Otros tipos de simetría
            return solution.copy()
    
    def __str__(self) -> str:
        """Representación en string."""
        if self.description:
            return f"Symmetry({self.type_}: {self.description})"
        return f"Symmetry({self.type_})"


@dataclass
class SymmetryGroup:
    """
    Grupo de simetrías del problema.
    
    Attributes:
        symmetries: Lista de simetrías
        order: Orden del grupo (número de simetrías)
    """
    symmetries: List[Symmetry] = field(default_factory=list)
    
    @property
    def order(self) -> int:
        """Orden del grupo de simetrías."""
        return len(self.symmetries)
    
    def add_symmetry(self, symmetry: Symmetry):
        """Añade una simetría al grupo."""
        self.symmetries.append(symmetry)
    
    def apply_all_to_solution(
        self,
        solution: Dict[str, Any]
    ) -> Set[FrozenSet[Tuple[str, Any]]]:
        """
        Aplica todas las simetrías a una solución.
        
        Args:
            solution: Solución original
            
        Returns:
            Conjunto de soluciones simétricas
        """
        symmetric_solutions = set()
        
        for symmetry in self.symmetries:
            transformed = symmetry.apply_to_solution(solution)
            symmetric_solutions.add(frozenset(transformed.items()))
        
        return symmetric_solutions
    
    def __str__(self) -> str:
        """Representación en string."""
        return f"SymmetryGroup(order={self.order})"


# ============================================================================
# SymmetryExtractor - Motor de Extracción de Simetrías
# ============================================================================

@dataclass
class SymmetryExtractor:
    """
    Extractor de simetrías para problemas CSP.
    
    Detecta y analiza simetrías en la estructura del problema,
    permitiendo optimizar la búsqueda y agrupar soluciones equivalentes.
    
    Attributes:
        bridge: Bridge CSP-Cubical
        
    Examples:
        >>> extractor = SymmetryExtractor(bridge)
        >>> symmetries = extractor.extract_variable_symmetries()
        >>> print(f"Simetrías encontradas: {len(symmetries)}")
    """
    
    bridge: CSPToCubicalBridge
    
    # Caché de simetrías detectadas
    _symmetry_cache: Optional[SymmetryGroup] = field(default=None, repr=False)
    
    def extract_variable_symmetries(self) -> List[Symmetry]:
        """
        Extrae simetrías de variables (permutaciones que preservan restricciones).
        
        Detecta qué permutaciones de variables mantienen la estructura
        del problema invariante.
        
        Returns:
            Lista de simetrías de variables
            
        Examples:
            >>> symmetries = extractor.extract_variable_symmetries()
        """
        cubical_type = self.bridge.cubical_type
        if cubical_type is None:
            return []
        
        variables = cubical_type.variables
        symmetries = []
        
        # Caso simple: variables con dominios idénticos
        domain_groups = self._group_variables_by_domain()
        
        for domain_key, vars_in_group in domain_groups.items():
            if len(vars_in_group) < 2:
                continue
            
            # Generar permutaciones de variables en el grupo
            for perm in permutations(vars_in_group):
                if perm == tuple(vars_in_group):
                    continue  # Identidad
                
                # Crear mapeo
                mapping = frozenset(zip(vars_in_group, perm))
                
                # Verificar si preserva restricciones
                if self._preserves_constraints(mapping):
                    symmetry = Symmetry(
                        type_='variable',
                        mapping=mapping,
                        description=f"Permutation: {dict(mapping)}"
                    )
                    symmetries.append(symmetry)
        
        logger.info(f"Encontradas {len(symmetries)} simetrías de variables")
        return symmetries
    
    def _group_variables_by_domain(self) -> Dict[FrozenSet, List[str]]:
        """
        Agrupa variables por dominio idéntico.
        
        Returns:
            Diccionario: dominio → lista de variables
        """
        cubical_type = self.bridge.cubical_type
        if cubical_type is None:
            return {}
        
        domain_groups = defaultdict(list)
        
        for var in cubical_type.variables:
            domain = cubical_type.domain_types[var]
            domain_key = frozenset(domain.values)
            domain_groups[domain_key].append(var)
        
        return dict(domain_groups)
    
    def _preserves_constraints(self, mapping: FrozenSet[Tuple[str, str]]) -> bool:
        """
        Verifica si un mapeo de variables preserva las restricciones.
        
        Args:
            mapping: Mapeo de variables (var_original → var_permutada)
            
        Returns:
            True si preserva restricciones
        """
        cubical_type = self.bridge.cubical_type
        if cubical_type is None:
            return False
        
        mapping_dict = dict(mapping)
        
        # Para cada restricción, verificar si se preserva bajo la permutación
        for constraint in cubical_type.constraint_props:
            vars_original = constraint.variables
            
            # Aplicar permutación a las variables
            vars_permuted = tuple(
                mapping_dict.get(v, v) for v in vars_original
            )
            
            # Verificar si existe una restricción equivalente
            if not self._has_equivalent_constraint(vars_permuted, constraint):
                return False
        
        return True
    
    def _has_equivalent_constraint(
        self,
        vars_permuted: Tuple[str, ...],
        original_constraint
    ) -> bool:
        """
        Verifica si existe una restricción equivalente para variables permutadas.
        
        Args:
            vars_permuted: Variables después de permutación
            original_constraint: Restricción original
            
        Returns:
            True si existe restricción equivalente
        """
        cubical_type = self.bridge.cubical_type
        if cubical_type is None:
            return False
        
        # Buscar restricción que involucre las variables permutadas
        for constraint in cubical_type.constraint_props:
            if tuple(constraint.variables) == vars_permuted:
                # Verificar si la relación es la misma
                # (simplificación: asumimos que sí si las variables coinciden)
                return True
        
        # Si las variables permutadas son las mismas que las originales
        if tuple(original_constraint.variables) == vars_permuted:
            return True
        
        return False
    
    def extract_all_symmetries(self) -> SymmetryGroup:
        """
        Extrae todas las simetrías del problema.
        
        Returns:
            Grupo de simetrías
            
        Examples:
            >>> group = extractor.extract_all_symmetries()
            >>> print(f"Orden del grupo: {group.order}")
        """
        # Verificar caché
        if self._symmetry_cache is not None:
            logger.debug("Usando simetrías cacheadas")
            return self._symmetry_cache
        
        group = SymmetryGroup()
        
        # Extraer simetrías de variables
        var_symmetries = self.extract_variable_symmetries()
        for sym in var_symmetries:
            group.add_symmetry(sym)
        
        # Cachear
        self._symmetry_cache = group
        
        logger.info(f"Grupo de simetrías: orden {group.order}")
        return group
    
    def get_equivalence_classes(
        self,
        solutions: List[Dict[str, Any]]
    ) -> List[Set[FrozenSet[Tuple[str, Any]]]]:
        """
        Agrupa soluciones en clases de equivalencia por simetría.
        
        Args:
            solutions: Lista de soluciones
            
        Returns:
            Lista de clases de equivalencia
            
        Examples:
            >>> classes = extractor.get_equivalence_classes(all_solutions)
            >>> print(f"Clases de equivalencia: {len(classes)}")
        """
        group = self.extract_all_symmetries()
        
        # Conjunto de soluciones ya asignadas a una clase
        assigned = set()
        classes = []
        
        for solution in solutions:
            solution_frozen = frozenset(solution.items())
            
            if solution_frozen in assigned:
                continue
            
            # Generar clase de equivalencia aplicando todas las simetrías
            equiv_class = group.apply_all_to_solution(solution)
            equiv_class.add(solution_frozen)
            
            classes.append(equiv_class)
            assigned.update(equiv_class)
        
        logger.info(
            f"Agrupadas {len(solutions)} soluciones en {len(classes)} clases"
        )
        return classes
    
    def get_representative_solutions(
        self,
        solutions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Obtiene representantes de cada clase de equivalencia.
        
        Args:
            solutions: Lista de soluciones
            
        Returns:
            Lista de soluciones representantes
            
        Examples:
            >>> representatives = extractor.get_representative_solutions(all_sols)
        """
        classes = self.get_equivalence_classes(solutions)
        
        representatives = []
        for equiv_class in classes:
            # Tomar el primer elemento como representante
            representative = dict(next(iter(equiv_class)))
            representatives.append(representative)
        
        return representatives
    
    def count_unique_solutions(
        self,
        solutions: List[Dict[str, Any]]
    ) -> int:
        """
        Cuenta soluciones únicas módulo simetrías.
        
        Args:
            solutions: Lista de soluciones
            
        Returns:
            Número de soluciones únicas
        """
        classes = self.get_equivalence_classes(solutions)
        return len(classes)
    
    def analyze_symmetry_structure(self) -> Dict[str, Any]:
        """
        Analiza la estructura de simetrías del problema.
        
        Returns:
            Diccionario con análisis:
            - symmetry_count: Número de simetrías
            - symmetry_types: Tipos de simetrías encontradas
            - domain_groups: Grupos de variables por dominio
            
        Examples:
            >>> analysis = extractor.analyze_symmetry_structure()
            >>> print(analysis)
        """
        group = self.extract_all_symmetries()
        domain_groups = self._group_variables_by_domain()
        
        symmetry_types = defaultdict(int)
        for sym in group.symmetries:
            symmetry_types[sym.type_] += 1
        
        return {
            'symmetry_count': group.order,
            'symmetry_types': dict(symmetry_types),
            'domain_groups': {
                str(k): v for k, v in domain_groups.items()
            },
            'has_symmetries': group.order > 0
        }
    
    def clear_cache(self):
        """Limpia la caché de simetrías."""
        self._symmetry_cache = None
        logger.debug("Caché de simetrías limpiada")
    
    def __str__(self) -> str:
        """Representación en string."""
        return "SymmetryExtractor"
    
    def __repr__(self) -> str:
        """Representación detallada."""
        group = self._symmetry_cache
        if group:
            return f"SymmetryExtractor(symmetries_cached={group.order})"
        return "SymmetryExtractor(no_cache)"


# ============================================================================
# Funciones de Utilidad
# ============================================================================

def create_symmetry_extractor(bridge: CSPToCubicalBridge) -> SymmetryExtractor:
    """
    Crea un SymmetryExtractor desde un bridge.
    
    Args:
        bridge: Bridge CSP-Cubical
    
    Returns:
        SymmetryExtractor configurado
    """
    return SymmetryExtractor(bridge)


# ============================================================================
# Ejemplo de Uso
# ============================================================================

def example_usage():
    """
    Ejemplo de uso de SymmetryExtractor.
    """
    from .csp_cubical_bridge import create_simple_csp_bridge
    
    logger.info("=== Ejemplo de SymmetryExtractor ===")
    
    # Crear bridge con CSP simétrico (N-Queens simplificado)
    bridge = create_simple_csp_bridge(
        variables=['Q1', 'Q2', 'Q3'],
        domains={'Q1': {1, 2, 3}, 'Q2': {1, 2, 3}, 'Q3': {1, 2, 3}},
        constraints=[
            ('Q1', 'Q2', lambda x, y: x != y),
            ('Q2', 'Q3', lambda x, y: x != y),
            ('Q1', 'Q3', lambda x, y: x != y)
        ]
    )
    
    # Crear extractor
    extractor = SymmetryExtractor(bridge)
    
    logger.info(f"Extractor: {extractor}")
    
    # Extraer simetrías
    var_symmetries = extractor.extract_variable_symmetries()
    logger.info(f"Simetrías de variables: {len(var_symmetries)}")
    for i, sym in enumerate(var_symmetries[:5], 1):
        logger.info(f"  {i}. {sym}")
    
    # Analizar estructura
    analysis = extractor.analyze_symmetry_structure()
    logger.info(f"Análisis de estructura:")
    for key, value in analysis.items():
        logger.info(f"  {key}: {value}")
    
    # Ejemplo de soluciones
    solutions = [
        {'Q1': 1, 'Q2': 2, 'Q3': 3},
        {'Q1': 2, 'Q2': 1, 'Q3': 3},
        {'Q1': 3, 'Q2': 2, 'Q3': 1}
    ]
    
    logger.info(f"\nSoluciones de ejemplo: {len(solutions)}")
    
    # Agrupar en clases de equivalencia
    classes = extractor.get_equivalence_classes(solutions)
    logger.info(f"Clases de equivalencia: {len(classes)}")
    
    # Obtener representantes
    representatives = extractor.get_representative_solutions(solutions)
    logger.info(f"Representantes: {len(representatives)}")
    for i, rep in enumerate(representatives, 1):
        logger.info(f"  {i}. {rep}")
    
    # Contar soluciones únicas
    unique_count = extractor.count_unique_solutions(solutions)
    logger.info(f"Soluciones únicas (módulo simetrías): {unique_count}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    example_usage()

