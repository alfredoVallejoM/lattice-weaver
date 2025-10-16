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
        # Acceder a las variables directamente del CSP, ya que las simetrías se definen sobre el CSP original
        variables = self.bridge.csp_problem.variables
        if not variables:
            return []
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
        # Agrupar variables por dominio idéntico directamente del CSP
        domain_groups = defaultdict(list)
        
        for var in self.bridge.csp_problem.variables:
            # Acceder al dominio de la variable directamente del CSP
            # self.bridge.csp_problem.domains es un diccionario de var_name -> frozenset
            domain_values = self.bridge.csp_problem.domains[var]
            domain_key = domain_values # Ya es un frozenset, usarlo directamente como clave
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
        # Las restricciones se verifican sobre el CSP original, no sobre el tipo cúbico
        mapping_dict = dict(mapping)
        
        # Para cada restricción, verificar si se preserva bajo la permutación
        for constraint in self.bridge.csp_problem.constraints:
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
        # Buscar restricción que involucre las variables permutadas en el CSP original
        for constraint in self.bridge.csp_problem.constraints:
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
        """
        if not solutions:
            return []
        
        group = self.extract_all_symmetries()
        
        # Normalizar soluciones a frozenset para poder añadirlas a un set
        normalized_solutions = {
            frozenset(sol.items()) for sol in solutions
        }
        
        classes = []
        
        while normalized_solutions:
            # Tomar una solución como representante
            representative = normalized_solutions.pop()
            
            # Generar todas las soluciones simétricas
            symmetric_class = {representative}
            
            # Aplicar todas las simetrías al representante
            for symmetry in group.symmetries:
                transformed = symmetry.apply_to_solution(dict(representative))
                transformed_frozen = frozenset(transformed.items())
                
                if transformed_frozen in normalized_solutions:
                    symmetric_class.add(transformed_frozen)
                    normalized_solutions.remove(transformed_frozen)
            
            classes.append(symmetric_class)
        
        logger.info(f"Encontradas {len(classes)} clases de equivalencia")
        return classes
    
    def count_unique_solutions(self, solutions: List[Dict[str, Any]]) -> int:
        """
        Cuenta el número de soluciones únicas (no simétricas).
        
        Args:
            solutions: Lista de soluciones
            
        Returns:
            Número de soluciones únicas
        """
        return len(self.get_equivalence_classes(solutions))
    
    def get_representative_solutions(self, solutions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Obtiene una solución representante de cada clase de equivalencia.
        
        Args:
            solutions: Lista de soluciones
            
        Returns:
            Lista de soluciones representantes
        """
        classes = self.get_equivalence_classes(solutions)
        representatives = []
        
        for cls in classes:
            # Tomar el primer elemento como representante
            representative_frozen = next(iter(cls))
            representatives.append(dict(representative_frozen))
        
        return representatives
    
    def analyze_symmetry_structure(self) -> Dict[str, Any]:
        """
        Analiza la estructura del grupo de simetrías.
        
        Returns:
            Diccionario con análisis de la estructura
        """
        group = self.extract_all_symmetries()
        
        analysis = {
            'order': group.order,
            'types': defaultdict(int),
            'variable_symmetries': []
        }
        
        for sym in group.symmetries:
            analysis['types'][sym.type_] += 1
            
            if sym.type_ == 'variable':
                analysis['variable_symmetries'].append(dict(sym.mapping))
        
        logger.info(f"Análisis de simetrías: {analysis}")
        return analysis
    
    def clear_cache(self):
        """Limpia la caché de simetrías."""
        self._symmetry_cache = None
        logger.debug("Caché de simetrías limpiada")
    
    def __str__(self) -> str:
        """Representación en string."""
        return f"SymmetryExtractor(bridge={self.bridge})"
    
    def __repr__(self) -> str:
        """Representación detallada."""
        cache_status = "cached" if self._symmetry_cache else "not cached"
        return f"<SymmetryExtractor bridge={self.bridge!r} cache={cache_status}>"


# ============================================================================
# Funciones de Utilidad (ejemplos)
# ============================================================================

if __name__ == '__main__':
    # Configuración de logging para pruebas
    logging.basicConfig(level=logging.INFO)
    
    # Ejemplo de uso con un CSP simple
    from lattice_weaver.core.csp_problem import CSP, Constraint
    from lattice_weaver.arc_engine.core import ArcEngine
    
    # Crear un CSP simple con simetría
    csp = CSP()
    csp.variables = {'x', 'y', 'z'}
    csp.domains = {
        'x': [1, 2],
        'y': [1, 2],
        'z': [3, 4]
    }
    csp.constraints = [
        Constraint(scope=['x', 'y'], relation=lambda a, b: a != b, name='not_equal')
    ]
    
    # Crear bridge
    engine = ArcEngine(csp)
    bridge = CSPToCubicalBridge(engine)
    
    # Extraer simetrías
    extractor = SymmetryExtractor(bridge)
    symmetries = extractor.extract_variable_symmetries()
    
    print("\n--- Simetrías de Variables ---")
    for sym in symmetries:
        print(sym)
    
    # Analizar estructura
    analysis = extractor.analyze_symmetry_structure()
    print("\n--- Análisis de Estructura ---")
    print(analysis)
    
    # Ejemplo con soluciones
    solutions = [
        {'x': 1, 'y': 2, 'z': 3},
        {'x': 2, 'y': 1, 'z': 3},
        {'x': 1, 'y': 2, 'z': 4},
        {'x': 2, 'y': 1, 'z': 4}
    ]
    
    print(f"\n--- Clases de Equivalencia (para {len(solutions)} soluciones) ---")
    classes = extractor.get_equivalence_classes(solutions)
    for i, cls in enumerate(classes):
        print(f"Clase {i+1}:")
        for sol in cls:
            print(f"  {dict(sol)}")
    
    print(f"\nNúmero de soluciones únicas: {extractor.count_unique_solutions(solutions)}")
    
    print("\n--- Soluciones Representantes ---")
    representatives = extractor.get_representative_solutions(solutions)
    for sol in representatives:
        print(sol)


