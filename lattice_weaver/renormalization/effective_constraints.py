# lattice_weaver/renormalization/effective_constraints.py

"""
Derivación de Restricciones Efectivas

Este módulo implementa la derivación de restricciones efectivas entre
variables renormalizadas. Una restricción efectiva captura el comportamiento
combinado de las restricciones originales que cruzan entre dos grupos de variables.

Principios de diseño:
- Tabulación: Precomputar la tabla de la restricción efectiva para O(1) lookup
- Lazy evaluation: Restricciones se computan solo cuando se necesitan
- Compresión: Almacenar tablas de forma compacta
"""

from typing import Set, FrozenSet, Dict, Tuple, List, Callable, Any
from collections import defaultdict
import hashlib
import itertools

from .effective_domains import LazyEffectiveDomain


class EffectiveConstraintDeriver:
    """
    Deriva restricciones efectivas entre variables renormalizadas.
    
    Una restricción efectiva R'(config_i, config_j) es una función booleana
    que indica si dos configuraciones de grupos de variables son compatibles
    dadas las restricciones originales que los conectan.
    """
    
    def __init__(self, use_cache: bool = True):
        """
        Inicializa derivador.
        
        Args:
            use_cache: Si usar caché de isomorfismos de restricciones
        """
        self.use_cache = use_cache
        
        # Caché: canonical_form → effective_constraint_table
        self.cache: Dict[str, FrozenSet[Tuple]] = {}
        
        # Estadísticas
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'computations': 0
        }
    
    def derive(
        self,
        csp,
        group1: Set[str],
        group2: Set[str],
        effective_domain1: LazyEffectiveDomain,
        effective_domain2: LazyEffectiveDomain
    ) -> Callable[[Tuple, Tuple], bool]:
        """
        Deriva una restricción efectiva entre dos grupos de variables.
        
        Args:
            csp: CSP original
            group1: Primer grupo de variables
            group2: Segundo grupo de variables
            effective_domain1: Dominio efectivo del primer grupo
            effective_domain2: Dominio efectivo del segundo grupo
        
        Returns:
            Función que representa la restricción efectiva
        """
        # Encontrar restricciones que cruzan entre grupos
        crossing_constraints = self._find_crossing_constraints(csp, group1, group2)
        
        if not crossing_constraints:
            # No hay restricciones entre grupos, siempre compatible
            return lambda c1, c2: True
        
        # Verificar caché
        if self.use_cache:
            canonical = self._canonicalize_crossing_constraints(
                csp, group1, group2, crossing_constraints
            )
            if canonical in self.cache:
                self.stats["cache_hits"] += 1
                table = self.cache[canonical]
                return lambda c1, c2: (c1, c2) in table
            self.stats["cache_misses"] += 1
        
        self.stats["computations"] += 1
        
        # Tabular la restricción efectiva
        table = self._tabulate_effective_constraint(
            group1, group2, effective_domain1, effective_domain2, crossing_constraints
        )
        
        # Cachear resultado
        if self.use_cache:
            self.cache[canonical] = table
        
        return lambda c1, c2: (c1, c2) in table
    
    def _find_crossing_constraints(self, csp, group1: Set[str], group2: Set[str]) -> List[Any]:
        """
        Encuentra restricciones que conectan variables de group1 y group2.
        """
        crossing_constraints = []
        for constraint in csp.constraints:
            vars_in_group1 = [v for v in constraint.scope if v in group1]
            vars_in_group2 = [v for v in constraint.scope if v in group2]
            
            if vars_in_group1 and vars_in_group2:
                crossing_constraints.append(constraint)
        return crossing_constraints
    
    def _tabulate_effective_constraint(
        self,
        group1: Set[str],
        group2: Set[str],
        effective_domain1: LazyEffectiveDomain,
        effective_domain2: LazyEffectiveDomain,
        crossing_constraints: List[Any]
    ) -> FrozenSet[Tuple]:
        """
        Precomputa la tabla de la restricción efectiva.
        
        La tabla contiene pares de configuraciones (config1, config2) que son
        compatibles según las restricciones que cruzan.
        """
        compatible_pairs = set()
        
        # Iterar sobre todas las combinaciones de configuraciones de los dominios efectivos
        for config1 in effective_domain1:
            for config2 in effective_domain2:
                # Crear una asignación combinada para verificar las restricciones
                assignment = {}
                # print(f"DEBUG: Checking config1: {config1}, config2: {config2}")
                for var, val in zip(sorted(list(group1)), config1):
                    assignment[var] = val
                for var, val in zip(sorted(list(group2)), config2):
                    assignment[var] = val
                
                # Verificar si esta asignación combinada satisface todas las restricciones que cruzan
                is_valid = self._is_valid_assignment(assignment, crossing_constraints)
                # if not is_valid:
                #     print(f"DEBUG: Invalid assignment for config1: {config1}, config2: {config2}")
                if is_valid:
                    compatible_pairs.add((config1, config2))
        

        return frozenset(compatible_pairs)
    
    def _is_valid_assignment(self, assignment: Dict, constraints: List) -> bool:
        """
        Verifica si una asignación satisface todas las restricciones utilizando
        la función `verify_solution` del módulo `csp_problem`.
        
        Args:
            assignment: Diccionario variable → valor
            constraints: Lista de restricciones a verificar
        
        Returns:
            True si todas las restricciones se satisfacen
        """
        # Crear un CSP temporal para usar verify_solution
        from ..core.csp_problem import CSP, verify_solution
        temp_csp = CSP(variables=set(assignment.keys()),
                       domains={var: frozenset([val]) for var, val in assignment.items()},
                       constraints=constraints)
        return verify_solution(temp_csp, assignment)
    
    def _canonicalize_crossing_constraints(
        self,
        csp,
        group1: Set[str],
        group2: Set[str],
        crossing_constraints: List[Any]
    ) -> str:
        """
        Canonicaliza el conjunto de restricciones que cruzan para el caché.
        
        Considera los grupos y las restricciones para generar un hash único.
        """
        # Ordenar grupos para consistencia
        sorted_group1 = tuple(sorted(list(group1)))
        sorted_group2 = tuple(sorted(list(group2)))
        
        # Canonicalizar cada restricción
        canonical_constraints = []
        for constraint in crossing_constraints:
            # Asumimos que las restricciones tienen un método para canonicalizarse
            # o que su representación por defecto es suficientemente canónica
            canonical_constraints.append(str(constraint))
        canonical_constraints.sort()
        
        # Combinar y hashear
        canonical_str = f"{sorted_group1}{sorted_group2}{tuple(canonical_constraints)}"
        return hashlib.md5(canonical_str.encode()).hexdigest()
    
    def get_cache_stats(self) -> Dict:
        """
        Obtiene estadísticas del caché.
        """
        total_requests = self.stats["cache_hits"] + self.stats["cache_misses"]
        hit_rate = self.stats["cache_hits"] / total_requests if total_requests > 0 else 0.0
        
        return {
            "cache_size": len(self.cache),
            "cache_hits": self.stats["cache_hits"],
            "cache_misses": self.stats["cache_misses"],
            "hit_rate": hit_rate,
            "computations": self.stats["computations"]
        }
    
    def clear_cache(self) -> None:
        """
        Limpia caché (útil para testing).
        """
        self.cache.clear()
        self.stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "computations": 0
        }


class LazyEffectiveConstraint:
    """
    Restricción efectiva lazy: se computa solo cuando se accede.
    """
    
    def __init__(
        self,
        deriver: EffectiveConstraintDeriver,
        csp,
        group1: Set[str],
        group2: Set[str],
        effective_domain1: LazyEffectiveDomain,
        effective_domain2: LazyEffectiveDomain
    ):
        self.deriver = deriver
        self.csp = csp
        self.group1 = group1
        self.group2 = group2
        self.effective_domain1 = effective_domain1
        self.effective_domain2 = effective_domain2
        self._constraint_func: Optional[Callable[[Tuple, Tuple], bool]] = None
    
    def get(self) -> Callable[[Tuple, Tuple], bool]:
        """
        Obtiene la función de restricción efectiva (computándola si es necesario).
        """
        if self._constraint_func is None:
            self._constraint_func = self.deriver.derive(
                self.csp,
                self.group1,
                self.group2,
                self.effective_domain1,
                self.effective_domain2
            )
        return self._constraint_func
    
    def __call__(self, config1: Tuple, config2: Tuple) -> bool:
        """
        Permite llamar a la restricción efectiva directamente.
        """
        return self.get()(config1, config2)

