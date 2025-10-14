# lattice_weaver/renormalization/effective_domains.py

"""
Derivación de Dominios Efectivos

Este módulo implementa la derivación de dominios efectivos para grupos
de variables en renormalización. Un dominio efectivo contiene todas las
configuraciones válidas de un grupo de variables que satisfacen las
restricciones internas del grupo.

Principios de diseño:
- Caché de isomorfismos: Subproblemas isomorfos se resuelven una vez
- Lazy evaluation: Dominios se computan solo cuando se necesitan
- Aproximación: Para grupos grandes, sampling en lugar de enumeración exhaustiva
"""

from typing import Set, FrozenSet, Dict, Tuple, Optional, List
from collections import defaultdict
import hashlib
import itertools


class EffectiveDomainDeriver:
    """
    Deriva dominios efectivos para grupos de variables.
    
    Un dominio efectivo D'_i para un grupo G_i es el conjunto de todas
    las configuraciones (asignaciones) válidas de las variables en G_i
    que satisfacen las restricciones internas del grupo.
    """
    
    def __init__(self, use_cache: bool = True, max_domain_size: int = 10000):
        """
        Inicializa derivador.
        
        Args:
            use_cache: Si usar caché de isomorfismos
            max_domain_size: Tamaño máximo de dominio efectivo antes de aproximar
        """
        self.use_cache = use_cache
        self.max_domain_size = max_domain_size
        
        # Caché: canonical_form → effective_domain
        self.cache: Dict[str, FrozenSet[Tuple]] = {}
        
        # Estadísticas
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'exact_computations': 0,
            'approximate_computations': 0
        }
    
    def derive(self, csp, group: Set[str]) -> FrozenSet[Tuple]:
        """
        Deriva dominio efectivo para un grupo de variables.
        
        Args:
            csp: CSP original
            group: Conjunto de variables del grupo
        
        Returns:
            Dominio efectivo (conjunto de tuplas válidas)
        """
        # Verificar caché
        if self.use_cache:
            canonical = self._canonicalize_subproblem(csp, group)
            if canonical in self.cache:
                self.stats['cache_hits'] += 1
                return self.cache[canonical]
            self.stats['cache_misses'] += 1
        
        # Extraer subproblema del grupo
        subproblem = self._extract_subproblem(csp, group)
        
        # Estimar tamaño del dominio efectivo
        estimated_size = self._estimate_domain_size(subproblem)
        
        # Decidir si computar exactamente o aproximar
        if estimated_size <= self.max_domain_size:
            effective_domain = self._compute_exact(subproblem)
            self.stats['exact_computations'] += 1
        else:
            effective_domain = self._compute_approximate(subproblem)
            self.stats['approximate_computations'] += 1
        
        # Cachear resultado
        if self.use_cache:
            self.cache[canonical] = effective_domain
        
        return effective_domain
    
    def _extract_subproblem(self, csp, group: Set[str]) -> Dict:
        """
        Extrae subproblema correspondiente a un grupo.
        
        Subproblema incluye:
        - Variables del grupo
        - Dominios de esas variables
        - Restricciones internas (solo involucran variables del grupo)
        """
        # Variables del grupo (ordenadas para consistencia)
        variables = sorted(list(group))
        
        # Dominios
        domains = {var: csp.domains[var] for var in variables}
        
        # Restricciones internas
        internal_constraints = []
        for constraint in csp.constraints:
            # Verificar si todas las variables de la restricción están en el grupo
            if all(var in group for var in constraint.scope):
                internal_constraints.append(constraint)
        
        return {
            'variables': variables,
            'domains': domains,
            'constraints': internal_constraints
        }
    
    def _compute_exact(self, subproblem: Dict) -> FrozenSet[Tuple]:
        """
        Computa dominio efectivo exactamente mediante enumeración.
        
        Genera todas las posibles configuraciones y filtra las válidas.
        """
        variables = subproblem['variables']
        domains = subproblem['domains']
        constraints = subproblem['constraints']
        
        # Generar todas las configuraciones posibles
        domain_lists = [list(domains[var]) for var in variables]
        all_configs = itertools.product(*domain_lists)
        
        # Filtrar configuraciones válidas
        valid_configs = []
        for config in all_configs:
            # Crear asignación
            assignment = dict(zip(variables, config))
            
            # Verificar todas las restricciones
            if self._is_valid_assignment(assignment, constraints):
                valid_configs.append(config)
        
        return frozenset(valid_configs)
    
    def _compute_approximate(self, subproblem: Dict, 
                            num_samples: int = 1000,
                            max_attempts: int = 10000) -> FrozenSet[Tuple]:
        """
        Computa dominio efectivo aproximadamente mediante sampling.
        
        Genera configuraciones aleatorias y retiene las válidas.
        Útil cuando enumeración exhaustiva es intratable.
        
        Args:
            subproblem: Subproblema a resolver
            num_samples: Número de muestras válidas deseadas
            max_attempts: Máximo número de intentos
        
        Returns:
            Conjunto aproximado de configuraciones válidas
        """
        import random
        
        variables = subproblem['variables']
        domains = subproblem['domains']
        constraints = subproblem['constraints']
        
        valid_configs = set()
        attempts = 0
        
        while len(valid_configs) < num_samples and attempts < max_attempts:
            # Generar configuración aleatoria
            config = tuple(random.choice(list(domains[var])) for var in variables)
            
            # Verificar validez
            assignment = dict(zip(variables, config))
            if self._is_valid_assignment(assignment, constraints):
                valid_configs.add(config)
            
            attempts += 1
        
        return frozenset(valid_configs)
    
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
    
    def _estimate_domain_size(self, subproblem: Dict) -> int:
        """
        Estima tamaño del dominio efectivo.
        
        Estimación: producto de tamaños de dominios individuales
        (cota superior, el tamaño real será menor debido a restricciones)
        """
        domains = subproblem['domains']
        size = 1
        for domain in domains.values():
            size *= len(domain)
        return size
    
    def _canonicalize_subproblem(self, csp, group: Set[str]) -> str:
        """
        Canonicaliza subproblema para detección de isomorfismos.
        
        Dos subproblemas son isomorfos si tienen:
        - Misma estructura de restricciones
        - Mismos dominios (módulo renombramiento de variables)
        
        Returns:
            Hash que identifica la clase de isomorfismo
        """
        subproblem = self._extract_subproblem(csp, group)
        
        # Extraer características canónicas
        variables = subproblem['variables']
        domains = subproblem['domains']
        constraints = subproblem['constraints']
        
        # Característica 1: Tamaños de dominios (ordenados)
        domain_sizes = tuple(sorted(len(domains[var]) for var in variables))
        
        # Característica 2: Número de restricciones
        num_constraints = len(constraints)
        
        # Característica 3: Aridades de restricciones (ordenadas)
        constraint_arities = tuple(sorted(len(c.scope) for c in constraints))
        
        # Característica 4: Grado de cada variable (ordenado)
        degrees = []
        for var in variables:
            degree = sum(1 for c in constraints if var in c.scope)
            degrees.append(degree)
        degrees = tuple(sorted(degrees))
        
        # Combinar características
        canonical_features = (
            len(variables),
            domain_sizes,
            num_constraints,
            constraint_arities,
            degrees
        )
        
        # Hash
        canonical_str = str(canonical_features)
        return hashlib.md5(canonical_str.encode()).hexdigest()
    
    def get_cache_stats(self) -> Dict:
        """Obtiene estadísticas del caché."""
        total_requests = self.stats['cache_hits'] + self.stats['cache_misses']
        hit_rate = self.stats['cache_hits'] / total_requests if total_requests > 0 else 0.0
        
        return {
            'cache_size': len(self.cache),
            'cache_hits': self.stats['cache_hits'],
            'cache_misses': self.stats['cache_misses'],
            'hit_rate': hit_rate,
            'exact_computations': self.stats['exact_computations'],
            'approximate_computations': self.stats['approximate_computations']
        }
    
    def clear_cache(self) -> None:
        """Limpia caché (útil para testing)."""
        self.cache.clear()
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'exact_computations': 0,
            'approximate_computations': 0
        }


class LazyEffectiveDomain:
    """
    Dominio efectivo lazy: se computa solo cuando se accede.
    
    Útil para evitar computar dominios que nunca se usan.
    """
    
    def __init__(self, deriver: EffectiveDomainDeriver, csp, group: Set[str]):
        """
        Inicializa dominio lazy.
        
        Args:
            deriver: Derivador de dominios efectivos
            csp: CSP original
            group: Grupo de variables
        """
        self.deriver = deriver
        self.csp = csp
        self.group = group
        self._domain: Optional[FrozenSet[Tuple]] = None
    
    def get(self) -> FrozenSet[Tuple]:
        """Obtiene dominio efectivo (computándolo si es necesario)."""
        if self._domain is None:
            self._domain = self.deriver.derive(self.csp, self.group)
        return self._domain
    
    def is_computed(self) -> bool:
        """Verifica si el dominio ya fue computado."""
        return self._domain is not None
    
    def __len__(self) -> int:
        """Tamaño del dominio."""
        return len(self.get())
    
    def __iter__(self):
        """Itera sobre configuraciones del dominio."""
        return iter(self.get())
    
    def __contains__(self, config: Tuple) -> bool:
        """Verifica si una configuración está en el dominio."""
        return config in self.get()

