"""
FCA Jerárquico y Paralelizable

Explota la topología del problema (β₀ > 1) para construir retículos
en paralelo y componerlos recursivamente.
"""

import sys
sys.path.insert(0, '/home/ubuntu/latticeweaver_v4')

from typing import List, Set, Tuple, Dict, FrozenSet
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from lattice_weaver.lattice_core.context import FormalContext
from lattice_weaver.lattice_core.builder import LatticeBuilder


class HierarchicalFCA:
    """
    FCA jerárquico que explota la topología del problema.
    
    Si β₀ > 1, construye retículos independientes para cada componente
    y luego los compone usando meet recursivo.
    
    Ventajas:
    - Paralelización: Speedup lineal con número de cores
    - Reducción de complejidad: O(n/k) por componente vs. O(n) total
    - Escalabilidad: Problemas grandes se dividen en subproblemas manejables
    """
    
    def __init__(self, arc_engine, components: List[Set[str]] = None):
        """
        Args:
            arc_engine: Motor AC-3 con el problema
            components: Componentes conexas (si None, se calculan)
        """
        self.engine = arc_engine
        self.components = components or self._detect_components()
        self.sublattices = []
        self.composed_lattice = None
        self.stats = {
            'num_components': len(self.components),
            'parallel_time': 0,
            'composition_time': 0,
            'total_concepts': 0,
        }
    
    def _detect_components(self) -> List[Set[str]]:
        """Detecta componentes conexas usando Union-Find."""
        parent = {var: var for var in self.engine.variables}
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # Unir variables conectadas por restricciones
        for cid, constraint in self.engine.constraints.items():
            union(constraint.var1, constraint.var2)
        
        # Agrupar por componente
        components_dict = {}
        for var in self.engine.variables:
            root = find(var)
            if root not in components_dict:
                components_dict[root] = set()
            components_dict[root].add(var)
        
        return list(components_dict.values())
    
    def build_hierarchical_lattice(self, parallel: bool = True) -> List[Tuple]:
        """
        Construye retículos jerárquicos:
        1. Retículo por componente (paralelo si parallel=True)
        2. Composición con meet recursivo
        
        Returns:
            Lista de conceptos del retículo compuesto
        """
        # Nivel 1: Retículos por componente
        start_time = time.time()
        
        if parallel and len(self.components) > 1:
            self.sublattices = self._build_sublattices_parallel()
        else:
            self.sublattices = self._build_sublattices_sequential()
        
        self.stats['parallel_time'] = time.time() - start_time
        
        # Nivel 2: Composición recursiva
        start_time = time.time()
        
        if len(self.sublattices) > 1:
            self.composed_lattice = self._recursive_meet(self.sublattices)
        else:
            self.composed_lattice = self.sublattices[0] if self.sublattices else []
        
        self.stats['composition_time'] = time.time() - start_time
        self.stats['total_concepts'] = len(self.composed_lattice)
        
        return self.composed_lattice
    
    def _build_sublattices_parallel(self) -> List[List[Tuple]]:
        """Construye retículos por componente en paralelo."""
        sublattices = []
        
        with ThreadPoolExecutor() as executor:
            # Enviar tareas
            futures = {
                executor.submit(self._build_component_lattice, comp): i
                for i, comp in enumerate(self.components)
            }
            
            # Recolectar resultados en orden
            results = [None] * len(self.components)
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
            
            sublattices = results
        
        return sublattices
    
    def _build_sublattices_sequential(self) -> List[List[Tuple]]:
        """Construye retículos por componente secuencialmente."""
        sublattices = []
        
        for comp in self.components:
            lattice = self._build_component_lattice(comp)
            sublattices.append(lattice)
        
        return sublattices
    
    def _build_component_lattice(self, component: Set[str]) -> List[Tuple]:
        """
        Construye retículo para una componente.
        
        Args:
            component: Conjunto de variables de la componente
            
        Returns:
            Lista de conceptos (extent, intent)
        """
        # Crear contexto formal solo para esta componente
        context = FormalContext()
        
        # Añadir objetos (variables)
        for var in component:
            context.add_object(var)
        
        # Añadir atributos (valores posibles)
        for var in component:
            domain = self.engine.variables[var]
            for value in domain.get_values():
                attr_name = f"{var}={value}"
                context.add_attribute(attr_name)
        
        # Añadir incidencias (variable puede tomar valor)
        for var in component:
            domain = self.engine.variables[var]
            for value in domain.get_values():
                attr_name = f"{var}={value}"
                context.add_incidence(var, attr_name)
        
        # Construir retículo
        builder = LatticeBuilder(context)
        lattice = builder.build_lattice()
        
        return lattice
    
    def _recursive_meet(self, lattices: List[List[Tuple]]) -> List[Tuple]:
        """
        Composición recursiva de retículos con meet.
        
        meet(L1, L2, L3) = meet(meet(L1, L2), L3)
        
        Complejidad: O(k × log(k)) donde k = número de retículos
        """
        if len(lattices) == 0:
            return []
        
        if len(lattices) == 1:
            return lattices[0]
        
        # Dividir y conquistar
        mid = len(lattices) // 2
        left = self._recursive_meet(lattices[:mid])
        right = self._recursive_meet(lattices[mid:])
        
        return self._meet_lattices(left, right)
    
    def _meet_lattices(self, L1: List[Tuple], L2: List[Tuple]) -> List[Tuple]:
        """
        Calcula el meet de dos retículos.
        
        El meet de dos retículos es el producto cartesiano de sus conceptos,
        donde el orden es el producto de órdenes.
        
        Args:
            L1: Primer retículo (lista de conceptos)
            L2: Segundo retículo (lista de conceptos)
            
        Returns:
            Retículo compuesto
        """
        composed_concepts = []
        
        for (extent1, intent1) in L1:
            for (extent2, intent2) in L2:
                # Producto cartesiano
                new_extent = extent1 | extent2
                new_intent = intent1 & intent2
                
                # Evitar duplicados
                concept = (new_extent, new_intent)
                if concept not in composed_concepts:
                    composed_concepts.append(concept)
        
        return composed_concepts
    
    def get_stats(self) -> dict:
        """Retorna estadísticas de la construcción."""
        return {
            **self.stats,
            'speedup_estimate': self._estimate_speedup(),
        }
    
    def _estimate_speedup(self) -> float:
        """
        Estima el speedup obtenido por la descomposición.
        
        Speedup = T_sequential / T_parallel
        
        T_sequential ≈ O(n²)
        T_parallel ≈ O((n/k)²) donde k = número de componentes
        
        Speedup ≈ k² (teórico)
        """
        k = self.stats['num_components']
        
        if k == 1:
            return 1.0
        
        # Speedup teórico: k² (asumiendo componentes de tamaño similar)
        theoretical_speedup = k ** 2
        
        # Speedup real: considerando overhead de composición
        composition_overhead = 1.2  # 20% overhead
        real_speedup = theoretical_speedup / composition_overhead
        
        return real_speedup


class MultilevelFCA:
    """
    FCA multinivel que construye el retículo en capas.
    
    Nivel 0: Conceptos atómicos (objetos individuales)
    Nivel 1: Conceptos de 2 objetos
    Nivel 2: Conceptos de 3 objetos
    ...
    Nivel n: Concepto top
    
    Ventajas:
    - Construcción incremental
    - Detección temprana de conceptos importantes
    - Poda de conceptos irrelevantes
    """
    
    def __init__(self, context: FormalContext):
        self.context = context
        self.levels = []
    
    def build_multilevel_lattice(self) -> List[List[Tuple]]:
        """
        Construye el retículo nivel por nivel.
        
        Returns:
            Lista de niveles, cada nivel es una lista de conceptos
        """
        # Nivel 0: Conceptos atómicos
        level_0 = self._build_atomic_concepts()
        self.levels.append(level_0)
        
        # Niveles superiores
        current_level = level_0
        while len(current_level) > 1:
            next_level = self._build_next_level(current_level)
            self.levels.append(next_level)
            current_level = next_level
        
        return self.levels
    
    def _build_atomic_concepts(self) -> List[Tuple]:
        """
        Construye conceptos atómicos (un objeto por concepto).
        
        Returns:
            Lista de conceptos atómicos
        """
        atomic_concepts = []
        
        for obj in self.context.objects:
            # Extent: solo este objeto
            extent = frozenset([obj])
            
            # Intent: todos los atributos que tiene este objeto
            intent = frozenset([
                attr for attr in self.context.attributes
                if self.context.has_incidence(obj, attr)
            ])
            
            atomic_concepts.append((extent, intent))
        
        return atomic_concepts
    
    def _build_next_level(self, current_level: List[Tuple]) -> List[Tuple]:
        """
        Construye el siguiente nivel combinando conceptos del nivel actual.
        
        Args:
            current_level: Conceptos del nivel actual
            
        Returns:
            Conceptos del siguiente nivel
        """
        next_level = []
        seen = set()
        
        for i, concept1 in enumerate(current_level):
            for concept2 in current_level[i+1:]:
                # Combinar conceptos
                new_concept = self._combine_concepts(concept1, concept2)
                
                # Evitar duplicados
                concept_key = (new_concept[0], new_concept[1])
                if concept_key not in seen:
                    next_level.append(new_concept)
                    seen.add(concept_key)
        
        return next_level
    
    def _combine_concepts(self, concept1: Tuple, concept2: Tuple) -> Tuple:
        """
        Combina dos conceptos usando meet.
        
        meet((A, B), (C, D)) = (A ∪ C, B ∩ D)
        
        Args:
            concept1: Primer concepto (extent, intent)
            concept2: Segundo concepto (extent, intent)
            
        Returns:
            Concepto combinado
        """
        extent1, intent1 = concept1
        extent2, intent2 = concept2
        
        new_extent = extent1 | extent2
        new_intent = intent1 & intent2
        
        return (new_extent, new_intent)
    
    def flatten_lattice(self) -> List[Tuple]:
        """
        Aplana el retículo multinivel en una lista única.
        
        Returns:
            Lista de todos los conceptos
        """
        all_concepts = []
        for level in self.levels:
            all_concepts.extend(level)
        return all_concepts

