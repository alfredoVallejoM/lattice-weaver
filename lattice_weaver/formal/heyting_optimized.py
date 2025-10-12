# lattice_weaver/formal/heyting_optimized.py

"""
Álgebra de Heyting Optimizada - Fase 11

Implementa operaciones meet y join optimizadas con caché y precomputación
para acelerar la construcción de retículos FCA y operaciones lógicas.

Autor: LatticeWeaver Team
Fecha: 11 de Octubre de 2025
Versión: 1.0
"""

from typing import Dict, Tuple, Set, List, Optional, FrozenSet
from functools import lru_cache
import logging

from .heyting_algebra import HeytingElement, HeytingAlgebra

logger = logging.getLogger(__name__)


class OptimizedHeytingAlgebra(HeytingAlgebra):
    """
    Álgebra de Heyting con operaciones meet/join optimizadas.
    
    Optimizaciones:
    1. Caché de resultados de meet/join
    2. Precomputación de meets frecuentes
    3. Algoritmo divide-and-conquer para meet múltiple
    4. Detección de casos especiales
    
    Extiende HeytingAlgebra sin romper compatibilidad.
    """
    
    def __init__(self, name: str = "H_opt"):
        """
        Inicializa álgebra de Heyting optimizada.
        
        Args:
            name: Nombre del álgebra
        """
        super().__init__(name)
        
        # Caché adicional para optimizaciones
        self._meet_cache: Dict[Tuple[str, str], HeytingElement] = {}
        self._join_cache: Dict[Tuple[str, str], HeytingElement] = {}
        
        # Estadísticas
        self._meet_hits = 0
        self._meet_misses = 0
        self._join_hits = 0
        self._join_misses = 0
        
        logger.info(f"OptimizedHeytingAlgebra '{name}' inicializada")
    
    def precompute_frequent_meets(self):
        """
        Precomputa meets entre elementos frecuentemente usados.
        
        Estrategia:
        1. Identificar pares de elementos "cercanos" en el orden
        2. Precomputar sus meets
        3. Almacenar en caché
        """
        logger.info("Precomputando meets frecuentes...")
        
        precomputed = 0
        
        for e1 in self.elements:
            # Encontrar elementos inmediatamente por debajo
            immediate_lower = self._find_immediate_lower(e1)
            
            # Precomputar meet con cada uno
            for e2 in immediate_lower:
                key = self._make_cache_key(e1, e2)
                if key not in self._meet_cache:
                    result = self._compute_meet_uncached(e1, e2)
                    self._meet_cache[key] = result
                    precomputed += 1
        
        logger.info(f"Precomputados {precomputed} meets")
    
    def _find_immediate_lower(self, element: HeytingElement) -> List[HeytingElement]:
        """
        Encuentra elementos inmediatamente por debajo de element en el orden.
        
        Args:
            element: Elemento de referencia
        
        Returns:
            Lista de elementos inmediatamente inferiores
        """
        immediate = []
        
        for e in self.elements:
            if e == element:
                continue
            
            # e está por debajo de element
            if not self.leq(e, element):
                continue
            
            # e es inmediato si no hay otro elemento entre e y element
            is_immediate = True
            for e2 in self.elements:
                if e2 == e or e2 == element:
                    continue
                if self.leq(e, e2) and self.leq(e2, element):
                    is_immediate = False
                    break
            
            if is_immediate:
                immediate.append(e)
        
        return immediate
    
    def meet(self, a: HeytingElement, b: HeytingElement) -> HeytingElement:
        """
        Calcula meet optimizado con caché.
        
        Args:
            a: Primer elemento
            b: Segundo elemento
        
        Returns:
            Ínfimo de a y b
        """
        # Casos triviales
        if a == b:
            return a
        if a == self.bottom or b == self.bottom:
            return self.bottom
        if a == self.top:
            return b
        if b == self.top:
            return a
        
        # Buscar en caché
        key = self._make_cache_key(a, b)
        if key in self._meet_cache:
            self._meet_hits += 1
            return self._meet_cache[key]
        
        # Buscar en tabla heredada
        table_key = (a, b)
        if table_key in self._meet_table:
            self._meet_hits += 1
            result = self._meet_table[table_key]
            self._meet_cache[key] = result
            return result
        
        # Calcular y cachear
        self._meet_misses += 1
        result = self._compute_meet_uncached(a, b)
        self._meet_cache[key] = result
        self._meet_table[table_key] = result
        self._meet_table[(b, a)] = result
        
        return result
    
    def _make_cache_key(self, a: HeytingElement, b: HeytingElement) -> Tuple[str, str]:
        """
        Crea clave de caché canónica.
        
        Usa orden lexicográfico para garantizar unicidad.
        
        Args:
            a: Primer elemento
            b: Segundo elemento
        
        Returns:
            Tupla ordenada de nombres
        """
        return tuple(sorted([a.name, b.name]))
    
    def _compute_meet_uncached(self, a: HeytingElement, b: HeytingElement) -> HeytingElement:
        """
        Calcula meet sin usar caché.
        
        Optimización: Si los elementos tienen valores (conjuntos),
        usar intersección directamente.
        
        Args:
            a: Primer elemento
            b: Segundo elemento
        
        Returns:
            Ínfimo de a y b
        """
        # Optimización para elementos con valores de conjunto
        if (hasattr(a, 'value') and hasattr(b, 'value') and 
            a.value is not None and b.value is not None):
            
            if isinstance(a.value, (set, frozenset)) and isinstance(b.value, (set, frozenset)):
                meet_value = a.value.intersection(b.value)
                
                # Buscar elemento con ese valor
                for e in self.elements:
                    if hasattr(e, 'value') and e.value == meet_value:
                        return e
        
        # Fallback: algoritmo estándar
        candidates = [c for c in self.elements if self.leq(c, a) and self.leq(c, b)]
        
        if not candidates:
            if self.bottom:
                return self.bottom
            raise ValueError(f"No se encontró ínfimo para {a} ∧ {b}")
        
        # Encontrar el máximo de los candidatos
        result = max(candidates, key=lambda c: sum(1 for d in self.elements if self.leq(d, c)))
        
        return result
    
    def join(self, a: HeytingElement, b: HeytingElement) -> HeytingElement:
        """
        Calcula join optimizado con caché.
        
        Args:
            a: Primer elemento
            b: Segundo elemento
        
        Returns:
            Supremo de a y b
        """
        # Casos triviales
        if a == b:
            return a
        if a == self.top or b == self.top:
            return self.top
        if a == self.bottom:
            return b
        if b == self.bottom:
            return a
        
        # Buscar en caché
        key = self._make_cache_key(a, b)
        if key in self._join_cache:
            self._join_hits += 1
            return self._join_cache[key]
        
        # Buscar en tabla heredada
        table_key = (a, b)
        if table_key in self._join_table:
            self._join_hits += 1
            result = self._join_table[table_key]
            self._join_cache[key] = result
            return result
        
        # Calcular y cachear
        self._join_misses += 1
        result = self._compute_join_uncached(a, b)
        self._join_cache[key] = result
        self._join_table[table_key] = result
        self._join_table[(b, a)] = result
        
        return result
    
    def _compute_join_uncached(self, a: HeytingElement, b: HeytingElement) -> HeytingElement:
        """
        Calcula join sin usar caché.
        
        Optimización: Si los elementos tienen valores (conjuntos),
        usar unión directamente.
        
        Args:
            a: Primer elemento
            b: Segundo elemento
        
        Returns:
            Supremo de a y b
        """
        # Optimización para elementos con valores de conjunto
        if (hasattr(a, 'value') and hasattr(b, 'value') and 
            a.value is not None and b.value is not None):
            
            if isinstance(a.value, (set, frozenset)) and isinstance(b.value, (set, frozenset)):
                join_value = a.value.union(b.value)
                
                # Buscar elemento con ese valor
                for e in self.elements:
                    if hasattr(e, 'value') and e.value == join_value:
                        return e
        
        # Fallback: algoritmo estándar
        candidates = [c for c in self.elements if self.leq(a, c) and self.leq(b, c)]
        
        if not candidates:
            if self.top:
                return self.top
            raise ValueError(f"No se encontró supremo para {a} ∨ {b}")
        
        # Encontrar el mínimo de los candidatos
        result = min(candidates, key=lambda c: sum(1 for d in self.elements if self.leq(c, d) and c != d))
        
        return result
    
    def meet_multiple(self, elements: List[HeytingElement]) -> HeytingElement:
        """
        Calcula meet de múltiples elementos de manera eficiente.
        
        Usa estrategia divide-and-conquer para aprovechar caché.
        
        Args:
            elements: Lista de elementos
        
        Returns:
            Meet de todos los elementos
        """
        if not elements:
            return self.top
        if len(elements) == 1:
            return elements[0]
        if len(elements) == 2:
            return self.meet(elements[0], elements[1])
        
        # Dividir en dos mitades
        mid = len(elements) // 2
        left_meet = self.meet_multiple(elements[:mid])
        right_meet = self.meet_multiple(elements[mid:])
        
        return self.meet(left_meet, right_meet)
    
    def join_multiple(self, elements: List[HeytingElement]) -> HeytingElement:
        """
        Calcula join de múltiples elementos de manera eficiente.
        
        Usa estrategia divide-and-conquer para aprovechar caché.
        
        Args:
            elements: Lista de elementos
        
        Returns:
            Join de todos los elementos
        """
        if not elements:
            return self.bottom
        if len(elements) == 1:
            return elements[0]
        if len(elements) == 2:
            return self.join(elements[0], elements[1])
        
        # Dividir en dos mitades
        mid = len(elements) // 2
        left_join = self.join_multiple(elements[:mid])
        right_join = self.join_multiple(elements[mid:])
        
        return self.join(left_join, right_join)
    
    def get_cache_statistics(self) -> Dict[str, int]:
        """
        Obtiene estadísticas de uso de caché.
        
        Returns:
            Diccionario con estadísticas
        """
        total_meet = self._meet_hits + self._meet_misses
        total_join = self._join_hits + self._join_misses
        
        meet_hit_rate = (self._meet_hits / total_meet * 100) if total_meet > 0 else 0
        join_hit_rate = (self._join_hits / total_join * 100) if total_join > 0 else 0
        
        return {
            'meet_hits': self._meet_hits,
            'meet_misses': self._meet_misses,
            'meet_hit_rate': round(meet_hit_rate, 2),
            'join_hits': self._join_hits,
            'join_misses': self._join_misses,
            'join_hit_rate': round(join_hit_rate, 2),
            'meet_cache_size': len(self._meet_cache),
            'join_cache_size': len(self._join_cache)
        }
    
    def clear_cache(self):
        """Limpia las cachés de meet y join."""
        self._meet_cache.clear()
        self._join_cache.clear()
        self._meet_hits = 0
        self._meet_misses = 0
        self._join_hits = 0
        self._join_misses = 0
        logger.info("Cachés limpiadas")

