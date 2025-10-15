"""
Predicate Cache - Inline Caching de Evaluaciones de Predicados

Cachea resultados de evaluación de predicados con invalidación inteligente.

Optimizaciones:
- LRU cache por predicado
- Invalidación basada en variables modificadas
- Estadísticas de hit rate

Autor: Agente Autónomo - Lattice Weaver
Fecha: 15 de Octubre, 2025
"""

from typing import Dict, Any, Callable, Optional, Tuple, Set
from functools import lru_cache
from collections import OrderedDict
import hashlib


class PredicateCache:
    """
    Caché de evaluaciones de predicados.
    
    Cachea resultados de predicados con invalidación inteligente.
    """
    
    def __init__(self, max_size: int = 10000):
        """
        Inicializa Predicate Cache.
        
        Args:
            max_size: Tamaño máximo del caché
        """
        self.max_size = max_size
        
        # Caché: (predicate_id, assignment_hash) -> resultado
        self.cache: OrderedDict[Tuple[int, str], bool] = OrderedDict()
        
        # Índice: variable -> entradas que la involucran
        self.var_to_entries: Dict[str, Set[Tuple[int, str]]] = {}
        
        # Estadísticas
        self.stats = {
            'hits': 0,
            'misses': 0,
            'invalidations': 0,
            'evictions': 0
        }
    
    def _hash_assignment(self, assignment: Dict[str, Any]) -> str:
        """
        Crea hash de asignación.
        
        Args:
            assignment: Asignación de variables
        
        Returns:
            Hash string
        """
        # Ordenar por clave para consistencia
        items = sorted(assignment.items())
        content = str(items)
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(
        self,
        predicate: Callable,
        assignment: Dict[str, Any],
        variables: Set[str]
    ) -> Optional[bool]:
        """
        Obtiene resultado cacheado de predicado.
        
        Args:
            predicate: Función predicado
            assignment: Asignación de variables
            variables: Variables involucradas en el predicado
        
        Returns:
            Resultado cacheado o None si no está en caché
        """
        predicate_id = id(predicate)
        
        # Filtrar assignment a solo variables relevantes
        relevant_assignment = {
            var: val for var, val in assignment.items()
            if var in variables
        }
        
        assignment_hash = self._hash_assignment(relevant_assignment)
        key = (predicate_id, assignment_hash)
        
        if key in self.cache:
            # Hit: mover al final (LRU)
            self.cache.move_to_end(key)
            self.stats['hits'] += 1
            return self.cache[key]
        else:
            # Miss
            self.stats['misses'] += 1
            return None
    
    def put(
        self,
        predicate: Callable,
        assignment: Dict[str, Any],
        variables: Set[str],
        result: bool
    ):
        """
        Guarda resultado de predicado en caché.
        
        Args:
            predicate: Función predicado
            assignment: Asignación de variables
            variables: Variables involucradas
            result: Resultado del predicado
        """
        predicate_id = id(predicate)
        
        # Filtrar assignment
        relevant_assignment = {
            var: val for var, val in assignment.items()
            if var in variables
        }
        
        assignment_hash = self._hash_assignment(relevant_assignment)
        key = (predicate_id, assignment_hash)
        
        # Añadir al caché
        self.cache[key] = result
        
        # Actualizar índice de variables
        for var in variables:
            if var not in self.var_to_entries:
                self.var_to_entries[var] = set()
            self.var_to_entries[var].add(key)
        
        # Eviction si excede tamaño
        if len(self.cache) > self.max_size:
            # Eliminar entrada más antigua (FIFO)
            oldest_key, _ = self.cache.popitem(last=False)
            self.stats['evictions'] += 1
            
            # Limpiar índice
            self._remove_from_index(oldest_key)
    
    def invalidate_variable(self, variable: str):
        """
        Invalida entradas que involucran una variable.
        
        Args:
            variable: Variable modificada
        """
        if variable not in self.var_to_entries:
            return
        
        # Obtener entradas a invalidar
        entries = self.var_to_entries[variable].copy()
        
        # Eliminar del caché
        for key in entries:
            if key in self.cache:
                del self.cache[key]
                self.stats['invalidations'] += 1
        
        # Limpiar índice
        del self.var_to_entries[variable]
    
    def _remove_from_index(self, key: Tuple[int, str]):
        """Elimina entrada del índice de variables."""
        # Buscar en índice y eliminar
        for var, entries in list(self.var_to_entries.items()):
            if key in entries:
                entries.discard(key)
                if not entries:
                    del self.var_to_entries[var]
    
    def clear(self):
        """Limpia todo el caché."""
        self.cache.clear()
        self.var_to_entries.clear()
    
    def get_stats(self) -> dict:
        """Retorna estadísticas."""
        total = self.stats['hits'] + self.stats['misses']
        hit_rate = self.stats['hits'] / total if total > 0 else 0
        
        return {
            **self.stats,
            'total_requests': total,
            'hit_rate': hit_rate,
            'cache_size': len(self.cache)
        }
    
    def reset_stats(self):
        """Resetea estadísticas."""
        self.stats = {
            'hits': 0,
            'misses': 0,
            'invalidations': 0,
            'evictions': 0
        }


class CachedPredicate:
    """
    Wrapper para predicado con caché inline.
    
    Uso:
        cached_pred = CachedPredicate(predicate, variables, cache)
        result = cached_pred(assignment)
    """
    
    def __init__(
        self,
        predicate: Callable,
        variables: Set[str],
        cache: PredicateCache
    ):
        """
        Inicializa Cached Predicate.
        
        Args:
            predicate: Función predicado original
            variables: Variables involucradas
            cache: Caché compartido
        """
        self.predicate = predicate
        self.variables = variables
        self.cache = cache
    
    def __call__(self, assignment: Dict[str, Any]) -> bool:
        """
        Evalúa predicado con caché.
        
        Args:
            assignment: Asignación de variables
        
        Returns:
            Resultado del predicado
        """
        # Intentar obtener del caché
        cached_result = self.cache.get(self.predicate, assignment, self.variables)
        
        if cached_result is not None:
            return cached_result
        
        # Evaluar predicado
        result = self.predicate(assignment)
        
        # Guardar en caché
        self.cache.put(self.predicate, assignment, self.variables, result)
        
        return result


class PredicateCacheManager:
    """
    Gestor de cachés de predicados.
    
    Mantiene un caché global y gestiona invalidaciones.
    """
    
    def __init__(self, max_size: int = 10000):
        """
        Inicializa manager.
        
        Args:
            max_size: Tamaño máximo del caché
        """
        self.cache = PredicateCache(max_size)
        self.cached_predicates: Dict[int, CachedPredicate] = {}
    
    def wrap_predicate(
        self,
        predicate: Callable,
        variables: Set[str]
    ) -> CachedPredicate:
        """
        Envuelve predicado con caché.
        
        Args:
            predicate: Función predicado
            variables: Variables involucradas
        
        Returns:
            Predicado cacheado
        """
        predicate_id = id(predicate)
        
        if predicate_id not in self.cached_predicates:
            cached = CachedPredicate(predicate, variables, self.cache)
            self.cached_predicates[predicate_id] = cached
        
        return self.cached_predicates[predicate_id]
    
    def notify_assignment(self, variable: str):
        """
        Notifica que una variable fue asignada.
        
        Invalida entradas relevantes.
        
        Args:
            variable: Variable asignada
        """
        self.cache.invalidate_variable(variable)
    
    def clear(self):
        """Limpia todos los cachés."""
        self.cache.clear()
    
    def get_stats(self) -> dict:
        """Retorna estadísticas."""
        return self.cache.get_stats()
    
    def reset_stats(self):
        """Resetea estadísticas."""
        self.cache.reset_stats()

