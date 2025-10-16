"""
Object Pool Genérico y Agresivo

Implementación de object pooling para reducir allocations y mejorar rendimiento.

Ventajas:
- Reduce allocations de memoria
- Mejora locality de caché
- Reduce presión en garbage collector

Autor: Agente Autónomo - Lattice Weaver
Fecha: 15 de Octubre, 2025
"""

from typing import TypeVar, Generic, Callable, Optional, List
from dataclasses import dataclass
import threading


T = TypeVar('T')


class ObjectPool(Generic[T]):
    """
    Object Pool genérico thread-safe.
    
    Mantiene un pool de objetos reutilizables para evitar allocations repetidas.
    """
    
    def __init__(
        self,
        factory: Callable[[], T],
        reset: Optional[Callable[[T], None]] = None,
        max_size: int = 1000
    ):
        """
        Inicializa Object Pool.
        
        Args:
            factory: Función para crear nuevos objetos
            reset: Función opcional para resetear objetos antes de reutilizar
            max_size: Tamaño máximo del pool
        """
        self.factory = factory
        self.reset = reset
        self.max_size = max_size
        
        self.pool: List[T] = []
        self.lock = threading.Lock()
        
        # Estadísticas
        self.stats = {
            'acquired': 0,
            'released': 0,
            'created': 0,
            'reused': 0,
            'discarded': 0
        }
    
    def acquire(self) -> T:
        """
        Adquiere un objeto del pool.
        
        Si el pool está vacío, crea un nuevo objeto.
        
        Returns:
            Objeto del pool
        """
        with self.lock:
            self.stats['acquired'] += 1
            
            if self.pool:
                obj = self.pool.pop()
                self.stats['reused'] += 1
                return obj
            else:
                obj = self.factory()
                self.stats['created'] += 1
                return obj
    
    def release(self, obj: T):
        """
        Libera un objeto al pool.
        
        Si el pool está lleno, descarta el objeto.
        
        Args:
            obj: Objeto a liberar
        """
        with self.lock:
            self.stats['released'] += 1
            
            # Resetear si hay función de reset
            if self.reset:
                self.reset(obj)
            
            # Añadir al pool si hay espacio
            if len(self.pool) < self.max_size:
                self.pool.append(obj)
            else:
                self.stats['discarded'] += 1
    
    def clear(self):
        """Limpia el pool."""
        with self.lock:
            self.pool.clear()
    
    def get_stats(self) -> dict:
        """Retorna estadísticas del pool."""
        with self.lock:
            return self.stats.copy()
    
    def __enter__(self):
        """Context manager support."""
        return self.acquire()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager support."""
        # No podemos liberar aquí porque no tenemos referencia al objeto
        pass


class PooledObject(Generic[T]):
    """
    Wrapper para objetos del pool que se liberan automáticamente.
    
    Uso:
        with pool.acquire_managed() as obj:
            # usar obj
        # obj se libera automáticamente
    """
    
    def __init__(self, obj: T, pool: ObjectPool[T]):
        """
        Inicializa PooledObject.
        
        Args:
            obj: Objeto del pool
            pool: Pool al que pertenece
        """
        self.obj = obj
        self.pool = pool
    
    def __enter__(self) -> T:
        """Context manager entry."""
        return self.obj
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - libera objeto."""
        self.pool.release(self.obj)


class ObjectPoolWithManagement(ObjectPool[T]):
    """
    Object Pool con soporte para context manager.
    """
    
    def acquire_managed(self) -> PooledObject[T]:
        """
        Adquiere un objeto con gestión automática.
        
        Returns:
            PooledObject que se libera automáticamente
        """
        obj = self.acquire()
        return PooledObject(obj, self)


# Pools globales para objetos comunes

_list_pool: Optional[ObjectPoolWithManagement[list]] = None
_dict_pool: Optional[ObjectPoolWithManagement[dict]] = None
_set_pool: Optional[ObjectPoolWithManagement[set]] = None


def get_list_pool() -> ObjectPoolWithManagement[list]:
    """Retorna pool global de listas."""
    global _list_pool
    if _list_pool is None:
        _list_pool = ObjectPoolWithManagement(
            factory=list,
            reset=lambda lst: lst.clear(),
            max_size=1000
        )
    return _list_pool


def get_dict_pool() -> ObjectPoolWithManagement[dict]:
    """Retorna pool global de diccionarios."""
    global _dict_pool
    if _dict_pool is None:
        _dict_pool = ObjectPoolWithManagement(
            factory=dict,
            reset=lambda d: d.clear(),
            max_size=1000
        )
    return _dict_pool


def get_set_pool() -> ObjectPoolWithManagement[set]:
    """Retorna pool global de sets."""
    global _set_pool
    if _set_pool is None:
        _set_pool = ObjectPoolWithManagement(
            factory=set,
            reset=lambda s: s.clear(),
            max_size=1000
        )
    return _set_pool


def acquire_list() -> list:
    """Adquiere lista del pool global."""
    return get_list_pool().acquire()


def release_list(lst: list):
    """Libera lista al pool global."""
    get_list_pool().release(lst)


def acquire_dict() -> dict:
    """Adquiere diccionario del pool global."""
    return get_dict_pool().acquire()


def release_dict(d: dict):
    """Libera diccionario al pool global."""
    get_dict_pool().release(d)


def acquire_set() -> set:
    """Adquiere set del pool global."""
    return get_set_pool().acquire()


def release_set(s: set):
    """Libera set al pool global."""
    get_set_pool().release(s)


# Context managers para uso conveniente

class pooled_list:
    """Context manager para listas del pool."""
    
    def __enter__(self) -> list:
        self.lst = acquire_list()
        return self.lst
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        release_list(self.lst)


class pooled_dict:
    """Context manager para diccionarios del pool."""
    
    def __enter__(self) -> dict:
        self.d = acquire_dict()
        return self.d
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        release_dict(self.d)


class pooled_set:
    """Context manager para sets del pool."""
    
    def __enter__(self) -> set:
        self.s = acquire_set()
        return self.s
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        release_set(self.s)

