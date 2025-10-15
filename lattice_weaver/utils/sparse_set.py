"""
Sparse Set - Estructura de Datos Eficiente para Dominios

Implementación de Sparse Set para representar dominios de variables en CSP.

Ventajas sobre listas:
- O(1) para añadir/eliminar valores
- O(1) para iterar sobre valores activos
- O(1) para verificar si un valor está presente
- O(1) para save/restore (trailing)

Complejidad espacial: O(n) donde n es el tamaño del universo

Autor: Agente Autónomo - Lattice Weaver
Fecha: 15 de Octubre, 2025
"""

from typing import List, Any, Iterator, Optional
from dataclasses import dataclass


@dataclass
class SparseSetSnapshot:
    """Snapshot de un Sparse Set para backtracking."""
    size: int  # Tamaño del conjunto en el momento del snapshot


class SparseSet:
    """
    Sparse Set para representación eficiente de dominios.
    
    Mantiene dos arrays:
    - dense: Valores activos (compacto)
    - sparse: Índices en dense (disperso)
    
    Permite operaciones O(1) para todas las operaciones críticas.
    """
    
    def __init__(self, universe: List[Any]):
        """
        Inicializa Sparse Set con un universo de valores.
        
        Args:
            universe: Lista de todos los valores posibles
        """
        self.universe = universe
        self.n = len(universe)
        
        # Mapeo valor -> índice en universe
        self.value_to_idx = {val: idx for idx, val in enumerate(universe)}
        
        # Dense array: valores activos (compacto)
        self.dense = list(range(self.n))
        
        # Sparse array: índices en dense (disperso)
        self.sparse = list(range(self.n))
        
        # Tamaño actual (número de valores activos)
        self.size = self.n
    
    def __len__(self) -> int:
        """Retorna número de valores activos."""
        return self.size
    
    def __contains__(self, value: Any) -> bool:
        """
        Verifica si un valor está en el conjunto.
        
        Complejidad: O(1)
        
        Args:
            value: Valor a verificar
        
        Returns:
            True si el valor está activo
        """
        if value not in self.value_to_idx:
            return False
        
        idx = self.value_to_idx[value]
        return self.sparse[idx] < self.size
    
    def __iter__(self) -> Iterator[Any]:
        """
        Itera sobre valores activos.
        
        Complejidad: O(k) donde k = número de valores activos
        
        Yields:
            Valores activos en orden de inserción
        """
        for i in range(self.size):
            yield self.universe[self.dense[i]]
    
    def __repr__(self) -> str:
        """Representación string."""
        values = list(self)
        return f"SparseSet({values})"
    
    def add(self, value: Any) -> bool:
        """
        Añade un valor al conjunto.
        
        Complejidad: O(1)
        
        Args:
            value: Valor a añadir
        
        Returns:
            True si el valor fue añadido, False si ya estaba
        """
        if value not in self.value_to_idx:
            return False
        
        idx = self.value_to_idx[value]
        
        # Ya está activo
        if self.sparse[idx] < self.size:
            return False
        
        # Intercambiar con el primer inactivo
        if self.sparse[idx] >= self.size:
            # Intercambiar en dense
            first_inactive_pos = self.size
            current_pos = self.sparse[idx]
            
            if current_pos != first_inactive_pos:
                # Intercambiar
                self.dense[current_pos], self.dense[first_inactive_pos] = \
                    self.dense[first_inactive_pos], self.dense[current_pos]
                
                # Actualizar sparse
                self.sparse[self.dense[current_pos]] = current_pos
                self.sparse[self.dense[first_inactive_pos]] = first_inactive_pos
        
        self.size += 1
        return True
    
    def remove(self, value: Any) -> bool:
        """
        Elimina un valor del conjunto.
        
        Complejidad: O(1)
        
        Args:
            value: Valor a eliminar
        
        Returns:
            True si el valor fue eliminado, False si no estaba
        """
        if value not in self.value_to_idx:
            return False
        
        idx = self.value_to_idx[value]
        
        # No está activo
        if self.sparse[idx] >= self.size:
            return False
        
        # Intercambiar con el último activo
        last_active_pos = self.size - 1
        current_pos = self.sparse[idx]
        
        if current_pos != last_active_pos:
            # Intercambiar en dense
            self.dense[current_pos], self.dense[last_active_pos] = \
                self.dense[last_active_pos], self.dense[current_pos]
            
            # Actualizar sparse
            self.sparse[self.dense[current_pos]] = current_pos
            self.sparse[self.dense[last_active_pos]] = last_active_pos
        
        self.size -= 1
        return True
    
    def clear(self):
        """
        Elimina todos los valores.
        
        Complejidad: O(1)
        """
        self.size = 0
    
    def reset(self):
        """
        Restaura todos los valores.
        
        Complejidad: O(1)
        """
        self.size = self.n
    
    def to_list(self) -> List[Any]:
        """
        Convierte a lista de valores activos.
        
        Complejidad: O(k) donde k = número de valores activos
        
        Returns:
            Lista de valores activos
        """
        return [self.universe[self.dense[i]] for i in range(self.size)]
    
    def snapshot(self) -> SparseSetSnapshot:
        """
        Crea un snapshot para backtracking.
        
        Complejidad: O(1)
        
        Returns:
            Snapshot del estado actual
        """
        return SparseSetSnapshot(size=self.size)
    
    def restore(self, snapshot: SparseSetSnapshot):
        """
        Restaura desde un snapshot.
        
        Complejidad: O(1)
        
        Args:
            snapshot: Snapshot a restaurar
        """
        self.size = snapshot.size
    
    def copy(self) -> 'SparseSet':
        """
        Crea una copia del Sparse Set.
        
        Complejidad: O(n)
        
        Returns:
            Copia del conjunto
        """
        new_set = SparseSet(self.universe)
        new_set.dense = self.dense.copy()
        new_set.sparse = self.sparse.copy()
        new_set.size = self.size
        return new_set


def create_sparse_set(values: List[Any]) -> SparseSet:
    """
    Factory function para crear Sparse Set.
    
    Args:
        values: Lista de valores iniciales (universo)
    
    Returns:
        Sparse Set inicializado
    """
    return SparseSet(values)


def sparse_set_from_list(universe: List[Any], active: List[Any]) -> SparseSet:
    """
    Crea Sparse Set con valores activos específicos.
    
    Args:
        universe: Universo de valores posibles
        active: Valores activos inicialmente
    
    Returns:
        Sparse Set con valores activos especificados
    """
    sparse_set = SparseSet(universe)
    
    # Eliminar todos
    sparse_set.clear()
    
    # Añadir solo los activos
    for value in active:
        sparse_set.add(value)
    
    return sparse_set

