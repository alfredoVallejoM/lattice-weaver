"""
StateManager: Gestión eficiente de estados con representación canónica.

Este módulo proporciona una gestión optimizada de estados para el HomotopyAnalyzer,
usando IDs numéricos y representaciones canónicas para reducir el uso de memoria
y acelerar las operaciones de hashing y comparación.

Autor: Manus AI
Fecha: 11 de Octubre de 2025
"""

from typing import Set, Tuple, FrozenSet, Any, Dict, Optional
from dataclasses import dataclass
import hashlib


# Tipo para restricciones
Constraint = Tuple[str, ...]


@dataclass(frozen=True)
class CanonicalState:
    """
    Representación canónica de un estado.
    
    Un estado es un conjunto de restricciones. La representación canónica
    garantiza que estados equivalentes tengan la misma representación,
    independientemente del orden en que se construyeron.
    
    Attributes:
        canonical_repr: Tupla ordenada de restricciones
        _hash: Hash precalculado para operaciones rápidas
    """
    canonical_repr: Tuple[Constraint, ...]
    _hash: int
    
    def __init__(self, constraints: Set[Constraint]):
        """
        Crea una representación canónica del estado.
        
        Args:
            constraints: Conjunto de restricciones del estado
        """
        # Ordenar restricciones por una clave determinística
        sorted_constraints = sorted(constraints, key=self._constraint_key)
        
        # Usar object.__setattr__ porque la clase es frozen
        object.__setattr__(self, 'canonical_repr', tuple(sorted_constraints))
        object.__setattr__(self, '_hash', hash(self.canonical_repr))
    
    @staticmethod
    def _constraint_key(c: Constraint) -> tuple:
        """
        Genera una clave de ordenamiento para una restricción.
        
        Args:
            c: Restricción a ordenar
            
        Returns:
            Tupla que sirve como clave de ordenamiento
        """
        if len(c) == 0:
            return (0,)
        
        # Clave: (tipo, variables ordenadas, valores ordenados)
        constraint_type = c[0]
        
        if constraint_type == 'var':
            # ('var', 'A', 'in', frozenset({1, 2, 3}))
            var_name = c[1]
            operation = c[2]
            values = tuple(sorted(c[3])) if isinstance(c[3], (set, frozenset)) else c[3]
            return (constraint_type, var_name, operation, values)
        
        elif constraint_type == 'constraint':
            # ('constraint', 'A', 'B', 'neq')
            vars_sorted = tuple(sorted(c[1:3]))
            relation = c[3] if len(c) > 3 else ''
            return (constraint_type, vars_sorted, relation)
        
        else:
            # Caso genérico
            return (constraint_type,) + tuple(sorted(str(x) for x in c[1:]))
    
    def __hash__(self) -> int:
        """Retorna el hash precalculado."""
        return self._hash
    
    def __eq__(self, other) -> bool:
        """Compara dos estados canónicos."""
        if not isinstance(other, CanonicalState):
            return False
        return self.canonical_repr == other.canonical_repr
    
    @property
    def constraints(self) -> Set[Constraint]:
        """Retorna el conjunto de restricciones."""
        return set(self.canonical_repr)
    
    def to_dict(self) -> dict:
        """Serializa el estado a un diccionario."""
        return {
            'constraints': [list(c) for c in self.canonical_repr],
            'hash': self._hash
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'CanonicalState':
        """Deserializa un estado desde un diccionario."""
        constraints = {tuple(c) for c in data['constraints']}
        return cls(constraints)


class StateManager:
    """
    Gestor de estados con representación canónica e IDs numéricos.
    
    Este gestor mantiene un registro de todos los estados únicos del problema,
    asignándoles IDs numéricos para operaciones eficientes. Estados equivalentes
    (mismo conjunto de restricciones) reciben el mismo ID.
    
    Attributes:
        states: Mapeo de ID -> CanonicalState
        state_to_id: Mapeo de representación canónica -> ID
        next_id: Próximo ID disponible
    """
    
    def __init__(self):
        """Inicializa el gestor de estados."""
        self.states: Dict[int, CanonicalState] = {}
        self.state_to_id: Dict[Tuple[Constraint, ...], int] = {}
        self.next_id: int = 0
        
        # Estadísticas
        self.total_registrations = 0
        self.cache_hits = 0
    
    def register_state(self, constraints: Set[Constraint]) -> int:
        """
        Registra un estado y retorna su ID.
        
        Si el estado ya existe, retorna el ID existente (cache hit).
        Si es nuevo, crea un nuevo ID y lo registra.
        
        Args:
            constraints: Conjunto de restricciones del estado
            
        Returns:
            ID numérico del estado
        """
        self.total_registrations += 1
        
        # Crear representación canónica
        canonical = CanonicalState(constraints)
        
        # Verificar si ya existe
        if canonical.canonical_repr in self.state_to_id:
            self.cache_hits += 1
            return self.state_to_id[canonical.canonical_repr]
        
        # Nuevo estado: asignar ID
        state_id = self.next_id
        self.next_id += 1
        
        # Registrar
        self.states[state_id] = canonical
        self.state_to_id[canonical.canonical_repr] = state_id
        
        return state_id
    
    def get_state(self, state_id: int) -> CanonicalState:
        """
        Obtiene el estado correspondiente a un ID.
        
        Args:
            state_id: ID del estado
            
        Returns:
            CanonicalState correspondiente
            
        Raises:
            KeyError: Si el ID no existe
        """
        return self.states[state_id]
    
    def exists(self, state_id: int) -> bool:
        """
        Verifica si un ID de estado existe.
        
        Args:
            state_id: ID a verificar
            
        Returns:
            True si existe, False en caso contrario
        """
        return state_id in self.states
    
    def get_statistics(self) -> dict:
        """
        Retorna estadísticas del gestor.
        
        Returns:
            Diccionario con estadísticas
        """
        cache_hit_rate = (self.cache_hits / self.total_registrations * 100 
                         if self.total_registrations > 0 else 0)
        
        return {
            'total_states': len(self.states),
            'total_registrations': self.total_registrations,
            'cache_hits': self.cache_hits,
            'cache_hit_rate': f"{cache_hit_rate:.2f}%",
            'memory_estimate_mb': self._estimate_memory()
        }
    
    def _estimate_memory(self) -> float:
        """
        Estima el uso de memoria del gestor.
        
        Returns:
            Memoria estimada en MB
        """
        # Estimación aproximada:
        # - Cada estado: ~100 bytes (promedio)
        # - Cada entrada en state_to_id: ~150 bytes
        # - Overhead de Python: ~50%
        
        bytes_per_state = 250  # Promedio conservador
        total_bytes = len(self.states) * bytes_per_state
        return total_bytes / (1024 * 1024)
    
    def clear(self):
        """Limpia todos los estados registrados."""
        self.states.clear()
        self.state_to_id.clear()
        self.next_id = 0
        self.total_registrations = 0
        self.cache_hits = 0
    
    def export_states(self) -> dict:
        """
        Exporta todos los estados a un diccionario serializable.
        
        Returns:
            Diccionario con todos los estados
        """
        return {
            'states': {
                state_id: state.to_dict() 
                for state_id, state in self.states.items()
            },
            'statistics': self.get_statistics()
        }
    
    def import_states(self, data: dict):
        """
        Importa estados desde un diccionario.
        
        Args:
            data: Diccionario con estados exportados
        """
        self.clear()
        
        for state_id_str, state_data in data['states'].items():
            state_id = int(state_id_str)
            state = CanonicalState.from_dict(state_data)
            
            self.states[state_id] = state
            self.state_to_id[state.canonical_repr] = state_id
            
            if state_id >= self.next_id:
                self.next_id = state_id + 1


# Constantes especiales
BOTTOM_STATE_ID = -1  # ID para el estado bottom (inconsistente)
TOP_STATE_ID = 0      # ID para el estado top (vacío)

