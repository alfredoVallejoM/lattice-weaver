# lattice_weaver/formal/cubical_types.py

"""
Este módulo define las clases base genéricas para el sistema de tipos cúbicos.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Any, Optional
from functools import lru_cache

class CubicalType(ABC):
    _cached_hash: Optional[int] = None
    _cached_string: Optional[str] = None

    @abstractmethod
    def _compute_hash(self) -> int:
        pass

    def __hash__(self) -> int:
        if self._cached_hash is None:
            self._cached_hash = self._compute_hash()
        return self._cached_hash

    @abstractmethod
    def _compute_string(self) -> str:
        pass

    def to_string(self) -> str:
        if self._cached_string is None:
            self._cached_string = self._compute_string()
        return self._cached_string

    def __repr__(self) -> str:
        return self.to_string()

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, CubicalType):
            return NotImplemented
        return hash(self) == hash(other) and self.to_string() == other.to_string()

class CubicalFiniteType(CubicalType):
    def __init__(self, size: int):
        if size < 0:
            raise ValueError("El tamaño de un tipo finito no puede ser negativo.")
        self.size = size
        self._cached_hash = None  # Reset cache on init
        self._cached_string = None # Reset cache on init

    def _compute_hash(self) -> int:
        return hash(self.size)

    def _compute_string(self) -> str:
        return f"Fin({self.size})"

class CubicalSigmaType(CubicalType):
    def __init__(self, components: List[Tuple[str, CubicalType]]):
        # Asegurar orden canónico para hashing
        self.components = sorted(components, key=lambda x: x[0])
        self._cached_hash = None
        self._cached_string = None

    def _compute_hash(self) -> int:
        return hash(tuple((name, hash(comp)) for name, comp in self.components))

    def _compute_string(self) -> str:
        comp_str = ", ".join([f"{name}: {comp.to_string()}" for name, comp in self.components])
        return f"Σ({comp_str})"

class CubicalTerm(ABC):
    pass

class CubicalPredicate(CubicalType):
    def __init__(self, left: CubicalTerm, right: CubicalTerm):
        self.left = left
        self.right = right
        self._cached_hash = None
        self._cached_string = None

    def _compute_hash(self) -> int:
        return hash((hash(self.left), hash(self.right)))

    def _compute_string(self) -> str:
        return f"Path({self.left}, {self.right})"

class CubicalSubtype(CubicalType):
    def __init__(self, base_type: CubicalType, predicate: CubicalPredicate):
        self.base_type = base_type
        self.predicate = predicate
        self._cached_hash = None
        self._cached_string = None

    def _compute_hash(self) -> int:
        return hash((hash(self.base_type), hash(self.predicate)))

    def _compute_string(self) -> str:
        return f"{{ {self.base_type.to_string()} | {self.predicate.to_string()} }}"

class VariableTerm(CubicalTerm):
    _cached_hash: Optional[int] = None

    def __init__(self, name: str):
        self.name = name
        self._cached_hash = None

    def __hash__(self) -> int:
        if self._cached_hash is None:
            self._cached_hash = hash(self.name)
        return self._cached_hash

    def __repr__(self) -> str:
        return self.name

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, VariableTerm):
            return NotImplemented
        return self.name == other.name

class ValueTerm(CubicalTerm):
    _cached_hash: Optional[int] = None

    def __init__(self, value: Any):
        self.value = value
        self._cached_hash = None

    def __hash__(self) -> int:
        if self._cached_hash is None:
            self._cached_hash = hash(self.value)
        return self._cached_hash

    def __repr__(self) -> str:
        return repr(self.value)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, ValueTerm):
            return NotImplemented
        return self.value == other.value

