# lattice_weaver/formal/cubical_types.py

"""
Este módulo define las clases base genéricas para el sistema de tipos cúbicos.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Any

class CubicalType(ABC):
    @abstractmethod
    def to_string(self) -> str:
        pass

    def __repr__(self) -> str:
        return self.to_string()

class CubicalFiniteType(CubicalType):
    def __init__(self, size: int):
        if size < 0:
            raise ValueError("El tamaño de un tipo finito no puede ser negativo.")
        self.size = size

    def to_string(self) -> str:
        return f"Fin({self.size})"

class CubicalSigmaType(CubicalType):
    def __init__(self, components: List[Tuple[str, CubicalType]]):
        self.components = components

    def to_string(self) -> str:
        comp_str = ", ".join([f"{name}: {comp.to_string()}" for name, comp in self.components])
        return f"Σ({comp_str})"

class CubicalTerm(ABC):
    pass

class CubicalPredicate(CubicalType):
    def __init__(self, left: CubicalTerm, right: CubicalTerm):
        self.left = left
        self.right = right

    def to_string(self) -> str:
        return f"Path({self.left}, {self.right})"

class CubicalSubtype(CubicalType):
    def __init__(self, base_type: CubicalType, predicate: CubicalPredicate):
        self.base_type = base_type
        self.predicate = predicate

    def to_string(self) -> str:
        return f"{{ {self.base_type.to_string()} | {self.predicate.to_string()} }}"

class VariableTerm(CubicalTerm):
    def __init__(self, name: str):
        self.name = name

    def __repr__(self) -> str:
        return self.name

class ValueTerm(CubicalTerm):
    def __init__(self, value: Any):
        self.value = value

    def __repr__(self) -> str:
        return repr(self.value)

