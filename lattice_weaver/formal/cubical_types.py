# lattice_weaver/formal/cubical_types.py

"""
Este módulo define las clases base genéricas para el sistema de tipos cúbicos.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Any, Optional, FrozenSet, Union
from functools import lru_cache
from dataclasses import dataclass, field


@dataclass(frozen=True)
class CubicalType(ABC):
    """Clase base abstracta para cualquier tipo en el sistema cúbico.

    Implementa caching para hash y representación en string para mejorar la eficiencia.
    """
    _cached_hash: Optional[int] = field(init=False, repr=False, default=None)
    _cached_string: Optional[str] = field(init=False, repr=False, default=None)

    def __post_init__(self):
        # No need to explicitly set to None here for frozen dataclasses;
        # the default=None in field() handles it.
        pass

    @abstractmethod
    def _compute_hash(self) -> int:
        pass

    def __hash__(self) -> int:
        if self._cached_hash is None:
            object.__setattr__(self, 
                '_cached_hash', self._compute_hash())
        return self._cached_hash

    @abstractmethod
    def _compute_string(self) -> str:
        pass

    def to_string(self) -> str:
        if self._cached_string is None:
            object.__setattr__(self, 
                '_cached_string', self._compute_string())
        return self._cached_string

    def __repr__(self) -> str:
        return self.to_string()

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, CubicalType):
            return NotImplemented
        # Comparar por hash y string para una igualdad profunda y eficiente
        return hash(self) == hash(other) and self.to_string() == other.to_string()


@dataclass(frozen=True)
class CubicalFiniteType(CubicalType):
    """Representa un tipo finito, equivalente a `Fin(n)` en HoTT.

    Usado para representar los dominios de las variables del CSP.

    Attributes:
        size: El número de elementos en el tipo.
    """
    size: int

    def __post_init__(self):
        super().__post_init__()
        if self.size < 0:
            raise ValueError("El tamaño de un tipo finito no puede ser negativo.")

    def _compute_hash(self) -> int:
        return hash(self.size)

    def _compute_string(self) -> str:
        return f"Fin({self.size})"


@dataclass(frozen=True)
class CubicalSigmaType(CubicalType):
    """Representa un tipo producto dependiente (Sigma), análogo a una tupla con nombre.

    Usado para representar el espacio de búsqueda completo del CSP (el producto
    cartesiano de todos los dominios).

    Attributes:
        components: Una lista de tuplas (nombre, tipo) que definen el producto.
    """
    components: List[Tuple[str, CubicalType]]

    def __post_init__(self):
        super().__post_init__()
        # Asegurar orden canónico para hashing consistente
        # Ensure canonical order for hashing consistent
        canonical_components = tuple(sorted(self.components, key=lambda x: x[0]))
        object.__setattr__(self, 'components', canonical_components)

    def _compute_hash(self) -> int:
        return hash(self.components)

    def _compute_string(self) -> str:
        comp_str = ", ".join([f"{name}: {comp.to_string()}" for name, comp in self.components])
        return f"Σ({comp_str})"


class CubicalTerm(ABC):
    """Clase base para un término o habitante de un tipo cúbico."""
    pass


@dataclass(frozen=True)
class VariableTerm(CubicalTerm):
    """Un término que representa una variable en el contexto cúbico."""
    name: str

    def __repr__(self) -> str:
        return self.name

    def __hash__(self) -> int:
        return hash(self.name)




@dataclass(frozen=True)
class ValueTerm(CubicalTerm):
    """Un término que representa un valor concreto en el contexto cúbico."""
    value: Any

    def __repr__(self) -> str:
        return repr(self.value)

    def __hash__(self) -> int:
        return hash(self.value)




@dataclass(frozen=True)
class CubicalPredicate(CubicalType):
    """Representa un predicado sobre un tipo, como una igualdad o una restricción.

    En HoTT, esto es en sí mismo un tipo. Si el tipo está habitado, el predicado
    se considera verdadero.
    """
    def __post_init__(self):
        super().__post_init__()


@dataclass(frozen=True)
class CubicalPath(CubicalPredicate):
    """Representa un predicado de igualdad (Path) entre dos términos."""
    left: CubicalTerm
    right: CubicalTerm

    def _compute_hash(self) -> int:
        return hash((hash(self.left), hash(self.right)))

    def _compute_string(self) -> str:
        return f"Path({self.left}, {self.right})"





@dataclass(frozen=True)
class CubicalNegation(CubicalPredicate):
    """
    Representa la negación de un predicado cúbico.
    """
    predicate: CubicalPredicate

    def _compute_hash(self) -> int:
        return hash(("Negation", hash(self.predicate)))

    def _compute_string(self) -> str:
        return f"¬({self.predicate.to_string()})"





@dataclass(frozen=True)
class CubicalAnd(CubicalPredicate):
    """
    Representa la conjunción (AND) de múltiples predicados cúbicos.
    """
    predicates: FrozenSet[CubicalPredicate]

    def __post_init__(self):
        super().__post_init__()
        # Ensure canonical order for hashing consistent
        # Convert to tuple of sorted predicates for consistent hashing for stable frozenset creation
        # Ensure canonical order for hashing consistent
        # Convert to tuple of sorted predicates for consistent hashing for stable frozenset creation
        # Sort predicates by their canonical string representation to ensure consistent ordering
        canonical_predicates_list = sorted(list(self.predicates), key=lambda p: hash(p))
        # Store as a frozenset for immutability and efficient lookup, ensuring canonical order
        object.__setattr__(self, 'predicates', frozenset(canonical_predicates_list))

    def _compute_hash(self) -> int:
        # Hash the frozenset directly, as its hash is stable if its elements are hashable and canonical
        # Hash the frozenset directly, as its hash is stable if its elements are hashable and canonical
        # The frozenset itself is hashable and its hash depends on the hashes of its elements.
        # Since elements are canonicalized in __post_init__, this hash should be stable.
        return hash((self.predicates, "CubicalAnd"))

    def _compute_string(self) -> str:
        return "(" + " & ".join(p.to_string() for p in self.predicates) + ")"


@dataclass(frozen=True)
class CubicalArithmetic(CubicalTerm):
    """
    Representa una expresión aritmética en el sistema cúbico.
    Actualmente soporta solo la suma.
    """
    operation: str  # e.g., "sum"
    terms: Tuple[CubicalTerm, ...]

    def _compute_hash(self) -> int:
        return hash((self.operation, frozenset(hash(t) for t in self.terms)))

    def _compute_string(self) -> str:
        sorted_terms = sorted(self.terms, key=lambda t: t.to_string())
        if self.operation == "sum":
            return f"({' + '.join(t.to_string() for t in sorted_terms)})""
        return f"{self.operation}({\', \'.join(t.to_string() for t in sorted_terms)})"

    def __post_init__(self):
        # Asegurar que los términos se ordenen canónicamente para la creación del frozenset
        canonical_terms_list = sorted(self.terms, key=lambda t: hash(t))
        object.__setattr__(self, 'terms', tuple(canonical_terms_list))


@dataclass(frozen=True)
class CubicalComparison(CubicalPredicate):
    """
    Representa una comparación en el sistema cúbico (e.g., left == right).
    """
    left: CubicalTerm
    operator: str  # e.g., "==", "<=", ">="
    right: CubicalTerm

    def _compute_hash(self) -> int:
        return hash((hash(self.left), self.operator, hash(self.right)))

    def _compute_string(self) -> str:
        return f"({self.left.to_string()} {self.operator} {self.right.to_string()})"


@dataclass(frozen=True)
class CubicalSubtype(CubicalType):
    """Representa un subtipo, definido por un tipo base y un predicado."""
    base_type: CubicalType
    predicate: CubicalPredicate

    def __post_init__(self):
        super().__post_init__()

    def _compute_hash(self) -> int:
        return hash((hash(self.base_type), hash(self.predicate)))

    def _compute_string(self) -> str:
        return f"{{ {self.base_type.to_string()} | {self.predicate.to_string()} }}"

