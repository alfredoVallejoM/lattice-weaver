# lattice_weaver/arc_engine/domains.py

from abc import ABC, abstractmethod
from typing import Iterable, Any, Set, Optional
# from bitarray import bitarray

class Domain(ABC):
    """Abstract base class for a variable's domain representation."""

    @abstractmethod
    def __contains__(self, value: Any) -> bool:
        pass

    @abstractmethod
    def remove(self, value: Any):
        pass

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def get_values(self) -> Iterable[Any]:
        """Return an iterable of the current values in the domain."""
        pass

class SetDomain(Domain):
    """Domain represented as a set. Good for non-numeric or small domains."""
    def __init__(self, values: Iterable[Any]):
        self._values = set(values)

    def __contains__(self, value: Any) -> bool:
        return value in self._values

    def remove(self, value: Any):
        self._values.discard(value)

    def __iter__(self):
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def add(self, value: Any):
        """Adds a value to the domain."""
        self._values.add(value)

    def get_values(self) -> Iterable[Any]:
        return self._values

    def intersect(self, other_values: Iterable[Any]):
        """Intersect the current domain with another set of values."""
        self._values.intersection_update(other_values)


class BitsetDomain(Domain):
    """Domain represented as a bitset. Optimal for dense integer domains."""
    def __init__(self, min_val: int, max_val: int, initial_values: Optional[Set[int]] = None):
        self.min_val = min_val
        self.max_val = max_val
        self.size = max_val - min_val + 1
        self.bits = bitarray(self.size)
        
        if initial_values is None:
            self.bits.setall(1)
        else:
            self.bits.setall(0)
            for v in initial_values:
                if min_val <= v <= max_val:
                    self.bits[v - min_val] = 1

    def __contains__(self, value: Any) -> bool:
        if not isinstance(value, int) or not (self.min_val <= value <= self.max_val):
            return False
        return self.bits[value - self.min_val]

    def remove(self, value: Any):
        if isinstance(value, int) and self.min_val <= value <= self.max_val:
            self.bits[value - self.min_val] = 0

    def __iter__(self):
        return (i + self.min_val for i, bit in enumerate(self.bits) if bit)

    def __len__(self) -> int:
        return self.bits.count()

    def get_values(self) -> Iterable[Any]:
        return self

class SparseSetDomain(Domain):
    """Domain represented as a sparse set. Optimal for sparse integer domains."""
    def __init__(self, max_val: int, initial_values: Iterable[int]):
        self.dense = list(initial_values)
        self.sparse = [-1] * (max_val + 1)
        self.n = len(self.dense)
        for i, v in enumerate(self.dense):
            self.sparse[v] = i

    def __contains__(self, value: Any) -> bool:
        if not isinstance(value, int) or not (0 <= value < len(self.sparse)):
            return False
        idx = self.sparse[value]
        return 0 <= idx < self.n and self.dense[idx] == value

    def remove(self, value: Any):
        if self.__contains__(value):
            idx = self.sparse[value]
            last_val = self.dense[self.n - 1]
            self.dense[idx] = last_val
            self.sparse[last_val] = idx
            self.n -= 1

    def __iter__(self):
        return (self.dense[i] for i in range(self.n))

    def __len__(self) -> int:
        return self.n

    def get_values(self) -> Iterable[Any]:
        return self

def create_optimal_domain(values: Iterable[Any]) -> Domain:
    # Simplified: always use SetDomain (bitarray not available)
    return SetDomain(set(values))

def create_optimal_domain_OLD(values: Iterable[Any]) -> Domain:
    """
    Factory function that selects the best domain representation based on the
    characteristics of the initial values.
    """
    value_list = list(values)
    if not value_list:
        return SetDomain([])

    if not all(isinstance(v, int) and v >= 0 for v in value_list):
        return SetDomain(value_list)

    min_val, max_val = min(value_list), max(value_list)
    range_size = max_val - min_val + 1
    density = len(value_list) / range_size if range_size > 0 else 0

    # Heuristic threshold for density
    if density > 0.5:
        return BitsetDomain(min_val, max_val, set(value_list))
    else:
        return SparseSetDomain(max_val, value_list)

