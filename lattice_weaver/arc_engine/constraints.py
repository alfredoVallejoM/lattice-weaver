# lattice_weaver/arc_engine/constraints.py

from typing import Callable, Any, NamedTuple

class Constraint(NamedTuple):
    """
    Represents a binary constraint between two variables.
    """
    var1: str
    var2: str
    relation: Callable[[Any, Any], bool]

