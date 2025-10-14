from typing import Callable, Any, NamedTuple, Dict
from dataclasses import field

# Registro global de funciones de relación serializables
RELATION_REGISTRY: Dict[str, Callable[[Any, Any, Dict[str, Any]], bool]] = {}

def register_relation(name: str, func: Callable[[Any, Any, Dict[str, Any]], bool]):
    """Registra una función de relación para hacerla serializable."""
    if name in RELATION_REGISTRY:
        raise ValueError(f"La relación \'{name}\' ya está registrada.")
    RELATION_REGISTRY[name] = func

class Constraint(NamedTuple):
    """
    Representa una restricción binaria entre dos variables.
    La relación ahora es una cadena que se busca en RELATION_REGISTRY.
    """
    var1: str
    var2: str
    relation_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)

def nqueens_not_equal(val1: Any, val2: Any, metadata: Dict[str, Any]) -> bool:
    """Relación para la restricción de no ser igual."""
    return val1 != val2

def nqueens_not_diagonal(val1: Any, val2: Any, metadata: Dict[str, Any]) -> bool:
    """Relación para la restricción de no estar en la misma diagonal en el problema de N-Reinas."""
    i = metadata["var1_idx"]
    j = metadata["var2_idx"]
    return abs(val1 - val2) != abs(i - j)

register_relation("nqueens_not_equal", nqueens_not_equal)
register_relation("nqueens_not_diagonal", nqueens_not_diagonal)

def get_relation(name: str) -> Callable[[Any, Any, Dict[str, Any]], bool]:
    """
    Obtiene una función de relación a partir de su nombre registrado.
    La función devuelta acepta val1, val2, metadata.
    """
    if name not in RELATION_REGISTRY:
        raise ValueError(f"Relación \'{name}\' no encontrada en el registro.")
    return RELATION_REGISTRY[name]



def not_equal(val1: Any, val2: Any, metadata: Dict[str, Any]) -> bool:
    return val1 != val2

register_relation("NE", not_equal)

