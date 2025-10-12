from typing import Callable, Any, NamedTuple, Dict

# Registro global de funciones de relación serializables
RELATION_REGISTRY: Dict[str, Callable[[Any, Any], bool]] = {}

def register_relation(name: str, func: Callable[[Any, Any], bool]):
    """Registra una función de relación para hacerla serializable."""
    if name in RELATION_REGISTRY:
        raise ValueError(f"La relación '{name}' ya está registrada.")
    RELATION_REGISTRY[name] = func

class Constraint(NamedTuple):
    """
    Representa una restricción binaria entre dos variables.
    La relación ahora es una cadena que se busca en RELATION_REGISTRY.
    """
    var1: str
    var2: str
    relation_name: str

    def get_relation(self) -> Callable[[Any, Any], bool]:
        """Obtiene la función de relación a partir de su nombre registrado."""
        if self.relation_name not in RELATION_REGISTRY:
            raise ValueError(f"Relación '{self.relation_name}' no encontrada en el registro.")
        return RELATION_REGISTRY[self.relation_name]



def nqueens_not_equal(val1, val2):
    return val1 != val2

def nqueens_not_diagonal(val1, val2, i, j):
    # Esta función requiere los índices i y j, lo cual es un desafío para la serialización
    # si se pasa directamente. La Constraint debería almacenar i y j.
    # Por ahora, se asume que los índices se manejan externamente o que la Constraint
    # encapsula esta lógica para que la relación solo reciba val1, val2.
    # Para este contexto, la dejaremos como un placeholder.
    return abs(val1 - val2) != abs(i - j)

register_relation("nqueens_not_equal", nqueens_not_equal)
# register_relation("nqueens_not_diagonal", nqueens_not_diagonal) # No registrar directamente así por el i,j
