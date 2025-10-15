import abc
from typing import List, Dict, Any, Tuple, Callable, Optional

class ConstraintLevel:
    LOCAL = "LOCAL"
    PATTERN = "PATTERN"
    GLOBAL = "GLOBAL"

class Hardness:
    HARD = "hard"
    SOFT = "soft"

class ConstraintHierarchyAPI(abc.ABC):
    """Interfaz abstracta para la jerarquía de restricciones de Fibration Flow.

    Esta API define los métodos esenciales para interactuar con la jerarquía de restricciones,
    permitiendo la adición, consulta y evaluación de restricciones HARD y SOFT en diferentes niveles.
    """

    @abc.abstractmethod
    def add_constraint(self, constraint: 'Constraint') -> None:
        """Añade una restricción a la jerarquía.

        Args:
            constraint: Objeto Constraint a añadir.
        """
        pass

    @abc.abstractmethod
    def evaluate_solution(self, solution: Dict[str, Any]) -> Tuple[bool, float]:
        """Evalúa una solución propuesta contra todas las restricciones en la jerarquía.

        Retorna un booleano indicando si todas las restricciones HARD son satisfechas
        y un flotante representando la energía total (suma de violaciones de SOFT constraints).
        """
        pass

    @abc.abstractmethod
    def get_constraints_by_level(self, level: str) -> List['Constraint']:
        """Retorna todas las restricciones (HARD y SOFT) para un nivel específico."""
        pass

    @abc.abstractmethod
    def get_all_constraints(self) -> Dict[str, List['Constraint']]:
        """Retorna todas las restricciones organizadas por nivel."""
        pass

    @abc.abstractmethod
    def to_json(self) -> Dict[str, Any]:
        """Serializa la jerarquía de restricciones a un formato JSON compatible."""
        pass

    @abc.abstractmethod
    def from_json(self, json_data: Dict[str, Any]) -> None:
        """Carga la jerarquía de restricciones desde un formato JSON compatible."""
        pass

    @abc.abstractmethod
    def add_level(self, level_name: str) -> None:
        """Añade un nuevo nivel de restricción dinámicamente.
        Args:
            level_name: Nombre del nuevo nivel.
        """
        pass

    @abc.abstractmethod
    def add_local_constraint(self,
                            var1: str,
                            var2: str,
                            predicate: Callable[[Dict[str, Any]], bool],
                            weight: float = 1.0,
                            hardness: str = Hardness.HARD,
                            metadata: Optional[Dict[str, Any]] = None) -> None:
        """Añade una restricción local (binaria) a la jerarquía."""
        pass

    @abc.abstractmethod
    def add_unary_constraint(self,
                            variable: str,
                            predicate: Callable[[Dict[str, Any]], bool],
                            weight: float = 1.0,
                            hardness: str = Hardness.HARD,
                            metadata: Optional[Dict[str, Any]] = None) -> None:
        """Añade una restricción unaria a la jerarquía."""
        pass

    @abc.abstractmethod
    def add_pattern_constraint(self,
                              variables: List[str],
                              predicate: Callable[[Dict[str, Any]], bool],
                              pattern_type: str = "custom",
                              weight: float = 2.0,
                              hardness: str = Hardness.HARD,
                              metadata: Optional[Dict[str, Any]] = None) -> None:
        """Añade una restricción de patrón a la jerarquía."""
        pass

    @abc.abstractmethod
    def add_global_constraint(self,
                             variables: List[str],
                             predicate: Callable[[Dict[str, Any]], bool],
                             weight: float = 5.0,
                             hardness: str = Hardness.HARD,
                             metadata: Optional[Dict[str, Any]] = None) -> None:
        """Añade una restricción global a la jerarquía."""
        pass


# La clase Constraint se moverá al archivo constraint_hierarchy.py
# y se importará desde allí. Por ahora, se mantiene una definición mínima
# para evitar errores de referencia circular si se importa ConstraintHierarchyAPI
# en otros módulos que también necesiten Constraint.
class Constraint:
    pass

