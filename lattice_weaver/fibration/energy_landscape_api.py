import abc
from typing import Dict, Any, Tuple, List

class EnergyLandscapeAPI(abc.ABC):
    """Interfaz abstracta para el paisaje de energía de Fibration Flow.

    Esta API define los métodos esenciales para interactuar con el paisaje de energía,
    permitiendo el cálculo de la energía de una solución y su gradiente.
    """

    @abc.abstractmethod
    def compute_energy(self, assignment: Dict[str, Any], use_cache: bool = True) -> Tuple[bool, float]:
        """Calcula la energía de una asignación parcial.

        Args:
            assignment: Diccionario {variable: valor}
            use_cache: Si usar caché de energías (por defecto True)

        Returns:
            Tupla (all_hard_satisfied, total_energy):
                - all_hard_satisfied: True si todas las restricciones HARD son satisfechas
                - total_energy: Energía total (suma ponderada de violaciones de SOFT constraints)
        """
        pass

    @abc.abstractmethod
    def compute_energy_incremental(self,
                                   base_assignment: Dict[str, Any],
                                   base_energy: Tuple[bool, float],
                                   new_var: str,
                                   new_value: Any) -> Tuple[bool, float]:
        """Calcula energía de forma incremental.

        Args:
            base_assignment: Asignación base
            base_energy: Energía de la asignación base (all_hard_satisfied, total_energy)
            new_var: Variable a añadir
            new_value: Valor de la nueva variable

        Returns:
            Nueva energía (all_hard_satisfied, total_energy)
        """
        pass

    @abc.abstractmethod
    def compute_energy_gradient_optimized(self,
                                         assignment: Dict[str, Any],
                                         base_energy: Tuple[bool, float],
                                         variable: str,
                                         domain: List[Any]) -> Dict[Any, float]:
        """Calcula el gradiente de energía de forma optimizada.

        Args:
            assignment: Asignación parcial actual
            base_energy: Energía de la asignación base
            variable: Variable para la cual calcular el gradiente
            domain: Dominio de valores posibles para la variable

        Returns:
            Diccionario {valor: energía}
        """
        pass

    @abc.abstractmethod
    def get_cache_statistics(self) -> Dict[str, int]:
        """Obtiene estadísticas de uso de caché."""
        pass

    @abc.abstractmethod
    def clear_cache(self):
        """Limpia el caché de energía."""
        pass

    @abc.abstractmethod
    def to_json(self) -> Dict[str, Any]:
        """Serializa el paisaje de energía a un formato JSON compatible."""
        pass

    @abc.abstractmethod
    def from_json(self, json_data: Dict[str, Any]) -> None:
        """Carga el paisaje de energía desde un formato JSON compatible."""
        pass

