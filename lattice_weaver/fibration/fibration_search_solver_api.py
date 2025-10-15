from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple

class FibrationSearchSolverAPI(ABC):
    """
    Interfaz abstracta para el FibrationSearchSolver.
    Define los métodos esenciales que cualquier implementación del solver debe proporcionar.
    """

    @abstractmethod
    def solve(self, time_limit_seconds: int = 60) -> Optional[Dict[str, Any]]:
        """
        Inicia el proceso de búsqueda para encontrar la mejor solución.

        Args:
            time_limit_seconds (int): Límite de tiempo en segundos para la búsqueda.

        Returns:
            Optional[Dict[str, Any]]: La mejor solución encontrada como un diccionario de asignaciones
                                      de variables, o None si no se encuentra ninguna solución.
        """
        pass

    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """
        Devuelve estadísticas sobre el proceso de búsqueda.

        Returns:
            Dict[str, Any]: Un diccionario que contiene varias métricas de rendimiento y estado del solver.
        """
        pass

