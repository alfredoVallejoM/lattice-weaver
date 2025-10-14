from typing import Dict, Any, List, Tuple
from abc import ABC, abstractmethod

from .constraint_hierarchy import ConstraintHierarchy

class MultiscaleCompilerAPI(ABC):
    """
    Interfaz para un compilador multiescala que optimiza problemas
    representados como ConstraintHierarchy para Fibration Flow.
    """

    @abstractmethod
    def compile_problem(self, 
                        original_hierarchy: ConstraintHierarchy,
                        original_variables_domains: Dict[str, List[Any]]) -> Tuple[ConstraintHierarchy, Dict[str, List[Any]], Dict[str, Any]]:
        """
        Toma una ConstraintHierarchy original y la transforma en una versión
        optimizada para Fibration Flow, posiblemente reduciendo la complejidad
        o reestructurando las restricciones.

        Args:
            original_hierarchy: La ConstraintHierarchy original del problema.
            original_variables_domains: Los dominios de las variables originales.

        Returns:
            Una tupla que contiene:
            - La ConstraintHierarchy optimizada.
            - Los dominios de las variables optimizadas.
            - Un diccionario de mapeo o metadatos para la reconstrucción de la solución.
        """
        pass

    @abstractmethod
    def decompile_solution(self, 
                           optimized_solution: Dict[str, Any],
                           compilation_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Toma una solución de la ConstraintHierarchy optimizada y la transforma
        de nuevo a una solución del problema original, utilizando los metadatos
        generados durante la compilación.

        Args:
            optimized_solution: La solución encontrada por Fibration Flow en el problema optimizado.
            compilation_metadata: Metadatos generados durante la compilación para la reconstrucción.

        Returns:
            La solución en el formato del problema original.
        """
        pass

