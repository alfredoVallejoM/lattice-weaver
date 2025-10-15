"""
Interfaces base para estrategias de CSPSolver.

Define las clases abstractas que todas las estrategias deben implementar.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from ...csp_problem import CSP


class VariableSelector(ABC):
    """
    Interfaz abstracta para estrategias de selección de variables.
    
    Una estrategia de selección de variables decide qué variable no asignada
    debe ser asignada a continuación durante la búsqueda de backtracking.
    
    Ejemplos de estrategias:
    - FirstUnassigned: Selecciona la primera variable no asignada
    - MRV: Selecciona la variable con menor dominio restante
    - Degree: Selecciona la variable con más restricciones
    - MRVDegree: Combina MRV y Degree como desempate
    """
    
    @abstractmethod
    def select(self, 
               csp: CSP,
               assignment: Dict[str, Any],
               current_domains: Dict[str, List[Any]]) -> Optional[str]:
        """
        Selecciona la siguiente variable a asignar.
        
        Args:
            csp: El problema CSP
            assignment: Asignación parcial actual (variable -> valor)
            current_domains: Dominios actuales de todas las variables
        
        Returns:
            Nombre de la variable a asignar, o None si todas están asignadas
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class ValueOrderer(ABC):
    """
    Interfaz abstracta para estrategias de ordenamiento de valores.
    
    Una estrategia de ordenamiento de valores decide en qué orden probar
    los valores del dominio de una variable durante la búsqueda.
    
    Ejemplos de estrategias:
    - NaturalOrder: Mantiene el orden original del dominio
    - LCV: Ordena valores menos restrictivos primero
    - Random: Ordena valores aleatoriamente
    """
    
    @abstractmethod
    def order(self,
              var: str,
              csp: CSP,
              assignment: Dict[str, Any],
              current_domains: Dict[str, List[Any]]) -> List[Any]:
        """
        Ordena los valores del dominio de una variable.
        
        Args:
            var: Variable cuyo dominio ordenar
            csp: El problema CSP
            assignment: Asignación parcial actual (variable -> valor)
            current_domains: Dominios actuales de todas las variables
        
        Returns:
            Lista de valores ordenados según la estrategia
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

