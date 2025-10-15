"""
Interfaces base para estrategias de búsqueda CSP

Este módulo define las interfaces abstractas para estrategias de selección
de variables y ordenamiento de valores.

Autor: Manus AI
Fecha: 15 de Octubre, 2025
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from ...csp_problem import CSP


class VariableSelector(ABC):
    """
    Interfaz abstracta para estrategias de selección de variables.
    
    Una estrategia de selección de variables decide qué variable no asignada
    debe ser asignada a continuación durante la búsqueda.
    """
    
    @abstractmethod
    def select(self, 
               csp: CSP, 
               assignment: Dict[str, Any], 
               current_domains: Dict[str, List[Any]]) -> Optional[str]:
        """
        Selecciona la siguiente variable a asignar.
        
        Args:
            csp: Problema CSP
            assignment: Asignación actual (variables ya asignadas)
            current_domains: Dominios actuales de todas las variables
        
        Returns:
            Nombre de la variable a asignar, o None si todas están asignadas
        """
        pass


class ValueOrderer(ABC):
    """
    Interfaz abstracta para estrategias de ordenamiento de valores.
    
    Una estrategia de ordenamiento de valores decide en qué orden probar
    los valores del dominio de una variable.
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
            csp: Problema CSP
            assignment: Asignación actual
            current_domains: Dominios actuales de todas las variables
        
        Returns:
            Lista de valores ordenados según la estrategia
        """
        pass

