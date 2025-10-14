# lattice_weaver/renormalization/hierarchy.py

"""
Gestión de la Jerarquía de Abstracción Multinivel

Este módulo define las estructuras de datos para gestionar la pila de CSPs
renormalizados a través de múltiples niveles de abstracción (L0-L6).
"""

import dataclasses
from typing import List, Dict, Any, Tuple, Set

from ..core.csp_problem import CSP

@dataclasses.dataclass
class AbstractionLevel:
    """
    Representa un único nivel en la jerarquía de abstracción.

    Attributes:
        level: El número del nivel (0 para el original).
        csp: El objeto CSP en este nivel de abstracción.
        partition: La partición de variables del nivel anterior que generó este nivel.
        variable_map: Mapeo de variables del nivel anterior a las variables de este nivel.
    """
    level: int
    csp: CSP
    partition: List[Set[str]]
    variable_map: Dict[str, str]

class AbstractionHierarchy:
    """
    Gestiona la colección de niveles de abstracción para un CSP.
    """
    def __init__(self, original_csp: CSP):
        """
        Inicializa la jerarquía con el CSP original en el nivel 0.

        Args:
            original_csp: El problema de satisfacción de restricciones original.
        """
        self.levels: Dict[int, AbstractionLevel] = {}
        # El nivel 0 no tiene partición ni mapa de variables que lo generen
        self.levels[0] = AbstractionLevel(level=0, csp=original_csp, partition=[], variable_map={})
        self.highest_level = 0

    def add_level(self, level: int, csp: CSP, partition: List[Set[str]], variable_map: Dict[str, str]):
        """
        Añade un nuevo nivel de abstracción a la jerarquía.

        Args:
            level: El número del nuevo nivel a añadir.
            csp: El CSP renormalizado para este nivel.
            partition: La partición del nivel anterior.
            variable_map: El mapeo de variables del nivel anterior a este.

        Raises:
            ValueError: Si el nivel ya existe o no es consecutivo.
        """
        if level != self.highest_level + 1:
            raise ValueError(f"Solo se puede añadir un nivel consecutivo. Se esperaba {self.highest_level + 1}, pero se recibió {level}.")
        if level in self.levels:
            raise ValueError(f"El nivel {level} ya existe en la jerarquía.")
        
        self.levels[level] = AbstractionLevel(level=level, csp=csp, partition=partition, variable_map=variable_map)
        self.highest_level = level

    def get_level(self, level: int) -> AbstractionLevel:
        """
        Obtiene un nivel de abstracción específico por su número.

        Args:
            level: El número del nivel a obtener.

        Returns:
            El objeto AbstractionLevel correspondiente.

        Raises:
            KeyError: Si el nivel no se encuentra en la jerarquía.
        """
        if level not in self.levels:
            raise KeyError(f"Nivel {level} no encontrado en la jerarquía. Niveles disponibles: {list(self.levels.keys())}")
        return self.levels[level]

    def get_highest_csp(self) -> CSP:
        """
        Obtiene el CSP del nivel más alto de abstracción actualmente en la jerarquía.

        Returns:
            El objeto CSP del nivel más alto.
        """
        return self.levels[self.highest_level].csp

    def __str__(self) -> str:
        return f"AbstractionHierarchy(highest_level={self.highest_level}, levels={list(self.levels.keys())})"

    def __repr__(self) -> str:
        return self.__str__()
