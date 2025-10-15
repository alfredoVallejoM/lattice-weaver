"""
Módulo de estrategias para CSPSolver.

Este módulo define las interfaces abstractas y las implementaciones concretas
de estrategias para selección de variables y ordenamiento de valores en el CSPSolver.

Las estrategias permiten modularizar y hacer intercambiables las heurísticas de búsqueda,
facilitando la experimentación con diferentes enfoques y preparando el terreno para
integración de técnicas avanzadas (ML, análisis estructural, etc.).
"""

from .base import VariableSelector, ValueOrderer
from .variable_selectors import (
    FirstUnassignedSelector,
    MRVSelector,
    DegreeSelector,
    MRVDegreeSelector
)
from .value_orderers import (
    NaturalOrderer,
    LCVOrderer,
    RandomOrderer
)

__all__ = [
    # Interfaces base
    'VariableSelector',
    'ValueOrderer',
    
    # Selectores de variables
    'FirstUnassignedSelector',
    'MRVSelector',
    'DegreeSelector',
    'MRVDegreeSelector',
    
    # Ordenadores de valores
    'NaturalOrderer',
    'LCVOrderer',
    'RandomOrderer',
]

