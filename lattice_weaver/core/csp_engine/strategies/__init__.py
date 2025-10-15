"""
Estrategias para CSPSolver

Este módulo exporta todas las estrategias disponibles para selección de variables
y ordenamiento de valores.

Autor: Manus AI
Fecha: 15 de Octubre, 2025
"""

from .fca_guided import (
    FCAGuidedSelector,
    FCAOnlySelector,
    FCAClusterSelector
)

__all__ = [
    'FCAGuidedSelector',
    'FCAOnlySelector',
    'FCAClusterSelector',
]

