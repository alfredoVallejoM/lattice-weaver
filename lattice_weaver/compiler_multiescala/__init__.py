"""
Compilador Multiescala de LatticeWeaver

Este módulo implementa el compilador multiescala que opera a través de múltiples
niveles de abstracción (L0-L6) para la renormalización y optimización de CSPs.
"""

from .base import AbstractionLevel
from .level_0 import Level0

__all__ = [
    'AbstractionLevel',
    'Level0',
]

