# lattice_weaver/renormalization/__init__.py

"""
Este paquete contiene toda la lógica para la renormalización computacional.

La API pública de este módulo incluye:
- `RenormalizationSolver`: Un solver de CSP que utiliza el flujo de renormalización multinivel.
- `renormalize_multilevel`: Función para construir una jerarquía de abstracción.
- `AbstractionHierarchy`: Clase para gestionar la jerarquía de abstracción.
- `VariablePartitioner`: Herramienta para particionar variables de un CSP.
"""

from .hierarchy import AbstractionHierarchy, AbstractionLevel
from .core import renormalize_multilevel, RenormalizationSolver
from .partition import VariablePartitioner

__all__ = [
    "RenormalizationSolver",
    "renormalize_multilevel",
    "AbstractionHierarchy",
    "AbstractionLevel",
    "VariablePartitioner",
]

