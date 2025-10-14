# lattice_weaver/renormalization/__init__.py

"""
Módulo de Renormalización Computacional

Este paquete contiene toda la lógica para la renormalización computacional, combinando
la aproximación multinivel con la derivación de dominios y restricciones efectivas.

La API pública de este módulo incluye:
- `RenormalizationSolver`: Un solver de CSP que utiliza el flujo de renormalización multinivel.
- `renormalize_multilevel`: Función para construir una jerarquía de abstracción.
- `AbstractionHierarchy`: Clase para gestionar la jerarquía de abstracción.
- `VariablePartitioner`: Herramienta para particionar variables de un CSP.
- `EffectiveDomainDeriver`: Derivador de dominios efectivos.
- `LazyEffectiveDomain`: Dominio efectivo de evaluación perezosa.
- `EffectiveConstraintDeriver`: Derivador de restricciones efectivas.
- `LazyEffectiveConstraint`: Restricción efectiva de evaluación perezosa.
"""

from .hierarchy import AbstractionHierarchy, AbstractionLevel
from .core import renormalize_multilevel, RenormalizationSolver, refine_solution # Se mantiene refine_solution de la rama feature
from .partition import VariablePartitioner
from .effective_domains import EffectiveDomainDeriver, LazyEffectiveDomain
from .effective_constraints import EffectiveConstraintDeriver, LazyEffectiveConstraint

__all__ = [
    "RenormalizationSolver",
    "renormalize_multilevel",
    "refine_solution", # Añadido de la rama feature
    "AbstractionHierarchy",
    "AbstractionLevel",
    "VariablePartitioner",
    "EffectiveDomainDeriver",
    "LazyEffectiveDomain",
    "EffectiveConstraintDeriver",
    "LazyEffectiveConstraint"
]

