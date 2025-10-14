# lattice_weaver/renormalization/__init__.py

"""
Módulo de Renormalización Computacional

Implementa el flujo de renormalización para CSPs, incluyendo particionamiento,
derivación de dominios y restricciones efectivas, y un solver basado en RG.
"""

from .core import renormalize_csp, refine_solution, RenormalizationSolver
from .partition import VariablePartitioner
from .effective_domains import EffectiveDomainDeriver, LazyEffectiveDomain
from .effective_constraints import EffectiveConstraintDeriver, LazyEffectiveConstraint

__all__ = [
    'renormalize_csp',
    'refine_solution',
    'RenormalizationSolver',
    'VariablePartitioner',
    'EffectiveDomainDeriver',
    'LazyEffectiveDomain',
    'EffectiveConstraintDeriver',
    'LazyEffectiveConstraint'
]

