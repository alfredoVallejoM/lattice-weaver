"""
Compilador Multiescala de LatticeWeaver

Este módulo implementa el compilador multiescala que opera a través de múltiples
niveles de abstracción (L0-L6) para la renormalización y optimización de CSPs.
"""

from .base import AbstractionLevel
from .level_0 import Level0
from .level_1 import Level1, ConstraintBlock
from .level_2 import Level2, LocalPattern, PatternSignature
from .level_3 import Level3, CompositeStructure, CompositeSignature
from .level_4 import Level4, DomainConcept, DomainConceptSignature
from .level_5 import Level5, MetaPattern, MetaPatternSignature

__all__ = [
    'AbstractionLevel',
    'Level0',
    'Level1',
    'Level2',
    'Level3',
    'Level4',
    'Level5',
    'ConstraintBlock',
    'LocalPattern',
    'PatternSignature',
    'CompositeStructure',
    'CompositeSignature',
    'DomainConcept',
    'DomainConceptSignature',
    'MetaPattern',
    'MetaPatternSignature',
]

