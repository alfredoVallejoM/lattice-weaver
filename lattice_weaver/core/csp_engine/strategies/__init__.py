"""
Módulo de estrategias para CSPSolver.

Este módulo define las interfaces abstractas y las implementaciones concretas
de estrategias para selección de variables y ordenamiento de valores en el CSPSolver.

Las estrategias permiten modularizar y hacer intercambiables las heurísticas de búsqueda,
facilitando la experimentación con diferentes enfoques y preparando el terreno para
integración de técnicas avanzadas (ML, análisis estructural, etc.).

Incluye estrategias básicas (Fase 2) y estrategias guiadas por FCA (Fase 3).
"""

from .base import VariableSelector, ValueOrderer

# Estrategias básicas (Fase 2)
try:
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
    _PHASE2_AVAILABLE = True
except ImportError:
    _PHASE2_AVAILABLE = False
    # Fallback: definir FirstUnassignedSelector y NaturalOrderer básicos
    class FirstUnassignedSelector(VariableSelector):
        def select(self, csp, assignment, current_domains):
            for var in csp.variables:
                if var not in assignment:
                    return var
            return None
    
    class NaturalOrderer(ValueOrderer):
        def order(self, var, csp, assignment, current_domains):
            return list(current_domains[var])

# Estrategias FCA (Fase 3)
try:
    from .fca_guided import (
        FCAGuidedSelector,
        FCAOnlySelector,
        FCAClusterSelector
    )
    _PHASE3_AVAILABLE = True
except ImportError:
    _PHASE3_AVAILABLE = False

# Estrategias Topológicas (Fase 4)
try:
    from .topology_guided import (
        TopologyGuidedSelector,
        ComponentBasedSelector
    )
    _PHASE4_TOPO_AVAILABLE = True
except ImportError:
    _PHASE4_TOPO_AVAILABLE = False

# Estrategias Híbridas Multiescala (Fase 4)
try:
    from .hybrid_multiescala import (
        HybridFCATopologySelector,
        AdaptiveMultiscaleSelector
    )
    _PHASE4_HYBRID_AVAILABLE = True
except ImportError:
    _PHASE4_HYBRID_AVAILABLE = False


__all__ = [
    # Interfaces
    'VariableSelector',
    'ValueOrderer',
    
    # Estrategias básicas (siempre disponibles)
    'FirstUnassignedSelector',
    'NaturalOrderer',
]

# Añadir estrategias de Fase 2 si están disponibles
if _PHASE2_AVAILABLE:
    __all__.extend([
        'MRVSelector',
        'DegreeSelector',
        'MRVDegreeSelector',
        'LCVOrderer',
        'RandomOrderer',
    ])

# Añadir estrategias de Fase 3 si están disponibles
if _PHASE3_AVAILABLE:
    __all__.extend([
        'FCAGuidedSelector',
        'FCAOnlySelector',
        'FCAClusterSelector',
    ])

# Añadir estrategias topológicas de Fase 4 si están disponibles
if _PHASE4_TOPO_AVAILABLE:
    __all__.extend([
        'TopologyGuidedSelector',
        'ComponentBasedSelector',
    ])

# Añadir estrategias híbridas de Fase 4 si están disponibles
if _PHASE4_HYBRID_AVAILABLE:
    __all__.extend([
        'HybridFCATopologySelector',
        'AdaptiveMultiscaleSelector',
    ])
