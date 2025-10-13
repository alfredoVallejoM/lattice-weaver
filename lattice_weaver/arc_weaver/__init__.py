"""
Motor de Coherencia Adaptativa (ACE) - LatticeWeaver v4.2

Este paquete implementa el Motor de Coherencia Adaptativa que usa
clustering dinámico y propagación de consistencia para resolver CSPs.

Módulos:
    graph_structures: Estructuras de grafo (GR y GCD)
    clustering: Detección y gestión de clústeres
    adaptive_consistency: Motor principal de resolución

Autor: LatticeWeaver Team
Versión: 4.2.0
"""

# Los módulos han sido movidos a lattice_weaver.core.csp_engine
# Este __init__.py ahora solo exporta los componentes del nuevo motor
# para mantener la compatibilidad temporalmente.

from ..core.csp_engine.graph import (
    ConstraintGraph,
    DynamicClusterGraph,
    Cluster,
    ConstraintEdge
)

from ..core.csp_engine.clustering import (
    ClusterDetector,
    BoundaryManager,
    ClusteringMetrics
)

from ..core.csp_engine.solver import (
    AdaptiveConsistencyEngine,
    AC3Solver,
    ClusterSolver,
    SolutionStats
)

from ..core.csp_engine.tracing import (
    SearchSpaceTracer,
    SearchEvent
)


__all__ = [
    # Estructuras de grafo
    'ConstraintGraph',
    'DynamicClusterGraph',
    'Cluster',
    'ConstraintEdge',
    
    # Clustering
    'ClusterDetector',
    'BoundaryManager',
    'ClusteringMetrics',
    
    # Resolución
    'AdaptiveConsistencyEngine',
    'AC3Solver',
    'ClusterSolver',
    'SolutionStats',
    
    # Tracing
    'SearchSpaceTracer',
    'SearchEvent',
]

__version__ = '4.2.0'

