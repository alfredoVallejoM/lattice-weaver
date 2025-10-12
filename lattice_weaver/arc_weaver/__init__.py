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

from .graph_structures import (
    ConstraintGraph,
    DynamicClusterGraph,
    Cluster,
    ConstraintEdge
)

from .clustering import (
    ClusterDetector,
    BoundaryManager,
    ClusteringMetrics
)

from .adaptive_consistency import (
    AdaptiveConsistencyEngine,
    AC3Solver,
    ClusterSolver,
    SolutionStats
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
]

__version__ = '4.2.0'

