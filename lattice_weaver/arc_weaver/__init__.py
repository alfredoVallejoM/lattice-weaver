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

from lattice_weaver.arc_engine.csp_solver import CSPSolver, CSPProblem, CSPSolution


# from ..core.csp_engine.clustering import (


# from ..core.csp_engine.solver import (


# from ..core.csp_engine.tracing import (



__all__ = [
    # Estructuras de grafo
    'CSPSolver',
    'CSPProblem',
    'CSPSolution',
]

__version__ = '4.2.0'

