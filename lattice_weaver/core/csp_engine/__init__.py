# lattice_weaver/core/csp_engine/__init__.py

"""
Módulo de Compatibilidad para Pruebas Legacy

Este módulo actúa como un adaptador que redirige las importaciones desde
la antigua estructura `lattice_weaver.core.csp_engine` a los módulos reales
en `lattice_weaver.arc_engine`.

NOTA: Este es un módulo de compatibilidad temporal. Las nuevas pruebas deben
importar directamente desde `lattice_weaver.arc_engine`.
"""

# Re-exportar desde arc_engine
from .solver import (
    CSPSolver as AdaptiveConsistencyEngine,
    CSPSolver as AC3Solver,
    CSPSolutionStats as SolutionStats,
    CSPSolution,
    solve_csp
)
from ..csp_problem import CSP as CSPProblem

from ..csp_problem import Constraint



__all__ = [
    'AdaptiveConsistencyEngine',
    'AC3Solver',
    'SolutionStats',
    'CSPProblem',
    'CSPSolution',
    'Constraint',


    'solve_csp'
]

