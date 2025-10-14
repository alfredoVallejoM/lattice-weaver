# lattice_weaver/core/csp_engine/solver.py

"""
Adaptador de compatibilidad para el módulo solver
"""

from lattice_weaver.arc_engine.csp_solver import (
    CSPSolver as AdaptiveConsistencyEngine,
    CSPSolver as AC3Solver,
    CSPSolverResult as SolutionStats,
    CSPProblem,
    CSPSolution,
    solve_csp
)

__all__ = [
    'AdaptiveConsistencyEngine',
    'AC3Solver',
    'SolutionStats',
    'CSPProblem',
    'CSPSolution',
    'solve_csp'
]

