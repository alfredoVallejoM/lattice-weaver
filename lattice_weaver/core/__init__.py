# lattice_weaver/core/__init__.py

"""
Módulo Core de LatticeWeaver

Contiene definiciones fundamentales para CSPs y otras estructuras.
"""

from .csp_problem import (
    CSP,
    Constraint,
    is_satisfiable,
    verify_solution,
    generate_nqueens,
    generate_random_csp,
    solve_subproblem_exhaustive
)
from .csp_solver_fibration_integration import solve_csp_with_fibration_flow

__all__ = [
    'CSP',
    'Constraint',
    'is_satisfiable',
    'verify_solution',
    'generate_nqueens',
    'generate_random_csp',
    'solve_subproblem_exhaustive',
    'solve_csp_with_fibration_flow'
]

