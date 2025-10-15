# lattice_weaver/core/__init__.py

"""
Módulo Core de LatticeWeaver

Contiene definiciones fundamentales para CSPs y otras estructuras.
"""

from .csp_problem import (
    SumConstraint,
    CSP,
    Constraint,
    is_satisfiable,
    verify_solution,
    generate_nqueens,
    generate_random_csp,
    solve_subproblem_exhaustive
)

__all__ = [
    'SumConstraint',
    'CSP',
    'Constraint',
    'is_satisfiable',
    'verify_solution',
    'generate_nqueens',
    'generate_random_csp',
    'solve_subproblem_exhaustive'
]

