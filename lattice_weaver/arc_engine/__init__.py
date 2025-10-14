# lattice_weaver/arc_engine/__init__.py

"""
Redirección para el módulo arc_engine (DEPRECATED).

Este módulo ahora redirige las importaciones a la nueva estructura en `lattice_weaver.core`.
"""

import warnings

warnings.warn(
    "El módulo `lattice_weaver.arc_engine` está DEPRECATED. "
    "Use `lattice_weaver.core.csp_problem` para las definiciones de CSP.",
    DeprecationWarning, stacklevel=2
)

# Redirigir las importaciones de CSP a la nueva ubicación
from ..core.csp_problem import (
    CSP,
    Constraint,
    is_satisfiable,
    verify_solution,
    generate_nqueens,
    generate_random_csp,
    solve_subproblem_exhaustive
)

__all__ = [
    'CSP',
    'Constraint',
    'is_satisfiable',
    'verify_solution',
    'generate_nqueens',
    'generate_random_csp',
    'solve_subproblem_exhaustive'
]

