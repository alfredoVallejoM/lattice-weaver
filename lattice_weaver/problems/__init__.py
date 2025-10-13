"""
Módulo de familias de problemas CSP para LatticeWeaver.

Este módulo proporciona un catálogo extenso de familias de problemas CSP
clásicos con generadores paramétricos, validadores y metadatos.

Familias disponibles:
- N-Queens
- Graph Coloring
- Sudoku
- Map Coloring
- Job Shop Scheduling
- Latin Square
- Magic Square
- Knapsack
- Logic Puzzles (Zebra, Einstein)

Usage:
    from lattice_weaver.problems import get_catalog
    
    catalog = get_catalog()
    engine = catalog.generate_problem('nqueens', n=8)
    
    from lattice_weaver.core.csp_engine.solver import AdaptiveConsistencyEngine as CSPSolver
    solver = CSPSolver(engine)
    solution = solver.solve()
"""

from .base import ProblemFamily
from .catalog import (
    ProblemCatalog,
    get_catalog,
    register_family,
    get_family
)

# Importar generadores para que se auto-registren
from .generators import (
    NQueensProblem,
    GraphColoringProblem,
    SudokuProblem,
)

__all__ = [
    'ProblemFamily',
    'ProblemCatalog',
    'get_catalog',
    'register_family',
    'get_family',
    'NQueensProblem',
    'GraphColoringProblem',
    'SudokuProblem',
]

__version__ = '1.0.0'

