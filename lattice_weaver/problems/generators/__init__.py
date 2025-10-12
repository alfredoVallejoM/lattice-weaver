"""
Generadores de familias de problemas CSP.

Este módulo contiene implementaciones de familias clásicas de problemas CSP:
- N-Queens
- Graph Coloring
- Sudoku
- Map Coloring
- Job Shop Scheduling
- Latin Square
- Magic Square
- Knapsack
- Logic Puzzles
"""

from .nqueens import NQueensProblem
from .graph_coloring import GraphColoringProblem
from .sudoku import SudokuProblem

__all__ = [
    'NQueensProblem',
    'GraphColoringProblem',
    'SudokuProblem',
]

