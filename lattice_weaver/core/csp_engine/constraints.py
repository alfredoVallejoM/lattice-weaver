# lattice_weaver/core/csp_engine/constraints.py

"""
Adaptador de compatibilidad para el módulo constraints
"""

from lattice_weaver.arc_engine.constraints import *
from typing import Any

class NE:
    """Restricción de desigualdad: X != Y"""
    
    def __call__(self, x: Any, y: Any) -> bool:
        return x != y
    
    def __repr__(self):
        return "NE()"

class LT:
    """Restricción de menor que: X < Y"""
    
    def __call__(self, x: Any, y: Any) -> bool:
        return x < y
    
    def __repr__(self):
        return "LT()"

class NoAttackQueensConstraint:
    """Restricción para el problema de N-Reinas: dos reinas no se atacan."""
    
    def __init__(self, row_diff: int):
        """
        Args:
            row_diff: Diferencia de filas entre las dos reinas
        """
        self.row_diff = row_diff
    
    def __call__(self, col1: int, col2: int) -> bool:
        """
        Verifica que dos reinas no se ataquen.
        
        Args:
            col1: Columna de la primera reina
            col2: Columna de la segunda reina
            
        Returns:
            True si no se atacan, False en caso contrario
        """
        # No pueden estar en la misma columna
        if col1 == col2:
            return False
        # No pueden estar en la misma diagonal
        if abs(col1 - col2) == self.row_diff:
            return False
        return True
    
    def __repr__(self):
        return f"NoAttackQueensConstraint(row_diff={self.row_diff})"

__all__ = ['Constraint', 'NE', 'LT', 'NoAttackQueensConstraint']

