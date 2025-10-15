"""
Restricciones Serializables para Multiprocessing

Implementa restricciones como clases serializables en lugar de lambdas,
permitiendo paralelización real con multiprocessing.

Autor: LatticeWeaver Team
Fecha: 11 de Octubre de 2025
"""

from abc import ABC, abstractmethod
from typing import Any, Set, List
import pickle


class SerializableConstraint(ABC):
    """
    Clase base para restricciones serializables.
    
    Todas las restricciones deben heredar de esta clase para ser
    compatibles con multiprocessing.
    """
    
    @abstractmethod
    def check(self, val1: Any, val2: Any) -> bool:
        """
        Verifica si dos valores satisfacen la restricción.
        
        Args:
            val1: Valor de la primera variable
            val2: Valor de la segunda variable
        
        Returns:
            True si la restricción se satisface
        """
        pass
    
    @abstractmethod
    def __repr__(self) -> str:
        """Representación en string."""
        pass
    
    def __call__(self, val1: Any, val2: Any) -> bool:
        """Permite usar la restricción como función."""
        return self.check(val1, val2)


# ============================================================================
# Restricciones de Comparación
# ============================================================================

class LessThanConstraint(SerializableConstraint):
    """Restricción: var1 < var2"""
    
    def check(self, val1: Any, val2: Any) -> bool:
        return val1 < val2
    
    def __repr__(self) -> str:
        return "LessThan"


class LessEqualConstraint(SerializableConstraint):
    """Restricción: var1 <= var2"""
    
    def check(self, val1: Any, val2: Any) -> bool:
        return val1 <= val2
    
    def __repr__(self) -> str:
        return "LessEqual"


class GreaterThanConstraint(SerializableConstraint):
    """Restricción: var1 > var2"""
    
    def check(self, val1: Any, val2: Any) -> bool:
        return val1 > val2
    
    def __repr__(self) -> str:
        return "GreaterThan"


class GreaterEqualConstraint(SerializableConstraint):
    """Restricción: var1 >= var2"""
    
    def check(self, val1: Any, val2: Any) -> bool:
        return val1 >= val2
    
    def __repr__(self) -> str:
        return "GreaterEqual"


class EqualConstraint(SerializableConstraint):
    """Restricción: var1 == var2"""
    
    def check(self, val1: Any, val2: Any) -> bool:
        return val1 == val2
    
    def __repr__(self) -> str:
        return "Equal"


class NotEqualConstraint(SerializableConstraint):
    """Restricción: var1 != var2"""
    
    def check(self, val1: Any, val2: Any) -> bool:
        return val1 != val2
    
    def __repr__(self) -> str:
        return "NotEqual"


# ============================================================================
# Restricciones Aritméticas
# ============================================================================

class SumEqualConstraint(SerializableConstraint):
    """Restricción: var1 + var2 == value"""
    
    def __init__(self, target: int):
        self.target = target
    
    def check(self, val1: Any, val2: Any) -> bool:
        return val1 + val2 == self.target
    
    def __repr__(self) -> str:
        return f"SumEqual({self.target})"


class DifferenceEqualConstraint(SerializableConstraint):
    """Restricción: var1 - var2 == value"""
    
    def __init__(self, target: int):
        self.target = target
    
    def check(self, val1: Any, val2: Any) -> bool:
        return val1 - val2 == self.target
    
    def __repr__(self) -> str:
        return f"DifferenceEqual({self.target})"


class ProductEqualConstraint(SerializableConstraint):
    """Restricción: var1 * var2 == value"""
    
    def __init__(self, target: int):
        self.target = target
    
    def check(self, val1: Any, val2: Any) -> bool:
        return val1 * val2 == self.target
    
    def __repr__(self) -> str:
        return f"ProductEqual({self.target})"


class ModuloEqualConstraint(SerializableConstraint):
    """Restricción: var1 % var2 == value"""
    
    def __init__(self, target: int):
        self.target = target
    
    def check(self, val1: Any, val2: Any) -> bool:
        if val2 == 0:
            return False
        return val1 % val2 == self.target
    
    def __repr__(self) -> str:
        return f"ModuloEqual({self.target})"


# ============================================================================
# Restricciones de Conjuntos
# ============================================================================

class InSetConstraint(SerializableConstraint):
    """Restricción: (var1, var2) in allowed_pairs"""
    
    def __init__(self, allowed_pairs: Set[tuple]):
        self.allowed_pairs = frozenset(allowed_pairs)
    
    def check(self, val1: Any, val2: Any) -> bool:
        return (val1, val2) in self.allowed_pairs
    
    def __repr__(self) -> str:
        return f"InSet({len(self.allowed_pairs)} pairs)"


class NotInSetConstraint(SerializableConstraint):
    """Restricción: (var1, var2) not in forbidden_pairs"""
    
    def __init__(self, forbidden_pairs: Set[tuple]):
        self.forbidden_pairs = frozenset(forbidden_pairs)
    
    def check(self, val1: Any, val2: Any) -> bool:
        return (val1, val2) not in self.forbidden_pairs
    
    def __repr__(self) -> str:
        return f"NotInSet({len(self.forbidden_pairs)} pairs)"


# ============================================================================
# Restricciones Lógicas
# ============================================================================

class AndConstraint(SerializableConstraint):
    """Restricción: constraint1 AND constraint2"""
    
    def __init__(self, constraint1: SerializableConstraint, constraint2: SerializableConstraint):
        self.constraint1 = constraint1
        self.constraint2 = constraint2
    
    def check(self, val1: Any, val2: Any) -> bool:
        return self.constraint1.check(val1, val2) and self.constraint2.check(val1, val2)
    
    def __repr__(self) -> str:
        return f"And({self.constraint1}, {self.constraint2})"


class OrConstraint(SerializableConstraint):
    """Restricción: constraint1 OR constraint2"""
    
    def __init__(self, constraint1: SerializableConstraint, constraint2: SerializableConstraint):
        self.constraint1 = constraint1
        self.constraint2 = constraint2
    
    def check(self, val1: Any, val2: Any) -> bool:
        return self.constraint1.check(val1, val2) or self.constraint2.check(val1, val2)
    
    def __repr__(self) -> str:
        return f"Or({self.constraint1}, {self.constraint2})"


class NotConstraint(SerializableConstraint):
    """Restricción: NOT constraint"""
    
    def __init__(self, constraint: SerializableConstraint):
        self.constraint = constraint
    
    def check(self, val1: Any, val2: Any) -> bool:
        return not self.constraint.check(val1, val2)
    
    def __repr__(self) -> str:
        return f"Not({self.constraint})"


# ============================================================================
# Restricciones Especializadas
# ============================================================================

class AllDifferentPairConstraint(SerializableConstraint):
    """Restricción: var1 != var2 (optimizada para AllDifferent)"""
    
    def check(self, val1: Any, val2: Any) -> bool:
        return val1 != val2
    
    def __repr__(self) -> str:
        return "AllDifferentPair"


class NoAttackQueensConstraint(SerializableConstraint):
    """
    Restricción para N-Reinas: dos reinas no se atacan.
    
    Asume que val1 y val2 son filas, y las columnas son implícitas
    por las posiciones de las variables.
    """
    
    def __init__(self, col_diff: int):
        """
        Args:
            col_diff: Diferencia de columnas entre las dos reinas
        """
        self.col_diff = abs(col_diff)
    
    def check(self, row1: Any, row2: Any) -> bool:
        # Diferentes filas
        if row1 == row2:
            return False
        
        # No en misma diagonal
        if abs(row1 - row2) == self.col_diff:
            return False
        
        return True
    
    def __repr__(self) -> str:
        return f"NoAttackQueens(col_diff={self.col_diff})"


class SudokuConstraint(SerializableConstraint):
    """Restricción para Sudoku: valores diferentes en fila/columna/bloque"""
    
    def check(self, val1: Any, val2: Any) -> bool:
        return val1 != val2
    
    def __repr__(self) -> str:
        return "SudokuDifferent"


# ============================================================================
# Funciones de Utilidad
# ============================================================================

def test_serializability(constraint: SerializableConstraint) -> bool:
    """
    Prueba si una restricción es serializable con pickle.
    
    Args:
        constraint: Restricción a probar
    
    Returns:
        True si es serializable
    """
    try:
        serialized = pickle.dumps(constraint)
        deserialized = pickle.loads(serialized)
        
        # Probar que funciona igual
        test_result = constraint.check(1, 2) == deserialized.check(1, 2)
        
        return test_result
    except Exception as e:
        print(f"Error de serialización: {e}")
        return False


def create_constraint_from_lambda(lambda_func: callable, description: str = "Custom") -> SerializableConstraint:
    """
    Crea una restricción serializable a partir de una lambda.
    
    ADVERTENCIA: Esto NO hace que la lambda sea serializable.
    Solo envuelve la lambda en una clase para compatibilidad.
    No usar con multiprocessing.
    
    Args:
        lambda_func: Función lambda
        description: Descripción de la restricción
    
    Returns:
        Restricción serializable (pero la lambda interna no lo es)
    """
    class LambdaConstraint(SerializableConstraint):
        def __init__(self, func, desc):
            self.func = func
            self.desc = desc
        
        def check(self, val1: Any, val2: Any) -> bool:
            return self.func(val1, val2)
        
        def __repr__(self) -> str:
            return f"Lambda({self.desc})"
    
    return LambdaConstraint(lambda_func, description)


# ============================================================================
# Aliases Comunes
# ============================================================================

# Aliases para facilitar uso
LT = LessThanConstraint
LE = LessEqualConstraint
GT = GreaterThanConstraint
GE = GreaterEqualConstraint
EQ = EqualConstraint
NE = NotEqualConstraint
AllDiff = AllDifferentPairConstraint

