"""
Validadores de soluciones para problemas CSP.

Este módulo proporciona funciones de utilidad para validar soluciones
de diferentes tipos de problemas CSP.
"""

from typing import Dict, Any, List, Tuple, Callable
import logging

logger = logging.getLogger(__name__)


def validate_all_different(values: List[Any]) -> bool:
    """
    Valida que todos los valores sean diferentes.
    
    Args:
        values: Lista de valores a verificar
        
    Returns:
        bool: True si todos los valores son diferentes
    """
    return len(values) == len(set(values))


def validate_binary_constraints(
    solution: Dict[str, Any],
    constraints: List[Tuple[str, str, Callable[[Any, Any], bool]]]
) -> bool:
    """
    Valida que una solución satisfaga todas las restricciones binarias.
    
    Args:
        solution: Diccionario variable -> valor
        constraints: Lista de tuplas (var1, var2, relation)
                    donde relation(val1, val2) retorna True si son consistentes
        
    Returns:
        bool: True si todas las restricciones se satisfacen
    """
    for var1, var2, relation in constraints:
        if var1 not in solution or var2 not in solution:
            logger.warning(f"Variable faltante en solución: {var1} o {var2}")
            return False
        
        val1 = solution[var1]
        val2 = solution[var2]
        
        if not relation(val1, val2):
            logger.debug(f"Restricción violada: {var1}={val1}, {var2}={val2}")
            return False
    
    return True


def validate_nqueens_solution(solution: Dict[str, int], n: int) -> bool:
    """
    Valida una solución del problema N-Queens.
    
    Args:
        solution: Diccionario {f'Q{i}': columna} para i en [0, n)
        n: Tamaño del tablero
        
    Returns:
        bool: True si la solución es válida
    """
    # Verificar que todas las reinas estén presentes
    if len(solution) != n:
        logger.debug(f"Número incorrecto de reinas: {len(solution)} != {n}")
        return False
    
    positions = []
    for i in range(n):
        var_name = f'Q{i}'
        if var_name not in solution:
            logger.debug(f"Falta reina: {var_name}")
            return False
        
        col = solution[var_name]
        if not isinstance(col, int) or not 0 <= col < n:
            logger.debug(f"Posición inválida para {var_name}: {col}")
            return False
        
        positions.append((i, col))
    
    # Verificar restricciones
    for i in range(n):
        for j in range(i + 1, n):
            row_i, col_i = positions[i]
            row_j, col_j = positions[j]
            
            # Misma columna
            if col_i == col_j:
                logger.debug(f"Reinas en misma columna: Q{i} y Q{j}")
                return False
            
            # Misma diagonal
            if abs(row_i - row_j) == abs(col_i - col_j):
                logger.debug(f"Reinas en misma diagonal: Q{i} y Q{j}")
                return False
    
    return True


def validate_graph_coloring_solution(
    solution: Dict[str, int],
    edges: List[Tuple[int, int]],
    n_nodes: int
) -> bool:
    """
    Valida una solución del problema Graph Coloring.
    
    Args:
        solution: Diccionario {f'V{i}': color} para i en [0, n_nodes)
        edges: Lista de aristas (i, j)
        n_nodes: Número de nodos
        
    Returns:
        bool: True si la solución es válida
    """
    # Verificar que todos los nodos estén coloreados
    if len(solution) != n_nodes:
        logger.debug(f"Número incorrecto de nodos: {len(solution)} != {n_nodes}")
        return False
    
    for i in range(n_nodes):
        var_name = f'V{i}'
        if var_name not in solution:
            logger.debug(f"Falta nodo: {var_name}")
            return False
    
    # Verificar que nodos adyacentes tengan colores diferentes
    for i, j in edges:
        var_i = f'V{i}'
        var_j = f'V{j}'
        
        if solution[var_i] == solution[var_j]:
            logger.debug(f"Nodos adyacentes con mismo color: {var_i} y {var_j}")
            return False
    
    return True


def validate_sudoku_solution(solution: Dict[str, int], size: int = 9) -> bool:
    """
    Valida una solución del problema Sudoku.
    
    Args:
        solution: Diccionario {f'C_{i}_{j}': valor} para celdas
        size: Tamaño del Sudoku (4, 9, 16, 25)
        
    Returns:
        bool: True si la solución es válida
    """
    import math
    block_size = int(math.sqrt(size))
    
    if block_size * block_size != size:
        raise ValueError(f"Tamaño de Sudoku inválido: {size}")
    
    # Verificar que todas las celdas estén llenas
    if len(solution) != size * size:
        logger.debug(f"Número incorrecto de celdas: {len(solution)} != {size*size}")
        return False
    
    # Verificar filas
    for row in range(size):
        values = []
        for col in range(size):
            var_name = f'C_{row}_{col}'
            if var_name not in solution:
                logger.debug(f"Falta celda: {var_name}")
                return False
            values.append(solution[var_name])
        
        if not validate_all_different(values):
            logger.debug(f"Fila {row} tiene valores repetidos")
            return False
        
        if not all(1 <= v <= size for v in values):
            logger.debug(f"Fila {row} tiene valores fuera de rango")
            return False
    
    # Verificar columnas
    for col in range(size):
        values = []
        for row in range(size):
            var_name = f'C_{row}_{col}'
            values.append(solution[var_name])
        
        if not validate_all_different(values):
            logger.debug(f"Columna {col} tiene valores repetidos")
            return False
    
    # Verificar bloques
    for block_row in range(block_size):
        for block_col in range(block_size):
            values = []
            for i in range(block_size):
                for j in range(block_size):
                    row = block_row * block_size + i
                    col = block_col * block_size + j
                    var_name = f'C_{row}_{col}'
                    values.append(solution[var_name])
            
            if not validate_all_different(values):
                logger.debug(f"Bloque ({block_row},{block_col}) tiene valores repetidos")
                return False
    
    return True


def validate_latin_square_solution(matrix: List[List[int]], n: int) -> bool:
    """
    Valida una solución del problema Latin Square.
    
    Args:
        matrix: Matriz n×n con valores
        n: Tamaño del cuadrado latino
        
    Returns:
        bool: True si la solución es válida
    """
    # Verificar tamaño
    if len(matrix) != n:
        logger.debug(f"Número incorrecto de filas: {len(matrix)} != {n}")
        return False
    
    for row in matrix:
        if len(row) != n:
            logger.debug(f"Número incorrecto de columnas en fila")
            return False
    
    # Verificar filas
    for row_idx, row in enumerate(matrix):
        if not validate_all_different(row):
            logger.debug(f"Fila {row_idx} tiene valores repetidos")
            return False
        
        if not all(1 <= v <= n for v in row):
            logger.debug(f"Fila {row_idx} tiene valores fuera de rango [1, {n}]")
            return False
    
    # Verificar columnas
    for col in range(n):
        values = [matrix[row][col] for row in range(n)]
        
        if not validate_all_different(values):
            logger.debug(f"Columna {col} tiene valores repetidos")
            return False
    
    return True


def validate_magic_square_solution(solution: Dict[str, int], n: int) -> bool:
    """
    Valida una solución del problema Magic Square.
    
    Args:
        solution: Diccionario {f'M_{i}_{j}': valor} para celdas
        n: Tamaño del cuadrado mágico
        
    Returns:
        bool: True si la solución es válida
    """
    magic_constant = n * (n * n + 1) // 2
    
    # Verificar que todas las celdas estén llenas
    if len(solution) != n * n:
        logger.debug(f"Número incorrecto de celdas: {len(solution)} != {n*n}")
        return False
    
    # Verificar que todos los valores sean diferentes y estén en [1, n²]
    values_set = set()
    for row in range(n):
        for col in range(n):
            var_name = f'M_{row}_{col}'
            if var_name not in solution:
                logger.debug(f"Falta celda: {var_name}")
                return False
            
            val = solution[var_name]
            if not 1 <= val <= n * n:
                logger.debug(f"Valor fuera de rango en {var_name}: {val}")
                return False
            
            if val in values_set:
                logger.debug(f"Valor repetido: {val}")
                return False
            
            values_set.add(val)
    
    # Verificar sumas de filas
    for row in range(n):
        row_sum = sum(solution[f'M_{row}_{col}'] for col in range(n))
        if row_sum != magic_constant:
            logger.debug(f"Suma de fila {row} incorrecta: {row_sum} != {magic_constant}")
            return False
    
    # Verificar sumas de columnas
    for col in range(n):
        col_sum = sum(solution[f'M_{row}_{col}'] for row in range(n))
        if col_sum != magic_constant:
            logger.debug(f"Suma de columna {col} incorrecta: {col_sum} != {magic_constant}")
            return False
    
    # Verificar suma de diagonal principal
    diag1_sum = sum(solution[f'M_{i}_{i}'] for i in range(n))
    if diag1_sum != magic_constant:
        logger.debug(f"Suma de diagonal principal incorrecta: {diag1_sum} != {magic_constant}")
        return False
    
    # Verificar suma de diagonal secundaria
    diag2_sum = sum(solution[f'M_{i}_{n-1-i}'] for i in range(n))
    if diag2_sum != magic_constant:
        logger.debug(f"Suma de diagonal secundaria incorrecta: {diag2_sum} != {magic_constant}")
        return False
    
    return True




def validate_magic_square_solution(matrix: List[List[int]], n: int) -> bool:
    """
    Valida una solución de cuadrado mágico.
    
    Un cuadrado mágico n×n válido debe cumplir:
    - Todas las filas suman M = n(n²+1)/2
    - Todas las columnas suman M
    - Ambas diagonales suman M
    - Todos los números de 1 a n² aparecen exactamente una vez
    
    Args:
        matrix: Matriz n×n con la solución
        n: Tamaño del cuadrado
    
    Returns:
        bool: True si es un cuadrado mágico válido
    """
    # Calcular suma mágica
    magic_sum = n * (n * n + 1) // 2
    
    # Verificar que la matriz sea n×n
    if len(matrix) != n:
        logger.debug(f"Número de filas incorrecto: {len(matrix)} != {n}")
        return False
    
    for row in matrix:
        if len(row) != n:
            logger.debug(f"Número de columnas incorrecto: {len(row)} != {n}")
            return False
    
    # Verificar sumas de filas
    for i, row in enumerate(matrix):
        row_sum = sum(row)
        if row_sum != magic_sum:
            logger.debug(f"Suma de fila {i} incorrecta: {row_sum} != {magic_sum}")
            return False
    
    # Verificar sumas de columnas
    for j in range(n):
        col_sum = sum(matrix[i][j] for i in range(n))
        if col_sum != magic_sum:
            logger.debug(f"Suma de columna {j} incorrecta: {col_sum} != {magic_sum}")
            return False
    
    # Verificar suma de diagonal principal
    diag_main_sum = sum(matrix[i][i] for i in range(n))
    if diag_main_sum != magic_sum:
        logger.debug(f"Suma de diagonal principal incorrecta: {diag_main_sum} != {magic_sum}")
        return False
    
    # Verificar suma de diagonal secundaria
    diag_anti_sum = sum(matrix[i][n-1-i] for i in range(n))
    if diag_anti_sum != magic_sum:
        logger.debug(f"Suma de diagonal secundaria incorrecta: {diag_anti_sum} != {magic_sum}")
        return False
    
    # Verificar que todos los números de 1 a n² aparezcan exactamente una vez
    all_values = []
    for row in matrix:
        all_values.extend(row)
    
    expected_values = set(range(1, n * n + 1))
    actual_values = set(all_values)
    
    if actual_values != expected_values:
        logger.debug(f"Valores incorrectos: esperado {expected_values}, obtenido {actual_values}")
        return False
    
    if len(all_values) != len(set(all_values)):
        logger.debug("Hay valores duplicados")
        return False
    
    return True

