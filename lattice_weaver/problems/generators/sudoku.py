"""
Generador de problemas Sudoku.

El Sudoku es un puzzle lógico donde se debe completar una cuadrícula
de tal forma que cada fila, columna y bloque contenga todos los dígitos
sin repetición.
"""

from typing import Dict, Any, List, Tuple, Set
import random
import math
import logging

from ..base import ProblemFamily
from ..utils.validators import validate_sudoku_solution

from lattice_weaver.core.csp_problem import CSP, Constraint

logger = logging.getLogger(__name__)


class SudokuProblem(ProblemFamily):
    """
    Familia de problemas Sudoku.
    
    Genera instancias del puzzle Sudoku donde se debe completar una cuadrícula
    con dígitos siguiendo las reglas del juego.
    
    Variables: C_{i}_{j} para cada celda (i, j)
    Dominios: [1, size] para cada variable
    Restricciones:
        - Fila: todos diferentes (AllDifferent)
        - Columna: todos diferentes (AllDifferent)
        - Bloque: todos diferentes (AllDifferent)
    """
    
    def __init__(self):
        """Inicializa la familia Sudoku."""
        super().__init__(
            name='sudoku',
            description='Puzzle Sudoku: completar cuadrícula con dígitos únicos por fila/columna/bloque'
        )
    
    def get_param_schema(self) -> Dict[str, Dict[str, Any]]:
        """
        Retorna el esquema de parámetros para Sudoku.
        
        Returns:
            Dict con esquema de parámetros
        """
        return {
            'size': {
                'type': int,
                'required': False,
                'default': 9,
                'choices': [4, 9, 16, 25],
                'description': 'Tamaño de la cuadrícula (4, 9, 16, 25)'
            },
            'n_clues': {
                'type': int,
                'required': False,
                'min': 0,
                'description': 'Número de pistas iniciales (opcional, se calcula automáticamente)'
            },
            'difficulty': {
                'type': str,
                'required': False,
                'default': 'medium',
                'choices': ['easy', 'medium', 'hard', 'expert', 'empty'],
                'description': 'Nivel de dificultad del puzzle'
            },
            'seed': {
                'type': int,
                'required': False,
                'description': 'Semilla para reproducibilidad'
            }
        }
    
    def get_default_params(self) -> Dict[str, Any]:
        """
        Retorna parámetros por defecto para Sudoku.
        
        Returns:
            Dict con parámetros por defecto (9x9, dificultad media)
        """
        return {
            'size': 9,
            'difficulty': 'medium'
        }
    
    def _get_clues_for_difficulty(self, size: int, difficulty: str) -> int:
        """
        Calcula el número de pistas según la dificultad.
        
        Args:
            size: Tamaño del Sudoku
            difficulty: Nivel de dificultad
            
        Returns:
            int: Número de pistas
        """
        total_cells = size * size
        
        if difficulty == 'empty':
            return 0
        
        # Porcentajes aproximados de celdas llenas
        percentages = {
            'easy': 0.60,    # 60% de celdas llenas
            'medium': 0.40,  # 40% de celdas llenas
            'hard': 0.30,    # 30% de celdas llenas
            'expert': 0.25   # 25% de celdas llenas
        }
        
        percentage = percentages.get(difficulty, 0.40)
        return int(total_cells * percentage)
    
    def _get_block_indices(self, size: int, row: int, col: int) -> List[Tuple[int, int]]:
        """
        Obtiene los índices de todas las celdas en el mismo bloque.
        
        Args:
            size: Tamaño del Sudoku
            row: Fila de la celda
            col: Columna de la celda
            
        Returns:
            Lista de tuplas (row, col) en el mismo bloque
        """
        block_size = int(math.sqrt(size))
        block_row = (row // block_size) * block_size
        block_col = (col // block_size) * block_size
        
        indices = []
        for r in range(block_row, block_row + block_size):
            for c in range(block_col, block_col + block_size):
                if (r, c) != (row, col):
                    indices.append((r, c))
        
        return indices
    
    def generate(self, **params):
        """
        Genera una instancia del problema Sudoku.
        
        Args:
            size: Tamaño de la cuadrícula (4, 9, 16, 25)
            n_clues: Número de pistas (opcional)
            difficulty: Nivel de dificultad ('easy', 'medium', 'hard', 'expert', 'empty')
            seed: Semilla para reproducibilidad (opcional)
            
        Returns:
            ArcEngine: Motor CSP configurado con el problema Sudoku
            
        Example:
            >>> from lattice_weaver.problems.generators.sudoku import SudokuProblem
            >>> family = SudokuProblem()
            >>> engine = family.generate(size=9, difficulty='medium')
            >>> print(f"Variables: {len(engine.variables)}")
            Variables: 81
        """
        # Validar parámetros
        self.validate_params(**params)
        
        size = params.get('size', 9)
        difficulty = params.get('difficulty', 'medium')
        seed = params.get('seed')
        
        if seed is not None:
            random.seed(seed)
        
        # Calcular número de pistas
        if 'n_clues' in params:
            n_clues = params['n_clues']
        else:
            n_clues = self._get_clues_for_difficulty(size, difficulty)
        
        logger.info(f"Generando problema Sudoku: {size}x{size}, dificultad={difficulty}, pistas={n_clues}")
        
        # Verificar que size sea un cuadrado perfecto
        block_size = int(math.sqrt(size))
        if block_size * block_size != size:
            raise ValueError(f"Tamaño de Sudoku inválido: {size} (debe ser 4, 9, 16, 25)")
        
        # Crear ArcEngine
        engine = CSP()
        
        # Añadir variables (una por celda)
        for row in range(size):
            for col in range(size):
                var_name = f'C_{row}_{col}'
                domain = list(range(1, size + 1))  # Dígitos [1, size]
                engine.variables.add(var_name)
                engine.domains[var_name] = domain
        
        logger.debug(f'Añadidas {size*size} variables')
        
        # Añadir restricciones de fila
        constraint_count = 0
        for row in range(size):
            for col1 in range(size):
                for col2 in range(col1 + 1, size):
                    var1 = f'C_{row}_{col1}'
                    var2 = f'C_{row}_{col2}'
                    
                    engine.constraints.append(Constraint(scope=[var1, var2], relation=lambda a, b: a != b, name='not_equal'))
                    constraint_count += 1
        
        logger.debug(f'Añadidas restricciones de fila: {constraint_count}')
        
        # Añadir restricciones de columna
        for col in range(size):
            for row1 in range(size):
                for row2 in range(row1 + 1, size):
                    var1 = f'C_{row1}_{col}'
                    var2 = f'C_{row2}_{col}'
                    
                    engine.constraints.append(Constraint(scope=[var1, var2], relation=lambda a, b: a != b, name='not_equal'))
                    constraint_count += 1
        
        logger.debug(f'Total restricciones (fila+columna): {constraint_count}')
        
        # Añadir restricciones de bloque
        for block_row in range(block_size):
            for block_col in range(block_size):
                # Obtener todas las celdas del bloque
                cells = []
                for i in range(block_size):
                    for j in range(block_size):
                        row = block_row * block_size + i
                        col = block_col * block_size + j
                        cells.append((row, col))
                
                # Añadir restricciones entre todas las parejas
                for idx1 in range(len(cells)):
                    for idx2 in range(idx1 + 1, len(cells)):
                        row1, col1 = cells[idx1]
                        row2, col2 = cells[idx2]
                        var1 = f'C_{row1}_{col1}'
                        var2 = f'C_{row2}_{col2}'
                        
                        engine.constraints.append(Constraint(scope=[var1, var2], relation=lambda a, b: a != b, name='not_equal'))
                        constraint_count += 1
        
        logger.info(f'Total restricciones: {constraint_count}')
        
        # Aplicar pistas (reducir dominios de celdas específicas)
        if n_clues > 0:
            all_cells = [(r, c) for r in range(size) for c in range(size)]
            random.shuffle(all_cells)
            clue_cells = all_cells[:n_clues]
            
            for row, col in clue_cells:
                var_name = f'C_{row}_{col}'
                # Obtener el dominio actual del ArcEngine
                current_domain_values = list(engine.domains[var_name])
                if current_domain_values:
                    clue_value = random.choice(current_domain_values)
                    # Reducir el dominio a un solo valor
                    engine.domains[var_name] = [clue_value]
                    logger.debug(f'Pista en {var_name}: {clue_value}')
        
        logger.info(f'Problema Sudoku generado: {size}x{size}, {n_clues} pistas')
        
        return engine
    
    def validate_solution(self, solution: Dict[str, Any], **params) -> bool:
        """
        Valida si una solución es correcta para el problema Sudoku.
        
        Args:
            solution: Diccionario {f'C_{i}_{j}': valor} para celdas
            **params: Debe incluir 'size'
            
        Returns:
            bool: True si la solución es válida
        """
        self.validate_params(**params)
        size = params.get('size', 9)
        
        return validate_sudoku_solution(solution, size)
    
    def get_metadata(self, **params) -> Dict[str, Any]:
        """
        Obtiene metadatos del problema Sudoku.
        
        Args:
            **params: Parámetros del problema
            
        Returns:
            Dict con metadatos del problema
        """
        self.validate_params(**params)
        
        size = params.get('size', 9)
        difficulty = params.get('difficulty', 'medium')
        
        if 'n_clues' in params:
            n_clues = params['n_clues']
        else:
            n_clues = self._get_clues_for_difficulty(size, difficulty)
        
        block_size = int(math.sqrt(size))
        n_variables = size * size
        
        # Calcular número de restricciones
        # Filas: size * C(size, 2)
        # Columnas: size * C(size, 2)
        # Bloques: (size) * C(block_size^2, 2)
        constraints_per_group = size * (size - 1) // 2
        n_constraints = 3 * size * constraints_per_group
        
        return {
            'family': self.name,
            'size': size,
            'block_size': block_size,
            'difficulty': difficulty,
            'n_clues': n_clues,
            'n_variables': n_variables,
            'n_constraints': n_constraints,
            'domain_size': size,
            'complexity': 'O(n^3)',
            'problem_type': 'logic_puzzle',
            'description': f'{size}x{size} Sudoku with {n_clues} clues ({difficulty})'
        }


# Auto-registro en el catálogo global
def _register():
    """Registra SudokuProblem en el catálogo global."""
    try:
        from ..catalog import register_family
        register_family(SudokuProblem())
        logger.info("SudokuProblem registrado en el catálogo")
    except Exception as e:
        logger.warning(f"No se pudo auto-registrar SudokuProblem: {e}")

_register()

