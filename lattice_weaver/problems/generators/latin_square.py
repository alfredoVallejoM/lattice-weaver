"""
Generador de problemas de cuadrados latinos (Latin Square).

Este módulo implementa la familia de problemas de cuadrados latinos,
una estructura combinatoria donde se debe rellenar una cuadrícula n×n
con símbolos 1..n de manera que cada símbolo aparezca exactamente una vez
en cada fila y en cada columna.

Los cuadrados latinos son generalizaciones de Sudoku (sin restricción de bloques)
y tienen aplicaciones en diseño de experimentos, criptografía y teoría de códigos.

Características:
- Tamaños variables (3 ≤ n ≤ 25)
- Pistas parciales (clues)
- Niveles de dificultad
- Validador de soluciones
- Generación con semilla para reproducibilidad

Referencias:
- Dénes, J., & Keedwell, A. D. (1974). Latin squares and their applications
- Colbourn, C. J., & Dinitz, J. H. (2006). Handbook of Combinatorial Designs
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
import random

from lattice_weaver.arc_engine import ArcEngine
from lattice_weaver.problems.base import ProblemFamily
from lattice_weaver.problems.catalog import register_family
from lattice_weaver.problems.utils.validators import validate_latin_square_solution

logger = logging.getLogger(__name__)


class LatinSquareProblem(ProblemFamily):
    """
    Familia de problemas de cuadrados latinos.
    
    El problema consiste en rellenar una cuadrícula n×n con símbolos 1..n
    de manera que:
    - Cada símbolo aparece exactamente una vez en cada fila
    - Cada símbolo aparece exactamente una vez en cada columna
    
    Es similar a Sudoku pero sin la restricción de bloques.
    
    Parámetros:
        size (int): Tamaño de la cuadrícula (default: 5)
        difficulty (str): Nivel de dificultad (default: 'medium')
        n_clues (int): Número de pistas (opcional, se calcula automáticamente)
        seed (int): Semilla para generación aleatoria (opcional)
    
    Niveles de dificultad:
        - 'empty': Sin pistas (0%)
        - 'easy': ~60% de celdas llenas
        - 'medium': ~40% de celdas llenas
        - 'hard': ~25% de celdas llenas
        - 'expert': ~15% de celdas llenas
    
    Ejemplo:
        >>> from lattice_weaver.problems import get_catalog
        >>> catalog = get_catalog()
        >>> engine = catalog.generate_problem('latin_square', size=5, difficulty='medium', seed=42)
        >>> # Resolver y validar
        >>> solution = {...}  # Solución completa
        >>> is_valid = catalog.validate_solution('latin_square', solution, size=5)
    """
    
    def __init__(self):
        super().__init__(
            name='latin_square',
            description='Cuadrados Latinos - rellenar cuadrícula n×n con símbolos 1..n sin repetir en filas/columnas'
        )
        logger.info(f"Inicializada familia de problemas: {self.name}")
    
    def get_default_params(self) -> Dict[str, Any]:
        """Retorna parámetros por defecto."""
        return {
            'size': 5,
            'difficulty': 'medium'
        }
    
    def get_param_schema(self) -> Dict[str, Dict[str, Any]]:
        """
        Retorna el esquema de parámetros para validación.
        
        Returns:
            Diccionario con especificación de parámetros
        """
        return {
            'size': {
                'type': int,
                'required': True,
                'min': 3,
                'max': 25,
                'description': 'Tamaño de la cuadrícula (n×n)'
            },
            'difficulty': {
                'type': str,
                'required': False,
                'default': 'medium',
                'choices': ['empty', 'easy', 'medium', 'hard', 'expert'],
                'description': 'Nivel de dificultad'
            },
            'n_clues': {
                'type': int,
                'required': False,
                'min': 0,
                'description': 'Número de pistas (opcional, se calcula automáticamente)'
            },
            'seed': {
                'type': int,
                'required': False,
                'description': 'Semilla para generación aleatoria'
            }
        }
    
    def generate(self, **params) -> ArcEngine:
        """
        Genera un problema de cuadrado latino.
        
        Args:
            **params: Parámetros del problema
        
        Returns:
            ArcEngine con el problema configurado
        """
        # Validar parámetros
        self.validate_params(**params)
        
        size = params['size']
        difficulty = params.get('difficulty', 'medium')
        seed = params.get('seed', None)
        
        if seed is not None:
            random.seed(seed)
        
        logger.info(f"Generando problema Latin Square: size={size}, difficulty={difficulty}")
        
        # Calcular número de pistas según dificultad
        if 'n_clues' in params:
            n_clues = params['n_clues']
        else:
            total_cells = size * size
            difficulty_ratios = {
                'empty': 0.0,
                'easy': 0.6,
                'medium': 0.4,
                'hard': 0.25,
                'expert': 0.15
            }
            ratio = difficulty_ratios.get(difficulty, 0.4)
            n_clues = int(total_cells * ratio)
        
        # Generar pistas aleatorias
        clues = self._generate_clues(size, n_clues, seed)
        
        # Crear ArcEngine
        engine = ArcEngine()
        
        # Añadir variables (una por celda)
        for row in range(size):
            for col in range(size):
                var_name = f'C_{row}_{col}'
                
                # Si hay pista, dominio es singleton
                if (row, col) in clues:
                    domain = [clues[(row, col)]]
                else:
                    domain = list(range(1, size + 1))
                
                engine.add_variable(var_name, domain)
        
        logger.debug(f"Añadidas {size * size} variables (celdas)")
        
        # Restricciones de fila: todos diferentes en cada fila
        constraint_count = 0
        for row in range(size):
            for col1 in range(size):
                for col2 in range(col1 + 1, size):
                    var1 = f'C_{row}_{col1}'
                    var2 = f'C_{row}_{col2}'
                    
                    def different(v1, v2):
                        return v1 != v2
                    
                    different.__name__ = f'diff_row{row}_c{col1}_c{col2}'
                    cid = f'row_{row}_{col1}_{col2}'
                    engine.add_constraint(var1, var2, different, cid=cid)
                    constraint_count += 1
        
        logger.debug(f"Añadidas {constraint_count} restricciones de fila")
        
        # Restricciones de columna: todos diferentes en cada columna
        for col in range(size):
            for row1 in range(size):
                for row2 in range(row1 + 1, size):
                    var1 = f'C_{row1}_{col}'
                    var2 = f'C_{row2}_{col}'
                    
                    def different(v1, v2):
                        return v1 != v2
                    
                    different.__name__ = f'diff_col{col}_r{row1}_r{row2}'
                    cid = f'col_{col}_{row1}_{row2}'
                    engine.add_constraint(var1, var2, different, cid=cid)
                    constraint_count += 1
        
        logger.info(f"Problema Latin Square generado: {size}×{size}, {n_clues} pistas, {constraint_count} restricciones")
        
        return engine
    
    def _generate_clues(self, size: int, n_clues: int, seed: Optional[int] = None) -> Dict[Tuple[int, int], int]:
        """
        Genera pistas aleatorias para el cuadrado latino.
        
        Args:
            size: Tamaño de la cuadrícula
            n_clues: Número de pistas
            seed: Semilla para reproducibilidad
        
        Returns:
            Diccionario {(row, col): value}
        """
        if seed is not None:
            random.seed(seed)
        
        # Generar un cuadrado latino válido completo
        # Método simple: permutaciones cíclicas
        base_row = list(range(1, size + 1))
        square = []
        for i in range(size):
            row = base_row[i:] + base_row[:i]
            square.append(row)
        
        # Seleccionar n_clues celdas aleatorias
        all_positions = [(r, c) for r in range(size) for c in range(size)]
        random.shuffle(all_positions)
        
        clues = {}
        for i in range(min(n_clues, len(all_positions))):
            row, col = all_positions[i]
            clues[(row, col)] = square[row][col]
        
        return clues
    
    def validate_solution(self, solution: Dict[str, int], **params) -> bool:
        """
        Valida una solución del problema de cuadrado latino.
        
        Args:
            solution: Diccionario {variable: valor}
            **params: Parámetros del problema
        
        Returns:
            True si la solución es válida, False en caso contrario
        """
        size = params['size']
        
        # Convertir solución a matriz
        matrix = []
        for row in range(size):
            matrix_row = []
            for col in range(size):
                var_name = f'C_{row}_{col}'
                if var_name not in solution:
                    logger.debug(f"Solución incompleta: falta {var_name}")
                    return False
                matrix_row.append(solution[var_name])
            matrix.append(matrix_row)
        
        # Usar validador de latin square
        return validate_latin_square_solution(matrix, size)
    
    def get_metadata(self, **params) -> Dict[str, Any]:
        """
        Retorna metadatos del problema.
        
        Args:
            **params: Parámetros del problema
        
        Returns:
            Diccionario con metadatos
        """
        size = params['size']
        difficulty = params.get('difficulty', 'medium')
        seed = params.get('seed', None)
        
        # Calcular número de pistas
        if 'n_clues' in params:
            n_clues = params['n_clues']
        else:
            total_cells = size * size
            difficulty_ratios = {
                'empty': 0.0,
                'easy': 0.6,
                'medium': 0.4,
                'hard': 0.25,
                'expert': 0.15
            }
            ratio = difficulty_ratios.get(difficulty, 0.4)
            n_clues = int(total_cells * ratio)
        
        # Calcular número de restricciones
        # Restricciones de fila: size * (size * (size-1) / 2)
        # Restricciones de columna: size * (size * (size-1) / 2)
        n_constraints = 2 * size * (size * (size - 1) // 2)
        
        return {
            'family': self.name,
            'size': size,
            'n_variables': size * size,
            'n_constraints': n_constraints,
            'domain_size': size,
            'complexity': 'O(n^3)',
            'problem_type': 'combinatorial',
            'difficulty': difficulty,
            'n_clues': n_clues,
            'fill_ratio': round(n_clues / (size * size), 2) if size > 0 else 0
        }


# Auto-registrar la familia en el catálogo global
register_family(LatinSquareProblem())

logger.info("Familia LatinSquareProblem registrada en el catálogo global")

