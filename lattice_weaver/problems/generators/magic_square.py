"""
Generador de problemas Magic Square.

Un cuadrado mágico es una matriz n×n de números donde:
- Cada celda contiene un número único del 1 al n²
- La suma de cada fila es igual a la suma mágica M = n(n²+1)/2
- La suma de cada columna es igual a M
- La suma de cada diagonal es igual a M

Este módulo implementa la generación de problemas de cuadrados mágicos
como problemas CSP usando el framework LatticeWeaver.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
import random

from lattice_weaver.arc_engine import ArcEngine
from lattice_weaver.problems.base import ProblemFamily
from lattice_weaver.problems.catalog import register_family
from lattice_weaver.problems.utils.validators import validate_magic_square_solution

logger = logging.getLogger(__name__)


class MagicSquareProblem(ProblemFamily):
    """
    Familia de problemas de cuadrados mágicos.
    
    Un cuadrado mágico n×n contiene los números 1 a n² dispuestos de forma
    que todas las filas, columnas y diagonales sumen lo mismo (suma mágica).
    
    Attributes:
        name: Nombre de la familia ('magic_square')
        description: Descripción del problema
    """
    
    def __init__(self):
        """Inicializa la familia Magic Square."""
        super().__init__(
            name='magic_square',
            description='Magic Square problem: arrange numbers 1 to n² in an n×n grid '
                       'such that all rows, columns, and diagonals sum to the same value'
        )
        logger.info(f"Inicializada familia de problemas: {self.name}")
    
    def get_default_params(self) -> Dict[str, Any]:
        """
        Retorna parámetros por defecto.
        
        Returns:
            Diccionario con parámetros por defecto
        """
        return {
            'size': 3,
            'difficulty': 'medium',
            'n_clues': None,
            'seed': None
        }
    
    def get_param_schema(self) -> Dict[str, Dict[str, Any]]:
        """
        Retorna el esquema de parámetros.
        
        Returns:
            Diccionario con el esquema de parámetros
        """
        return {
            'size': {
                'type': int,
                'required': True,
                'min': 3,
                'max': 10,
                'description': 'Tamaño del cuadrado mágico (n×n)'
            },
            'difficulty': {
                'type': str,
                'required': False,
                'choices': ['empty', 'easy', 'medium', 'hard', 'expert'],
                'description': 'Nivel de dificultad (determina número de pistas)'
            },
            'n_clues': {
                'type': int,
                'required': False,
                'min': 0,
                'description': 'Número de pistas (sobreescribe difficulty)'
            },
            'seed': {
                'type': int,
                'required': False,
                'description': 'Semilla para generación aleatoria reproducible'
            }
        }
    
    def _calculate_magic_sum(self, n: int) -> int:
        """
        Calcula la suma mágica para un cuadrado de tamaño n.
        
        Args:
            n: Tamaño del cuadrado
        
        Returns:
            Suma mágica M = n(n²+1)/2
        """
        return n * (n * n + 1) // 2
    
    def _generate_clues(self, n: int, n_clues: int, seed: Optional[int] = None) -> Dict[Tuple[int, int], int]:
        """
        Genera pistas para el cuadrado mágico.
        
        Args:
            n: Tamaño del cuadrado
            n_clues: Número de pistas a generar
            seed: Semilla para reproducibilidad
        
        Returns:
            Diccionario {(row, col): value} con las pistas
        """
        if seed is not None:
            random.seed(seed)
        
        # Para cuadrados mágicos, generar pistas válidas es complejo
        # Por simplicidad, generamos algunas posiciones fijas
        clues = {}
        
        if n == 3 and n_clues > 0:
            # Cuadrado mágico 3×3 clásico (Lo Shu)
            magic_3x3 = [
                [2, 7, 6],
                [9, 5, 1],
                [4, 3, 8]
            ]
            
            # Seleccionar n_clues posiciones aleatorias
            positions = [(i, j) for i in range(3) for j in range(3)]
            random.shuffle(positions)
            
            for i in range(min(n_clues, 9)):
                row, col = positions[i]
                clues[(row, col)] = magic_3x3[row][col]
        
        elif n == 4 and n_clues > 0:
            # Cuadrado mágico 4×4 clásico (Durero)
            magic_4x4 = [
                [16, 3, 2, 13],
                [5, 10, 11, 8],
                [9, 6, 7, 12],
                [4, 15, 14, 1]
            ]
            
            positions = [(i, j) for i in range(4) for j in range(4)]
            random.shuffle(positions)
            
            for i in range(min(n_clues, 16)):
                row, col = positions[i]
                clues[(row, col)] = magic_4x4[row][col]
        
        elif n == 5 and n_clues > 0:
            # Cuadrado mágico 5×5
            magic_5x5 = [
                [17, 24, 1, 8, 15],
                [23, 5, 7, 14, 16],
                [4, 6, 13, 20, 22],
                [10, 12, 19, 21, 3],
                [11, 18, 25, 2, 9]
            ]
            
            positions = [(i, j) for i in range(5) for j in range(5)]
            random.shuffle(positions)
            
            for i in range(min(n_clues, 25)):
                row, col = positions[i]
                clues[(row, col)] = magic_5x5[row][col]
        
        return clues
    
    def generate(self, **params) -> ArcEngine:
        """
        Genera un problema de cuadrado mágico.
        
        Args:
            **params: Parámetros del problema
        
        Returns:
            ArcEngine configurado con el problema
        """
        # Validar parámetros
        self.validate_params(**params)
        
        size = params['size']
        difficulty = params.get('difficulty', 'medium')
        n_clues = params.get('n_clues', None)
        seed = params.get('seed', None)
        
        # Calcular número de pistas según dificultad
        if n_clues is None:
            difficulty_ratios = {
                'empty': 0.0,
                'easy': 0.5,
                'medium': 0.3,
                'hard': 0.2,
                'expert': 0.1
            }
            n_clues = int(size * size * difficulty_ratios[difficulty])
        
        # Generar pistas
        clues = self._generate_clues(size, n_clues, seed)
        
        # Crear ArcEngine
        engine = ArcEngine()
        
        # Añadir variables (celdas del cuadrado)
        for row in range(size):
            for col in range(size):
                var_name = f'M_{row}_{col}'
                
                # Si hay pista, dominio de un solo valor
                if (row, col) in clues:
                    domain = [clues[(row, col)]]
                else:
                    # Dominio: todos los números de 1 a n²
                    domain = list(range(1, size * size + 1))
                
                engine.add_variable(var_name, domain)
        
        # Calcular suma mágica
        magic_sum = self._calculate_magic_sum(size)
        
        # Añadir restricciones de AllDifferent (todos los números únicos)
        all_variables = [f'M_{row}_{col}' for row in range(size) for col in range(size)]
        
        # Restricciones binarias de diferencia
        for i in range(len(all_variables)):
            for j in range(i + 1, len(all_variables)):
                var_i = all_variables[i]
                var_j = all_variables[j]
                
                def neq_constraint(v1, v2):
                    return v1 != v2
                
                constraint_id = f'neq_{var_i}_{var_j}'
                engine.add_constraint(var_i, var_j, neq_constraint, cid=constraint_id)

        # Nota: Las restricciones de suma (filas, columnas, diagonales) son n-arias.
        # El ArcEngine actual solo soporta restricciones binarias directamente.
        # Para implementar sumas, necesitaríamos un tipo de restricción más complejo
        # o descomponerlas en restricciones binarias auxiliares, lo cual es no trivial.
        # Por ahora, el validador de solución se encargará de verificar las sumas.
        # Esto significa que el ArcEngine propagará solo las restricciones AllDifferent.
        # La resolución completa del Magic Square requeriría un solver de orden superior
        # que pueda manejar restricciones n-arias o una transformación a binarias.
        logger.warning("Las restricciones de suma para Magic Square no se añaden al ArcEngine directamente. "
                       "Solo se añaden las restricciones AllDifferent. La validación final se realiza en validate_solution.")
        
        logger.info(f"Generado Magic Square {size}×{size} con {n_clues} pistas (suma mágica = {magic_sum})")
        
        return engine
    
    def validate_solution(self, solution: Dict[str, int], **params) -> bool:
        """
        Valida una solución del problema de cuadrado mágico.
        
        Args:
            solution: Diccionario {f'M_{i}_{j}': valor}
            **params: Parámetros del problema
        
        Returns:
            True si la solución es válida, False en caso contrario
        """
        size = params['size']
        
        # Convertir a matriz
        matrix = []
        for row in range(size):
            row_values = []
            for col in range(size):
                var_name = f'M_{row}_{col}'
                if var_name not in solution:
                    logger.debug(f"Falta celda: {var_name}")
                    return False
                row_values.append(solution[var_name])
            matrix.append(row_values)
        
        # Usar validador de magic square
        return validate_magic_square_solution(matrix, size)
    
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
        n_clues = params.get('n_clues', None)
        
        # Calcular número de pistas
        if n_clues is None:
            difficulty_ratios = {
                'empty': 0.0,
                'easy': 0.5,
                'medium': 0.3,
                'hard': 0.2,
                'expert': 0.1
            }
            n_clues = int(size * size * difficulty_ratios[difficulty])
        
        magic_sum = self._calculate_magic_sum(size)
        
        # Calcular número de restricciones
        # Filas: n
        # Columnas: n
        # Diagonales: 2
        # AllDifferent: n² * (n² - 1) / 2
        n_sum_constraints = size + size + 2
        n_diff_constraints = (size * size) * (size * size - 1) // 2
        n_constraints = n_sum_constraints + n_diff_constraints
        
        return {
            'family': self.name,
            'size': size,
            'n_variables': size * size,
            'domain_size': size * size,
            'difficulty': difficulty,
            'n_clues': n_clues,
            'fill_ratio': n_clues / (size * size),
            'magic_sum': magic_sum,
            'n_constraints': n_constraints,
            'complexity': 'NP-complete'
        }


# Auto-registrar la familia en el catálogo global
register_family(MagicSquareProblem())
logger.info("Familia MagicSquareProblem registrada en el catálogo global")

