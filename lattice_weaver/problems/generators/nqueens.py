"""
Generador de problemas N-Queens.

El problema de las N-Reinas consiste en colocar N reinas en un tablero de
ajedrez NxN de tal forma que ninguna reina ataque a otra.
"""

from typing import Dict, Any
import logging

from ..base import ProblemFamily
from ..utils.validators import validate_nqueens_solution

logger = logging.getLogger(__name__)


class NQueensProblem(ProblemFamily):
    """
    Familia de problemas N-Queens.
    
    Genera instancias del clásico problema de las N-Reinas donde se deben
    colocar N reinas en un tablero NxN sin que se ataquen entre sí.
    
    Variables: Q0, Q1, ..., Q(n-1) donde Qi representa la columna de la reina en la fila i
    Dominios: [0, n-1] para cada variable
    Restricciones:
        - No dos reinas en la misma columna: Qi != Qj para i != j
        - No dos reinas en la misma diagonal: |i - j| != |Qi - Qj|
    """
    
    def __init__(self):
        """Inicializa la familia N-Queens."""
        super().__init__(
            name='nqueens',
            description='Problema de las N-Reinas: colocar N reinas en un tablero NxN sin que se ataquen'
        )
    
    def get_param_schema(self) -> Dict[str, Dict[str, Any]]:
        """
        Retorna el esquema de parámetros para N-Queens.
        
        Returns:
            Dict con esquema de parámetros
        """
        return {
            'n': {
                'type': int,
                'required': True,
                'min': 4,
                'max': 1000,
                'description': 'Tamaño del tablero (número de reinas)'
            }
        }
    
    def get_default_params(self) -> Dict[str, Any]:
        """
        Retorna parámetros por defecto para N-Queens.
        
        Returns:
            Dict con n=8 (problema clásico de 8 reinas)
        """
        return {'n': 8}
    
    def generate(self, **params):
        """
        Genera una instancia del problema N-Queens.
        
        Args:
            n: Tamaño del tablero (número de reinas)
            
        Returns:
            ArcEngine: Motor CSP configurado con el problema N-Queens
            
        Raises:
            ValueError: Si n < 4 o n > 1000
            
        Example:
            >>> from lattice_weaver.problems.generators.nqueens import NQueensProblem
            >>> family = NQueensProblem()
            >>> engine = family.generate(n=8)
            >>> print(f"Variables: {len(engine.variables)}")
            Variables: 8
        """
        # Validar parámetros
        self.validate_params(**params)
        
        n = params['n']
        
        logger.info(f"Generando problema N-Queens con n={n}")
        
        # Importar ArcEngine
        from lattice_weaver.arc_engine import ArcEngine
        
        # Crear motor CSP
        engine = ArcEngine()
        
        # Añadir variables (una por fila, valor = columna)
        for i in range(n):
            var_name = f'Q{i}'
            domain = list(range(n))  # Columnas posibles [0, n-1]
            engine.add_variable(var_name, domain)
            logger.debug(f"Añadida variable {var_name} con dominio {domain}")
        
        # Añadir restricciones
        constraint_count = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                var_i = f'Q{i}'
                var_j = f'Q{j}'
                
                # Restricción: no misma columna Y no misma diagonal
                def nqueens_constraint(col_i, col_j, metadata: Dict[str, Any]) -> bool:
                    row_i = metadata["var1_idx"]
                    row_j = metadata["var2_idx"]
                    """
                    Restricción N-Queens: dos reinas no pueden estar en la misma
                    columna ni en la misma diagonal.
                    
                    Args:
                        col_i: Columna de la reina en fila row_i
                        col_j: Columna de la reina en fila row_j
                        row_i: Fila de la primera reina (capturado del closure)
                        row_j: Fila de la segunda reina (capturado del closure)
                    
                    Returns:
                        bool: True si las reinas no se atacan
                    """
                    # No misma columna
                    if col_i == col_j:
                        return False
                    
                    # No misma diagonal
                    if abs(row_i - row_j) == abs(col_i - col_j):
                        return False
                    
                    return True
                
                # Registrar la función de restricción con un nombre único
                relation_name = f'nqueens_constraint_{i}_{j}'
                from lattice_weaver.arc_engine.constraints import register_relation
                register_relation(relation_name, nqueens_constraint)
                engine.add_constraint(var_i, var_j, relation_name, metadata={'var1_idx': i, 'var2_idx': j})
                constraint_count += 1
                logger.debug(f"Añadida restricción entre {var_i} y {var_j}")
        
        logger.info(f"Problema N-Queens generado: {n} variables, {constraint_count} restricciones")
        
        return engine
    
    def validate_solution(self, solution: Dict[str, Any], **params) -> bool:
        """
        Valida si una solución es correcta para el problema N-Queens.
        
        Args:
            solution: Diccionario {f'Q{i}': columna} para i en [0, n)
            **params: Debe incluir 'n' (tamaño del tablero)
            
        Returns:
            bool: True si la solución es válida
            
        Example:
            >>> family = NQueensProblem()
            >>> solution = {'Q0': 0, 'Q1': 4, 'Q2': 7, 'Q3': 5, 
            ...             'Q4': 2, 'Q5': 6, 'Q6': 1, 'Q7': 3}
            >>> family.validate_solution(solution, n=8)
            True
        """
        self.validate_params(**params)
        n = params['n']
        
        return validate_nqueens_solution(solution, n)
    
    def get_metadata(self, **params) -> Dict[str, Any]:
        """
        Obtiene metadatos del problema N-Queens.
        
        Args:
            **params: Debe incluir 'n' (tamaño del tablero)
            
        Returns:
            Dict con metadatos del problema
            
        Example:
            >>> family = NQueensProblem()
            >>> metadata = family.get_metadata(n=8)
            >>> print(metadata['n_variables'])
            8
        """
        self.validate_params(**params)
        n = params['n']
        
        n_variables = n
        n_constraints = n * (n - 1) // 2  # Combinaciones de 2 reinas
        
        return {
            'family': self.name,
            'n': n,
            'n_variables': n_variables,
            'n_constraints': n_constraints,
            'domain_size': n,
            'complexity': 'O(n^2)',
            'problem_type': 'combinatorial',
            'description': f'{n}-Queens problem on {n}x{n} board',
            'difficulty': self._estimate_difficulty(n)
        }
    
    def _estimate_difficulty(self, n: int) -> str:
        """
        Estima la dificultad del problema basándose en n.
        
        Args:
            n: Tamaño del tablero
            
        Returns:
            str: Nivel de dificultad ('easy', 'medium', 'hard', 'very_hard')
        """
        if n <= 8:
            return 'easy'
        elif n <= 16:
            return 'medium'
        elif n <= 50:
            return 'hard'
        else:
            return 'very_hard'


# Auto-registro en el catálogo global al importar
def _register():
    """Registra NQueensProblem en el catálogo global."""
    try:
        from ..catalog import register_family
        register_family(NQueensProblem())
        logger.info("NQueensProblem registrado en el catálogo")
    except Exception as e:
        logger.warning(f"No se pudo auto-registrar NQueensProblem: {e}")

_register()

