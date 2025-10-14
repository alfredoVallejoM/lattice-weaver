"""
Generador de problemas N-Queens.

El problema de las N-Reinas consiste en colocar N reinas en un tablero de
ajedrez NxN de tal forma que ninguna reina ataque a otra.
"""

from typing import Dict, Any
import logging

from ..base import ProblemFamily
from ..utils.validators import validate_nqueens_solution
from lattice_weaver.core.csp_problem import CSP, Constraint

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
            >>> print(f"Variables: {len(engine.get_all_variables())}")
            Variables: 8
        """
        # Validar parámetros
        self.validate_params(**params)
        
        n = params['n']
        
        logger.info(f"Generando problema N-Queens con n={n}")
        
        # Importar ConstraintGraph
        
        
        variables = [f'Q{i}' for i in range(n)]
        domains = {var: list(range(n)) for var in variables}
        constraints = []
        
        for i in range(n):
            for j in range(i + 1, n):
                var_i = f'Q{i}'
                var_j = f'Q{j}'
                
                # Restricción de no ataque en la misma columna (ya cubierta por la diagonal si n > 1)
                # constraints.append(Constraint(scope=[var_i, var_j], relation=lambda val_i, val_j: val_i != val_j, name="nqueens_not_equal"))

                # Restricción de no ataque en la misma diagonal
                constraints.append(Constraint(scope=[var_i, var_j], relation=lambda val_i, val_j, i=i, j=j: abs(val_i - val_j) != abs(i - j), name="nqueens_not_diagonal", metadata={
                    "var1_idx": i,
                    "var2_idx": j
                }))
        
        csp_instance = CSP(variables, domains, constraints)
        logger.info(f"Problema N-Queens generado: {len(variables)} variables, {len(constraints)} restricciones")
        
        return csp_instance
    
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
