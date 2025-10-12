"""
Clase base abstracta para familias de problemas CSP.

Este módulo define la interfaz que deben implementar todas las familias
de problemas en LatticeWeaver.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class ProblemFamily(ABC):
    """
    Clase base abstracta para familias de problemas CSP.
    
    Cada familia de problemas debe implementar:
    - generate(): Generar una instancia del problema
    - validate_solution(): Validar una solución
    - get_metadata(): Obtener metadatos del problema
    
    Attributes:
        name (str): Nombre único de la familia de problemas
        description (str): Descripción detallada de la familia
    """
    
    def __init__(self, name: str, description: str):
        """
        Inicializa la familia de problemas.
        
        Args:
            name: Nombre único de la familia (ej. 'nqueens', 'graph_coloring')
            description: Descripción detallada de qué tipo de problemas genera
        """
        self.name = name
        self.description = description
        logger.debug(f"Inicializada familia de problemas: {name}")
    
    @abstractmethod
    def generate(self, **params):
        """
        Genera una instancia del problema con los parámetros dados.
        
        Este método debe crear y configurar un ArcEngine con todas las
        variables, dominios y restricciones necesarias para representar
        el problema CSP.
        
        Args:
            **params: Parámetros específicos de la familia (ej. n=8 para 8-Queens)
            
        Returns:
            ArcEngine: Motor CSP configurado con el problema
            
        Raises:
            ValueError: Si los parámetros son inválidos
        """
        pass
    
    @abstractmethod
    def validate_solution(self, solution: Dict[str, Any], **params) -> bool:
        """
        Valida si una solución es correcta para el problema.
        
        Args:
            solution: Diccionario variable -> valor representando una solución
            **params: Parámetros del problema (deben coincidir con los de generate)
            
        Returns:
            bool: True si la solución es válida, False en caso contrario
        """
        pass
    
    @abstractmethod
    def get_metadata(self, **params) -> Dict[str, Any]:
        """
        Obtiene metadatos del problema generado con los parámetros dados.
        
        Los metadatos típicamente incluyen:
        - n_variables: Número de variables
        - n_constraints: Número de restricciones
        - complexity: Complejidad algorítmica (ej. 'O(n^2)')
        - domain_size: Tamaño promedio de dominios
        - problem_type: Tipo de problema (ej. 'combinatorial', 'scheduling')
        
        Args:
            **params: Parámetros del problema
            
        Returns:
            Dict con metadatos del problema
        """
        pass
    
    def get_default_params(self) -> Dict[str, Any]:
        """
        Retorna parámetros por defecto para esta familia.
        
        Útil para generar instancias de ejemplo sin especificar parámetros.
        
        Returns:
            Dict con parámetros por defecto
        """
        return {}
    
    def get_param_schema(self) -> Dict[str, Dict[str, Any]]:
        """
        Retorna el esquema de parámetros aceptados por esta familia.
        
        El esquema describe qué parámetros acepta generate(), sus tipos,
        rangos válidos y descripciones.
        
        Returns:
            Dict donde cada clave es un parámetro y el valor es un dict con:
            - type: Tipo del parámetro (int, float, str, etc.)
            - required: Si es obligatorio (bool)
            - default: Valor por defecto (si no es obligatorio)
            - min/max: Rango válido (para numéricos)
            - choices: Opciones válidas (para strings)
            - description: Descripción del parámetro
        
        Example:
            {
                'n': {
                    'type': int,
                    'required': True,
                    'min': 4,
                    'max': 1000,
                    'description': 'Tamaño del tablero (número de reinas)'
                }
            }
        """
        return {}
    
    def validate_params(self, **params) -> None:
        """
        Valida que los parámetros sean correctos según el esquema.
        
        Args:
            **params: Parámetros a validar
            
        Raises:
            ValueError: Si algún parámetro es inválido
        """
        schema = self.get_param_schema()
        
        # Verificar parámetros requeridos
        for param_name, param_info in schema.items():
            if param_info.get('required', False) and param_name not in params:
                raise ValueError(f"Parámetro requerido faltante: {param_name}")
        
        # Validar cada parámetro proporcionado
        for param_name, param_value in params.items():
            if param_name not in schema:
                logger.warning(f"Parámetro desconocido: {param_name}")
                continue
            
            param_info = schema[param_name]
            expected_type = param_info.get('type')
            
            # Validar tipo
            if expected_type and not isinstance(param_value, expected_type):
                raise ValueError(
                    f"Parámetro '{param_name}' debe ser de tipo {expected_type.__name__}, "
                    f"recibido {type(param_value).__name__}"
                )
            
            # Validar rango (para numéricos)
            if isinstance(param_value, (int, float)):
                if 'min' in param_info and param_value < param_info['min']:
                    raise ValueError(
                        f"Parámetro '{param_name}' debe ser >= {param_info['min']}, "
                        f"recibido {param_value}"
                    )
                if 'max' in param_info and param_value > param_info['max']:
                    raise ValueError(
                        f"Parámetro '{param_name}' debe ser <= {param_info['max']}, "
                        f"recibido {param_value}"
                    )
            
            # Validar opciones (para strings)
            if 'choices' in param_info and param_value not in param_info['choices']:
                raise ValueError(
                    f"Parámetro '{param_name}' debe ser uno de {param_info['choices']}, "
                    f"recibido '{param_value}'"
                )
    
    def __repr__(self) -> str:
        """Representación string de la familia."""
        return f"<ProblemFamily: {self.name}>"
    
    def __str__(self) -> str:
        """Descripción legible de la familia."""
        return f"{self.name}: {self.description}"

