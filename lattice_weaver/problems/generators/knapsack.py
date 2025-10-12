"""
Generador de problemas Knapsack (0/1).

El problema de la mochila 0/1 consiste en seleccionar un subconjunto de items
para maximizar el valor total sin exceder la capacidad de la mochila.

Cada item tiene:
- Un peso (weight)
- Un valor (value)
- Una decisión binaria: incluir (1) o no incluir (0)

Restricción: suma de pesos ≤ capacidad
Objetivo: maximizar suma de valores

Este módulo implementa la generación de problemas de mochila
como problemas CSP usando el framework LatticeWeaver.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
import random

from lattice_weaver.arc_engine import ArcEngine
from lattice_weaver.problems.base import ProblemFamily
from lattice_weaver.problems.catalog import register_family

logger = logging.getLogger(__name__)


class KnapsackProblem(ProblemFamily):
    """
    Familia de problemas de mochila 0/1.
    
    Cada item puede ser incluido (1) o no incluido (0) en la mochila.
    La suma de pesos no debe exceder la capacidad.
    
    Attributes:
        name: Nombre de la familia ('knapsack')
        description: Descripción del problema
    """
    
    def __init__(self):
        """Inicializa la familia Knapsack."""
        super().__init__(
            name='knapsack',
            description='0/1 Knapsack problem: select items to maximize value '
                       'without exceeding capacity constraint'
        )
        logger.info(f"Inicializada familia de problemas: {self.name}")
    
    def get_default_params(self) -> Dict[str, Any]:
        """
        Retorna parámetros por defecto.
        
        Returns:
            Diccionario con parámetros por defecto
        """
        return {
            'n_items': 10,
            'capacity': None,  # Se calcula automáticamente
            'max_weight': 20,
            'max_value': 100,
            'seed': None
        }
    
    def get_param_schema(self) -> Dict[str, Dict[str, Any]]:
        """
        Retorna el esquema de parámetros.
        
        Returns:
            Diccionario con el esquema de parámetros
        """
        return {
            'n_items': {
                'type': int,
                'required': True,
                'min': 2,
                'max': 100,
                'description': 'Número de items disponibles'
            },
            'capacity': {
                'type': int,
                'required': False,
                'min': 1,
                'description': 'Capacidad de la mochila (se calcula automáticamente si no se especifica)'
            },
            'max_weight': {
                'type': int,
                'required': False,
                'min': 1,
                'max': 1000,
                'description': 'Peso máximo de un item'
            },
            'max_value': {
                'type': int,
                'required': False,
                'min': 1,
                'max': 10000,
                'description': 'Valor máximo de un item'
            },
            'weights': {
                'type': list,
                'required': False,
                'description': 'Lista de pesos de items (sobreescribe generación aleatoria)'
            },
            'choices': {
                'type': list,
                'required': False,
                'description': 'Lista de valores de items (sobreescribe generación aleatoria)'
            },
            'seed': {
                'type': int,
                'required': False,
                'description': 'Semilla para generación aleatoria reproducible'
            }
        }
    
    def _generate_items(
        self,
        n_items: int,
        max_weight: int,
        max_value: int,
        seed: Optional[int] = None
    ) -> Tuple[List[int], List[int]]:
        """
        Genera pesos y valores aleatorios para los items.
        
        Args:
            n_items: Número de items
            max_weight: Peso máximo de un item
            max_value: Valor máximo de un item
            seed: Semilla para reproducibilidad
        
        Returns:
            Tupla (weights, values)
        """
        if seed is not None:
            random.seed(seed)
        
        weights = [random.randint(1, max_weight) for _ in range(n_items)]
        values = [random.randint(1, max_value) for _ in range(n_items)]
        
        return weights, values
    
    def generate(self, **params) -> ArcEngine:
        """
        Genera un problema de mochila 0/1.
        
        Args:
            **params: Parámetros del problema
        
        Returns:
            ArcEngine configurado con el problema
        """
        # Validar parámetros
        self.validate_params(**params)
        
        n_items = params['n_items']
        capacity = params.get('capacity', None)
        max_weight = params.get('max_weight', 20)
        max_value = params.get('max_value', 100)
        seed = params.get('seed', None)
        
        # Obtener pesos y valores
        if 'weights' in params and 'values' in params:
            weights = params['weights']
            values = params['values']
            
            if len(weights) != n_items or len(values) != n_items:
                raise ValueError(f"Longitud de weights y values debe ser {n_items}")
        else:
            weights, values = self._generate_items(n_items, max_weight, max_value, seed)
        
        # Calcular capacidad si no se especifica
        if capacity is None:
            # Capacidad = 50% de la suma total de pesos
            capacity = sum(weights) // 2
        
        # Crear ArcEngine
        engine = ArcEngine()
        
        # Añadir variables (decisión binaria para cada item)
        for i in range(n_items):
            var_name = f'item_{i}'
            domain = [0, 1]  # 0 = no incluir, 1 = incluir
            engine.add_variable(var_name, domain)
        
        # Nota: La restricción de capacidad es n-aria.
        # El ArcEngine actual solo soporta restricciones binarias directamente.
        # Para implementar esta restricción, necesitaríamos un tipo de restricción más complejo
        # o descomponerla en restricciones binarias auxiliares, lo cual es no trivial.
        # Por ahora, el validador de solución se encargará de verificar la capacidad.
        # Esto significa que el ArcEngine no propagará esta restricción.
        # La resolución completa del Knapsack requeriría un solver de orden superior
        # que pueda manejar restricciones n-arias o una transformación a binarias.
        logger.warning("La restricción de capacidad para Knapsack no se añade al ArcEngine directamente. "
                       "La validación final se realiza en validate_solution.")
        
        logger.info(
            f"Generado Knapsack con {n_items} items, "
            f"capacidad = {capacity}, "
            f"peso total = {sum(weights)}, "
            f"valor total = {sum(values)}"
        )
        
        # Almacenar metadatos en el engine para uso posterior
        engine._knapsack_weights = weights
        engine._knapsack_values = values
        engine._knapsack_capacity = capacity
        
        return engine
    
    def validate_solution(self, solution: Dict[str, int], **params) -> bool:
        """
        Valida una solución del problema de mochila.
        
        Args:
            solution: Diccionario {f'item_{i}': decision} donde decision ∈ {0, 1}
            **params: Parámetros del problema
        
        Returns:
            True si la solución es válida, False en caso contrario
        """
        n_items = params['n_items']
        capacity = params.get('capacity', None)
        
        # Obtener pesos
        if 'weights' in params:
            weights = params['weights']
        else:
            max_weight = params.get('max_weight', 20)
            max_value = params.get('max_value', 100)
            seed = params.get('seed', None)
            weights, _ = self._generate_items(n_items, max_weight, max_value, seed)
        
        # Calcular capacidad si no se especifica
        if capacity is None:
            capacity = sum(weights) // 2
        
        # Verificar que todas las decisiones estén presentes
        if len(solution) != n_items:
            logger.debug(f"Número incorrecto de items: {len(solution)} != {n_items}")
            return False
        
        # Verificar que las decisiones sean binarias
        for i in range(n_items):
            var_name = f'item_{i}'
            if var_name not in solution:
                logger.debug(f"Falta item: {var_name}")
                return False
            
            decision = solution[var_name]
            if decision not in [0, 1]:
                logger.debug(f"Decisión inválida para {var_name}: {decision}")
                return False
        
        # Verificar restricción de capacidad
        total_weight = sum(
            weights[i] * solution[f'item_{i}']
            for i in range(n_items)
        )
        
        if total_weight > capacity:
            logger.debug(
                f"Capacidad excedida: {total_weight} > {capacity}"
            )
            return False
        
        return True
    
    def get_metadata(self, **params) -> Dict[str, Any]:
        """
        Retorna metadatos del problema.
        
        Args:
            **params: Parámetros del problema
        
        Returns:
            Diccionario con metadatos
        """
        n_items = params['n_items']
        capacity = params.get('capacity', None)
        max_weight = params.get('max_weight', 20)
        max_value = params.get('max_value', 100)
        seed = params.get('seed', None)
        
        # Obtener pesos y valores
        if 'weights' in params and 'values' in params:
            weights = params['weights']
            values = params['values']
        else:
            weights, values = self._generate_items(n_items, max_weight, max_value, seed)
        
        # Calcular capacidad si no se especifica
        if capacity is None:
            capacity = sum(weights) // 2
        
        total_weight = sum(weights)
        total_value = sum(values)
        
        # Calcular dificultad basada en el ratio capacidad/peso_total
        capacity_ratio = capacity / total_weight
        if capacity_ratio < 0.3:
            difficulty = 'hard'
        elif capacity_ratio < 0.5:
            difficulty = 'medium'
        else:
            difficulty = 'easy'
        
        return {
            'family': self.name,
            'n_items': n_items,
            'n_variables': n_items,
            'domain_size': 2,  # Binario
            'capacity': capacity,
            'total_weight': total_weight,
            'total_value': total_value,
            'capacity_ratio': capacity_ratio,
            'difficulty': difficulty,
            'n_constraints': 1,  # Solo restricción de capacidad
            'complexity': 'NP-complete'
        }


# Auto-registrar la familia en el catálogo global
register_family(KnapsackProblem())
logger.info("Familia KnapsackProblem registrada en el catálogo global")

