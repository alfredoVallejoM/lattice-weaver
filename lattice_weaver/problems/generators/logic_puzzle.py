"""
Generador de Logic Puzzles (estilo Zebra Puzzle).

Los puzzles lógicos tipo Zebra consisten en asignar atributos a entidades
basándose en un conjunto de pistas lógicas.

Ejemplo clásico (Zebra Puzzle):
- 5 casas en fila
- Cada casa tiene: color, nacionalidad, bebida, cigarrillo, mascota
- Pistas como "El noruego vive en la primera casa"
- Pregunta: ¿Quién tiene la zebra?

Este módulo implementa la generación de puzzles lógicos
como problemas CSP usando el framework LatticeWeaver.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
import random

from lattice_weaver.core.csp_problem import CSP, Constraint
from lattice_weaver.problems.base import ProblemFamily
from lattice_weaver.problems.catalog import register_family

logger = logging.getLogger(__name__)


class LogicPuzzleProblem(ProblemFamily):
    """
    Familia de problemas de puzzles lógicos (estilo Zebra).
    
    Asigna atributos a entidades basándose en restricciones lógicas.
    
    Attributes:
        name: Nombre de la familia ('logic_puzzle')
        description: Descripción del problema
    """
    
    def __init__(self):
        """Inicializa la familia Logic Puzzle."""
        super().__init__(
            name='logic_puzzle',
            description='Logic Puzzle (Zebra-style): assign attributes to entities '
                       'based on logical constraints'
        )
        logger.info(f"Inicializada familia de problemas: {self.name}")
    
    def get_default_params(self) -> Dict[str, Any]:
        """
        Retorna parámetros por defecto.
        
        Returns:
            Diccionario con parámetros por defecto
        """
        return {
            'puzzle_type': 'simple',
            'n_entities': 5,
            'n_attributes': 3,
            'seed': None
        }
    
    def get_param_schema(self) -> Dict[str, Dict[str, Any]]:
        """
        Retorna el esquema de parámetros.
        
        Returns:
            Diccionario con el esquema de parámetros
        """
        return {
            'puzzle_type': {
                'type': str,
                'required': False,
                'choices': ['simple', 'zebra', 'custom'],
                'description': 'Tipo de puzzle (simple, zebra clásico, o custom)'
            },
            'n_entities': {
                'type': int,
                'required': False,
                'min': 3,
                'max': 10,
                'description': 'Número de entidades (ej. casas)'
            },
            'n_attributes': {
                'type': int,
                'required': False,
                'min': 2,
                'max': 6,
                'description': 'Número de categorías de atributos'
            },
            'attributes': {
                'type': dict,
                'required': False,
                'description': 'Diccionario {categoría: [valores]} para puzzle custom'
            },
            'seed': {
                'type': int,
                'required': False,
                'description': 'Semilla para generación aleatoria reproducible'
            }
        }
    
    def _get_zebra_attributes(self) -> Dict[str, List[str]]:
        """
        Retorna los atributos del Zebra Puzzle clásico.
        
        Returns:
            Diccionario {categoría: [valores]}
        """
        return {
            'color': ['red', 'green', 'white', 'yellow', 'blue'],
            'nationality': ['english', 'spanish', 'ukrainian', 'norwegian', 'japanese'],
            'drink': ['coffee', 'tea', 'milk', 'orange_juice', 'water'],
            'smoke': ['old_gold', 'kools', 'chesterfields', 'lucky_strike', 'parliaments'],
            'pet': ['dog', 'snails', 'fox', 'horse', 'zebra']
        }
    
    def _get_simple_attributes(self, n_entities: int, n_attributes: int) -> Dict[str, List[str]]:
        """
        Genera atributos simples para un puzzle genérico.
        
        Args:
            n_entities: Número de entidades
            n_attributes: Número de categorías de atributos
        
        Returns:
            Diccionario {categoría: [valores]}
        """
        categories = ['color', 'shape', 'size', 'position', 'number', 'letter']
        
        attributes = {}
        for i in range(n_attributes):
            category = categories[i]
            values = [f'{category}_{j}' for j in range(n_entities)]
            attributes[category] = values
        
        return attributes
    
    def generate(self, **params) -> CSP:
        """
        Genera un problema de logic puzzle.
        
        Args:
            **params: Parámetros del problema
        
        Returns:
            ArcEngine configurado con el problema
        """
        # Validar parámetros
        self.validate_params(**params)
        
        puzzle_type = params.get('puzzle_type', 'simple')
        n_entities = params.get('n_entities', 5)
        n_attributes = params.get('n_attributes', 3)
        seed = params.get('seed', None)
        
        # Obtener atributos según el tipo de puzzle
        if puzzle_type == 'zebra':
            attributes = self._get_zebra_attributes()
            n_entities = 5  # Zebra puzzle siempre tiene 5 casas
        elif puzzle_type == 'custom' and 'attributes' in params:
            attributes = params['attributes']
            # Verificar que todas las categorías tengan n_entities valores
            for category, values in attributes.items():
                if len(values) != n_entities:
                    raise ValueError(
                        f"Categoría '{category}' debe tener {n_entities} valores, "
                        f"tiene {len(values)}"
                    )
        else:
            attributes = self._get_simple_attributes(n_entities, n_attributes)
        
        # Crear ArcEngine
        csp_problem = CSP(variables=set(), domains={}, constraints=[], name=f"LogicPuzzle_{puzzle_type}")
        
        # Añadir variables
        # Para cada categoría de atributo, cada entidad tiene una variable
        # que indica qué valor de esa categoría tiene
        
        # Representación: variable_{category}_{entity} = value_index
        # Ejemplo: variable_color_0 = 2 significa "entidad 0 tiene color 2"
        
        for category, values in attributes.items():
            for entity_id in range(n_entities):
                var_name = f'{category}_{entity_id}'
                # Dominio: índices de los valores posibles
                domain = list(range(len(values)))
                csp_problem.add_variable(var_name, domain)
        
        # Añadir restricciones de AllDifferent por categoría
        # Cada valor de una categoría debe asignarse a exactamente una entidad
        for category, values in attributes.items():
            variables = [f'{category}_{entity_id}' for entity_id in range(n_entities)]
            
            # Restricciones binarias de diferencia
            for i in range(len(variables)):
                for j in range(i + 1, len(variables)):
                    var_i = variables[i]
                    var_j = variables[j]
                    
                    def neq_constraint(v1, v2):
                        return v1 != v2
                    
                    constraint_id = f'neq_{var_i}_{var_j}'
                    csp_problem.add_constraint(Constraint(scope=frozenset({var_i, var_j}), relation=neq_constraint, name=constraint_id))

        
        # Añadir algunas pistas específicas para Zebra puzzle
        if puzzle_type == 'zebra':
            # Pista 1: El noruego vive en la primera casa
            # nationality_0 = 3 (norwegian es el índice 3)
            # Esto se implementa reduciendo el dominio de la variable
            csp_problem.domains["nationality_0"] = {3}
            logger.debug("Pista aplicada: noruego en casa 0")
            
            # Pista 2: La casa verde está inmediatamente a la derecha de la blanca
            # Si color_i = 2 (white), entonces color_{i+1} = 1 (green)
            for i in range(n_entities - 1):
                def green_right_of_white(color_i, color_next):
                    if color_i == 2:  # white
                        return color_next == 1  # green
                    return True
                
                csp_problem.add_constraint(Constraint(
                    scope=frozenset({f'color_{i}', f'color_{i+1}'}),
                    relation=green_right_of_white,
                    name=f'clue_green_right_white_{i}'
                ))
            
            # Pista 3: Se bebe café en la casa verde
            # Si color_i = 1 (green), entonces drink_i = 0 (coffee)
            for i in range(n_entities):
                def coffee_in_green(color, drink):
                    if color == 1:  # green
                        return drink == 0  # coffee
                    return True
                
                csp_problem.add_constraint(Constraint(
                    scope=frozenset({f'color_{i}', f'drink_{i}'}),
                    relation=coffee_in_green,
                    name=f'clue_coffee_in_green_{i}'
                ))
        
        logger.info(
            f"Generado Logic Puzzle tipo '{puzzle_type}' con "
            f"{n_entities} entidades y {len(attributes)} categorías"
        )
        
        # Almacenar metadatos en el engine
        csp_problem.metadata["logic_puzzle_attributes"] = attributes
        csp_problem.metadata["logic_puzzle_type"] = puzzle_type
        
        return csp_problem
    
    def validate_solution(self, solution: Dict[str, int], **params) -> bool:
        """
        Valida una solución del logic puzzle.
        
        Args:
            solution: Diccionario {f'{category}_{entity_id}': value_index}
            **params: Parámetros del problema
        
        Returns:
            True si la solución es válida, False en caso contrario
        """
        puzzle_type = params.get('puzzle_type', 'simple')
        n_entities = params.get('n_entities', 5)
        n_attributes = params.get('n_attributes', 3)
        
        # Obtener atributos
        if puzzle_type == 'zebra':
            attributes = self._get_zebra_attributes()
            n_entities = 5
        elif puzzle_type == 'custom' and 'attributes' in params:
            attributes = params['attributes']
        else:
            attributes = self._get_simple_attributes(n_entities, n_attributes)
        
        # Verificar que todas las variables estén presentes
        expected_vars = n_entities * len(attributes)
        if len(solution) != expected_vars:
            logger.debug(
                f"Número incorrecto de variables: {len(solution)} != {expected_vars}"
            )
            return False
        
        # Verificar AllDifferent por categoría
        for category, values in attributes.items():
            assigned_values = []
            for entity_id in range(n_entities):
                var_name = f'{category}_{entity_id}'
                if var_name not in solution:
                    logger.debug(f"Falta variable: {var_name}")
                    return False
                
                value_idx = solution[var_name]
                
                # Verificar rango
                if not 0 <= value_idx < len(values):
                    logger.debug(
                        f"Valor fuera de rango para {var_name}: {value_idx}"
                    )
                    return False
                
                assigned_values.append(value_idx)
            
            # Verificar que todos los valores sean diferentes
            if len(set(assigned_values)) != len(assigned_values):
                logger.debug(f"Valores repetidos en categoría {category}")
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
        puzzle_type = params.get('puzzle_type', 'simple')
        n_entities = params.get('n_entities', 5)
        n_attributes = params.get('n_attributes', 3)
        
        # Obtener atributos
        if puzzle_type == 'zebra':
            attributes = self._get_zebra_attributes()
            n_entities = 5
        elif puzzle_type == 'custom' and 'attributes' in params:
            attributes = params['attributes']
        else:
            attributes = self._get_simple_attributes(n_entities, n_attributes)
        
        n_categories = len(attributes)
        n_variables = n_entities * n_categories
        
        # Calcular número de restricciones
        # AllDifferent por categoría: n_categories * (n_entities * (n_entities - 1) / 2)
        n_diff_constraints = n_categories * (n_entities * (n_entities - 1) // 2)
        
        # Pistas adicionales para Zebra
        n_clue_constraints = 0
        if puzzle_type == 'zebra':
            n_clue_constraints = 1 + (n_entities - 1) + n_entities  # 3 pistas básicas
        
        n_constraints = n_diff_constraints + n_clue_constraints
        
        # Dificultad basada en número de entidades y categorías
        if n_entities <= 3:
            difficulty = 'easy'
        elif n_entities <= 5:
            difficulty = 'medium'
        else:
            difficulty = 'hard'
        
        return {
            'family': self.name,
            'puzzle_type': puzzle_type,
            'n_entities': n_entities,
            'n_categories': n_categories,
            'n_variables': n_variables,
            'domain_size': n_entities,
            'n_constraints': n_constraints,
            'difficulty': difficulty,
            'complexity': 'NP-complete'
        }


# Auto-registrar la familia en el catálogo global
register_family(LogicPuzzleProblem())
logger.info("Familia LogicPuzzleProblem registrada en el catálogo global")

