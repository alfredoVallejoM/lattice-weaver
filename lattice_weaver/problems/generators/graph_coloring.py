"""
Generador de problemas Graph Coloring.

El problema de coloración de grafos consiste en asignar colores a los nodos
de un grafo de tal forma que nodos adyacentes tengan colores diferentes.
"""

from typing import Dict, Any, List, Tuple
import logging

from ..base import ProblemFamily
from lattice_weaver.core.csp_engine.graph import ConstraintGraph
from lattice_weaver.core.csp_engine.constraints import NE
from ..utils.validators import validate_graph_coloring_solution
from ..utils.graph_generators import (
    generate_random_graph,
    generate_complete_graph,
    generate_bipartite_graph,
    generate_grid_graph,
    generate_cycle_graph,
    generate_path_graph,
    generate_star_graph,
    generate_wheel_graph,
    get_graph_chromatic_number_lower_bound
)

logger = logging.getLogger(__name__)


class GraphColoringProblem(ProblemFamily):
    """
    Familia de problemas Graph Coloring.
    
    Genera instancias del problema de coloración de grafos donde se deben
    asignar colores a nodos tal que nodos adyacentes tengan colores diferentes.
    
    Variables: V0, V1, ..., V(n-1) donde Vi representa el color del nodo i
    Dominios: [0, k-1] donde k es el número de colores disponibles
    Restricciones: Para cada arista (i, j): color[i] != color[j]
    """
    
    def __init__(self):
        """Inicializa la familia Graph Coloring."""
        super().__init__(
            name='graph_coloring',
            description='Problema de coloración de grafos: asignar colores a nodos sin conflictos'
        )
    
    def get_param_schema(self) -> Dict[str, Dict[str, Any]]:
        """
        Retorna el esquema de parámetros para Graph Coloring.
        
        Returns:
            Dict con esquema de parámetros
        """
        return {
            'graph_type': {
                'type': str,
                'required': False,
                'default': 'random',
                'choices': ['random', 'complete', 'bipartite', 'grid', 'cycle', 
                           'path', 'star', 'wheel'],
                'description': 'Tipo de grafo a generar'
            },
            'n_nodes': {
                'type': int,
                'required': True,
                'min': 2,
                'max': 1000,
                'description': 'Número de nodos en el grafo'
            },
            'n_colors': {
                'type': int,
                'required': True,
                'min': 2,
                'max': 20,
                'description': 'Número de colores disponibles'
            },
            'edge_probability': {
                'type': float,
                'required': False,
                'default': 0.3,
                'min': 0.0,
                'max': 1.0,
                'description': 'Probabilidad de arista (solo para grafos aleatorios)'
            },
            'grid_rows': {
                'type': int,
                'required': False,
                'min': 2,
                'max': 100,
                'description': 'Número de filas (solo para grafos grid)'
            },
            'grid_cols': {
                'type': int,
                'required': False,
                'min': 2,
                'max': 100,
                'description': 'Número de columnas (solo para grafos grid)'
            },
            'seed': {
                'type': int,
                'required': False,
                'description': 'Semilla para reproducibilidad (grafos aleatorios)'
            }
        }
    
    def get_default_params(self) -> Dict[str, Any]:
        """
        Retorna parámetros por defecto para Graph Coloring.
        
        Returns:
            Dict con parámetros por defecto (grafo aleatorio de 10 nodos, 3 colores)
        """
        return {
            'graph_type': 'random',
            'n_nodes': 10,
            'n_colors': 3,
            'edge_probability': 0.3
        }
    
    def _generate_edges(self, **params) -> List[Tuple[int, int]]:
        """
        Genera las aristas del grafo según el tipo especificado.
        
        Args:
            **params: Parámetros del problema
            
        Returns:
            Lista de aristas como tuplas (i, j)
        """
        graph_type = params.get('graph_type', 'random')
        n_nodes = params['n_nodes']
        seed = params.get('seed')
        
        if graph_type == 'random':
            edge_prob = params.get('edge_probability', 0.3)
            return generate_random_graph(n_nodes, edge_prob, seed)
        
        elif graph_type == 'complete':
            return generate_complete_graph(n_nodes)
        
        elif graph_type == 'bipartite':
            n_left = n_nodes // 2
            n_right = n_nodes - n_left
            edge_prob = params.get('edge_probability', 0.5)
            return generate_bipartite_graph(n_left, n_right, edge_prob, seed)
        
        elif graph_type == 'grid':
            if 'grid_rows' in params and 'grid_cols' in params:
                rows = params['grid_rows']
                cols = params['grid_cols']
            else:
                # Calcular grid cuadrado aproximado
                import math
                rows = int(math.sqrt(n_nodes))
                cols = (n_nodes + rows - 1) // rows
            return generate_grid_graph(rows, cols)
        
        elif graph_type == 'cycle':
            return generate_cycle_graph(n_nodes)
        
        elif graph_type == 'path':
            return generate_path_graph(n_nodes)
        
        elif graph_type == 'star':
            return generate_star_graph(n_nodes)
        
        elif graph_type == 'wheel':
            return generate_wheel_graph(n_nodes)
        
        else:
            raise ValueError(f"Tipo de grafo desconocido: {graph_type}")
    
    def generate(self, **params):
        """
        Genera una instancia del problema Graph Coloring.
        
        Args:
            graph_type: Tipo de grafo ('random', 'complete', etc.)
            n_nodes: Número de nodos
            n_colors: Número de colores disponibles
            edge_probability: Probabilidad de arista (para grafos aleatorios)
            seed: Semilla para reproducibilidad (opcional)
            
        Returns:
            ArcEngine: Motor CSP configurado con el problema Graph Coloring
            
        Example:
            >>> from lattice_weaver.problems.generators.graph_coloring import GraphColoringProblem
            >>> family = GraphColoringProblem()
            >>> engine = family.generate(graph_type='cycle', n_nodes=5, n_colors=3)
            >>> print(f"Variables: {len(engine.variables)}")
            Variables: 5
        """
        # Validar parámetros
        self.validate_params(**params)
        
        n_nodes = params['n_nodes']
        n_colors = params['n_colors']
        graph_type = params.get('graph_type', 'random')
        
        logger.info(f"Generando problema Graph Coloring: {graph_type} con {n_nodes} nodos, {n_colors} colores")
        
        # Generar aristas del grafo
        edges = self._generate_edges(**params)
        
        # Crear ConstraintGraph
        cg = ConstraintGraph()
        
        # Añadir variables (una por nodo, valor = color)
        for i in range(n_nodes):
            var_name = f'V{i}'
            domain = list(range(n_colors))  # Colores disponibles [0, n_colors-1]
            cg.add_variable(var_name, set(domain)) # Convertir a set
            logger.debug(f"Añadida variable {var_name} con dominio {domain}")
        
        # Añadir restricciones (una por arista)
        for i, j in edges:
            var_i = f'V{i}'
            var_j = f'V{j}'
            
            # Restricción: colores diferentes
            cg.add_constraint(var_i, var_j, NE())
            logger.debug(f"Añadida restricción entre {var_i} y {var_j}")
        
        # Guardar las aristas en el ConstraintGraph para validación posterior
        cg._graph_coloring_edges = edges
        
        logger.info(f"Problema Graph Coloring generado: {n_nodes} variables, {len(edges)} restricciones")
        
        return cg
    
    def validate_solution(self, solution: Dict[str, Any], **params) -> bool:
        """
        Valida si una solución es correcta para el problema Graph Coloring.
        
        Args:
            solution: Diccionario {f'V{i}': color} para i en [0, n_nodes)
            **params: Debe incluir 'n_nodes' y los parámetros del grafo
            
        Returns:
            bool: True si la solución es válida
        """
        self.validate_params(**params)
        
        n_nodes = params['n_nodes']
        edges = self._generate_edges(**params)
        
        return validate_graph_coloring_solution(solution, edges, n_nodes)
    
    def get_metadata(self, **params) -> Dict[str, Any]:
        """
        Obtiene metadatos del problema Graph Coloring.
        
        Args:
            **params: Parámetros del problema
            
        Returns:
            Dict con metadatos del problema
        """
        self.validate_params(**params)
        
        n_nodes = params['n_nodes']
        n_colors = params['n_colors']
        graph_type = params.get('graph_type', 'random')
        
        edges = self._generate_edges(**params)
        n_edges = len(edges)
        
        # Calcular cota inferior del número cromático
        chromatic_lower_bound = get_graph_chromatic_number_lower_bound(edges, n_nodes)
        
        return {
            'family': self.name,
            'graph_type': graph_type,
            'n_nodes': n_nodes,
            'n_colors': n_colors,
            'n_variables': n_nodes,
            'n_constraints': n_edges,
            'n_edges': n_edges,
            'domain_size': n_colors,
            'complexity': 'O(|E|)',
            'problem_type': 'graph_coloring',
            'chromatic_lower_bound': chromatic_lower_bound,
            'description': f'{graph_type} graph with {n_nodes} nodes, {n_colors} colors',
            'difficulty': self._estimate_difficulty(n_nodes, n_edges, n_colors)
        }
    
    def _estimate_difficulty(self, n_nodes: int, n_edges: int, n_colors: int) -> str:
        """
        Estima la dificultad del problema.
        
        Args:
            n_nodes: Número de nodos
            n_edges: Número de aristas
            n_colors: Número de colores
            
        Returns:
            str: Nivel de dificultad
        """
        density = n_edges / (n_nodes * (n_nodes - 1) / 2) if n_nodes > 1 else 0
        
        if n_nodes <= 10 and density < 0.3:
            return 'easy'
        elif n_nodes <= 50 and density < 0.5:
            return 'medium'
        elif n_nodes <= 100:
            return 'hard'
        else:
            return 'very_hard'


# Auto-registro en el catálogo global
def _register():
    """Registra GraphColoringProblem en el catálogo global."""
    try:
        from ..catalog import register_family
        register_family(GraphColoringProblem())
        logger.info("GraphColoringProblem registrado en el catálogo")
    except Exception as e:
        logger.warning(f"No se pudo auto-registrar GraphColoringProblem: {e}")

_register()

