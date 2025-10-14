# lattice_weaver/core/csp_engine/graph.py

"""
Adaptador de compatibilidad para el módulo graph

Este módulo proporciona funcionalidades relacionadas con grafos de restricciones.
Si no existe un equivalente directo en arc_engine, se proporcionan implementaciones stub.
"""

from typing import List, Dict, Set, Tuple, Any, Optional
import networkx as nx
from collections import defaultdict

class ConstraintGraph:
    """
    Representa un grafo de restricciones para un CSP.
    Los nodos son variables y las aristas representan restricciones entre variables.
    """
    
    def __init__(self):
        """Inicializa un grafo de restricciones vacío."""
        self.variables: Dict[str, Set[Any]] = {}
        self.constraints: Dict[Tuple[str, str], List[Any]] = defaultdict(list)
        self.graph = nx.Graph()
    
    def add_variable(self, variable: str, domain: Set[Any]):
        """
        Añade una variable al grafo con su dominio.
        
        Args:
            variable: Nombre de la variable
            domain: Conjunto de valores posibles para la variable
        """
        self.variables[variable] = set(domain)
        self.graph.add_node(variable)
    
    def add_constraint(self, var1: str, var2: str, constraint: Any):
        """
        Añade una restricción binaria entre dos variables.
        
        Args:
            var1: Primera variable
            var2: Segunda variable
            constraint: Objeto de restricción
        """
        key = tuple(sorted([var1, var2]))
        self.constraints[key].append(constraint)
        self.graph.add_edge(var1, var2, constraint=constraint)
    
    def get_domain(self, variable: str) -> Set[Any]:
        """Obtiene el dominio de una variable."""
        return self.variables.get(variable, set())
    
    def set_domain(self, variable: str, domain: Set[Any]):
        """Establece el dominio de una variable."""
        self.variables[variable] = set(domain)
    
    def get_neighbors(self, variable: str) -> List[str]:
        """Obtiene las variables vecinas (conectadas por restricciones)."""
        return list(self.graph.neighbors(variable))
    
    def get_constraints(self, var1: str, var2: str) -> List[Any]:
        """Obtiene las restricciones entre dos variables."""
        key = tuple(sorted([var1, var2]))
        return self.constraints.get(key, [])
    
    def get_all_variables(self) -> List[str]:
        """Obtiene todas las variables del grafo."""
        return list(self.variables.keys())
    
    def get_degree(self, variable: str) -> int:
        """Obtiene el grado de una variable (número de vecinos)."""
        return self.graph.degree(variable)
    
    def copy(self) -> 'ConstraintGraph':
        """Crea una copia del grafo de restricciones."""
        new_graph = ConstraintGraph()
        for var, domain in self.variables.items():
            new_graph.add_variable(var, domain)
        for (var1, var2), constraints in self.constraints.items():
            for constraint in constraints:
                new_graph.add_constraint(var1, var2, constraint)
        return new_graph

def build_constraint_graph(variables: List[str], constraints: List[Any]) -> nx.Graph:
    """
    Construye un grafo de restricciones a partir de variables y restricciones.
    
    Args:
        variables: Lista de nombres de variables
        constraints: Lista de restricciones
        
    Returns:
        Un grafo de NetworkX donde los nodos son variables y las aristas representan restricciones
    """
    G = nx.Graph()
    G.add_nodes_from(variables)
    
    for constraint in constraints:
        # Asumiendo que las restricciones tienen un atributo 'scope' o similar
        if hasattr(constraint, 'scope'):
            scope_list = list(constraint.scope) if hasattr(constraint.scope, '__iter__') else [constraint.scope]
            if len(scope_list) == 2:
                G.add_edge(scope_list[0], scope_list[1], constraint=constraint)
        elif isinstance(constraint, tuple) and len(constraint) >= 2:
            # Formato (var1, var2, relation)
            G.add_edge(constraint[0], constraint[1], constraint=constraint)
    
    return G

def get_neighbors(graph: nx.Graph, variable: str) -> Set[str]:
    """Obtiene los vecinos de una variable en el grafo de restricciones."""
    return set(graph.neighbors(variable))

def get_degree(graph: nx.Graph, variable: str) -> int:
    """Obtiene el grado de una variable en el grafo de restricciones."""
    return graph.degree(variable)

__all__ = [
    'ConstraintGraph',
    'build_constraint_graph',
    'get_neighbors',
    'get_degree'
]

