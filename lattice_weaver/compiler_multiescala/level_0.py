"""
Nivel L0: Primitivas CSP

Este módulo implementa el nivel más bajo de abstracción del compilador multiescala,
que representa un CSP en su forma canónica con variables individuales, dominios
atómicos y restricciones binarias básicas.
"""

from typing import Dict, List, Set, FrozenSet, Any, Optional
import math
from collections import defaultdict
import networkx as nx

from .base import AbstractionLevel
from ..core.csp_problem import CSP, Constraint


class Level0(AbstractionLevel):
    """
    Nivel L0: Primitivas CSP
    
    Representa el CSP en su forma más básica, sin agregación ni abstracción.
    Este nivel sirve como punto de entrada para la jerarquía de abstracción.
    
    Attributes:
        csp: El CSP en su forma canónica.
        constraint_graph: Grafo de restricciones para análisis estructural.
    """

    def __init__(self, csp: CSP, config: dict = None):
        """
        Inicializa el nivel L0 con un CSP.
        
        Args:
            csp: El CSP a representar en este nivel.
            config: Configuración opcional para el nivel.
        """
        super().__init__(level=0, config=config)
        self.csp = csp
        self.data = csp
        self.constraint_graph = self._build_constraint_graph()

    def _build_constraint_graph(self) -> nx.Graph:
        """
        Construye un grafo de restricciones donde los nodos son variables
        y las aristas representan restricciones entre ellas.
        
        Returns:
            Un grafo NetworkX con las variables como nodos.
        """
        G = nx.Graph()
        G.add_nodes_from(self.csp.variables)
        
        for constraint in self.csp.constraints:
            # Para restricciones binarias, añadir una arista
            if len(constraint.scope) == 2:
                v1, v2 = constraint.scope
                if G.has_edge(v1, v2):
                    # Si ya existe una arista, incrementar el peso
                    G[v1][v2]['weight'] += 1
                    G[v1][v2]['constraints'].append(constraint)
                else:
                    G.add_edge(v1, v2, weight=1, constraints=[constraint])
            # Para restricciones de aridad mayor, añadir un hiper-nodo
            elif len(constraint.scope) > 2:
                # Crear un nodo especial para la restricción
                constraint_node = f"_constraint_{id(constraint)}"
                G.add_node(constraint_node, is_constraint=True, constraint=constraint)
                for var in constraint.scope:
                    G.add_edge(var, constraint_node, weight=1, constraints=[constraint])
        
        return G

    def build_from_lower(self, lower_level: AbstractionLevel):
        """
        L0 es el nivel más bajo, no tiene nivel inferior.
        
        Raises:
            NotImplementedError: Siempre, ya que L0 no tiene nivel inferior.
        """
        raise NotImplementedError("L0 is the lowest level and cannot be built from a lower level.")

    def refine_to_lower(self) -> AbstractionLevel:
        """
        L0 es el nivel más bajo, no puede refinarse a un nivel inferior.
        
        Raises:
            NotImplementedError: Siempre, ya que L0 no tiene nivel inferior.
        """
        raise NotImplementedError("L0 is the lowest level and cannot be refined to a lower level.")

    def renormalize(self, partitioner, k: int) -> 'Level0':
        """
        Aplica la renormalización en el nivel L0.
        
        Esta operación utiliza el sistema de renormalización existente para
        reducir la complejidad del CSP.
        
        Args:
            partitioner: Estrategia de particionamiento ('metis', 'simple', 'symmetry', etc.)
            k: Número de particiones deseadas.
        
        Returns:
            Un nuevo Level0 con el CSP renormalizado.
        """
        from ..renormalization.core import renormalize_single_level
        
        renormalized_csp, _, _ = renormalize_single_level(
            source_csp=self.csp,
            source_level=0,
            k=k,
            partition_strategy=partitioner
        )
        
        if renormalized_csp is None:
            # Si la renormalización falla, devolver el CSP original
            return Level0(self.csp, config=self.config)
        
        return Level0(renormalized_csp, config=self.config)

    def validate(self) -> bool:
        """
        Valida la coherencia interna del CSP en L0.
        
        Verifica:
        - Todas las variables tienen dominios definidos
        - Todas las restricciones referencian variables existentes
        - Los dominios no están vacíos (excepto para CSPs vacíos)
        - El grafo de restricciones es consistente
        
        Returns:
            True si el CSP es válido, False en caso contrario.
        """
        try:
            # Permitir CSPs vacíos
            if len(self.csp.variables) == 0:
                return True
            
            # Verificar que todas las variables tienen dominios
            for var in self.csp.variables:
                if var not in self.csp.domains:
                    return False
                if len(self.csp.domains[var]) == 0:
                    return False
            
            # Verificar que todas las restricciones referencian variables existentes
            for constraint in self.csp.constraints:
                for var in constraint.scope:
                    if var not in self.csp.variables:
                        return False
            
            # Verificar que el grafo de restricciones es consistente
            if not nx.is_connected(self.constraint_graph):
                # Permitir grafos no conectados (múltiples componentes independientes)
                pass
            
            return True
        except Exception:
            return False

    @property
    def complexity(self) -> float:
        """
        Calcula la complejidad del CSP en L0.
        
        La complejidad se define como el logaritmo del tamaño del espacio de búsqueda,
        que es el producto de los tamaños de los dominios de todas las variables.
        
        Returns:
            El logaritmo del tamaño del espacio de búsqueda.
        """
        if not self.csp.variables:
            return 0.0
        
        # Calcular el logaritmo del producto de los tamaños de los dominios
        log_complexity = sum(math.log(len(self.csp.domains[var]) + 1) for var in self.csp.variables)
        return log_complexity

    def detect_constraint_blocks(self) -> List[Set[str]]:
        """
        Detecta bloques de restricciones fuertemente acopladas.
        
        Un bloque es un conjunto de variables que están fuertemente conectadas
        en el grafo de restricciones. Esta función utiliza la detección de
        comunidades para identificar estos bloques.
        
        Returns:
            Una lista de conjuntos de variables, donde cada conjunto es un bloque.
        """
        # Usar la detección de comunidades de Louvain si está disponible
        try:
            import community as community_louvain
            partition = community_louvain.best_partition(self.constraint_graph)
            
            # Agrupar variables por comunidad
            blocks = defaultdict(set)
            for var, community_id in partition.items():
                # Ignorar nodos de restricción
                if not self.constraint_graph.nodes[var].get('is_constraint', False):
                    blocks[community_id].add(var)
            
            return list(blocks.values())
        except ImportError:
            # Si Louvain no está disponible, usar componentes conexas
            # Filtrar nodos de restricción
            variable_nodes = [n for n in self.constraint_graph.nodes() 
                            if not self.constraint_graph.nodes[n].get('is_constraint', False)]
            subgraph = self.constraint_graph.subgraph(variable_nodes)
            
            components = list(nx.connected_components(subgraph))
            return components

    def get_statistics(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas sobre el CSP en L0.
        
        Returns:
            Un diccionario con estadísticas del CSP.
        """
        # Manejar grafos vacíos
        graph_diameter = None
        num_connected_components = 0
        
        if len(self.constraint_graph.nodes()) > 0:
            try:
                if nx.is_connected(self.constraint_graph):
                    graph_diameter = nx.diameter(self.constraint_graph)
            except nx.NetworkXError:
                pass
            num_connected_components = nx.number_connected_components(self.constraint_graph)
        
        return {
            'level': self.level,
            'num_variables': len(self.csp.variables),
            'num_constraints': len(self.csp.constraints),
            'avg_domain_size': sum(len(d) for d in self.csp.domains.values()) / len(self.csp.domains) if self.csp.domains else 0,
            'graph_density': nx.density(self.constraint_graph) if len(self.constraint_graph.nodes()) > 0 else 0,
            'graph_diameter': graph_diameter,
            'num_connected_components': num_connected_components,
            'complexity': self.complexity,
        }

    def __repr__(self) -> str:
        return f"Level0(variables={len(self.csp.variables)}, constraints={len(self.csp.constraints)}, complexity={self.complexity:.2f})"

