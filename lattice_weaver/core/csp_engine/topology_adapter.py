"""
topology_adapter.py: Adaptador CSP para TopologyAnalyzer.

Este módulo proporciona un adaptador que permite a TopologyAnalyzer trabajar
directamente con instancias de CSP sin necesidad de arc_engine.

Autor: Manus AI
Fecha: 15 de Octubre de 2025
"""

from typing import Dict, List, Tuple, Set, Any, Optional
import networkx as nx
from lattice_weaver.core.csp_problem import CSP, Constraint


class CSPTopologyAdapter:
    """
    Adaptador que permite a TopologyAnalyzer trabajar con CSP.
    
    Este adaptador construye el grafo de consistencia directamente desde un CSP,
    sin necesidad de arc_engine. El grafo captura la microestructura del problema:
    - Nodos: Pares (variable, valor)
    - Aristas: Entre pares consistentes según las restricciones
    
    Attributes:
        csp: Problema CSP a analizar
        consistency_graph: Grafo de consistencia construido
        node_to_id: Mapeo de nodos a IDs
        id_to_node: Mapeo de IDs a nodos
    """
    
    def __init__(self, csp: CSP):
        """
        Inicializa el adaptador.
        
        Args:
            csp: Problema CSP a analizar
        """
        self.csp = csp
        self.consistency_graph = nx.Graph()
        self.node_to_id: Dict[Tuple[str, Any], int] = {}
        self.id_to_node: Dict[int, Tuple[str, Any]] = {}
    
    def build_consistency_graph(self) -> nx.Graph:
        """
        Construye el grafo de consistencia del CSP.
        
        Returns:
            Grafo de consistencia (nodos = pares (var, val), aristas = consistencia)
        """
        # Limpiar grafo anterior
        self.consistency_graph.clear()
        self.node_to_id.clear()
        self.id_to_node.clear()
        
        # Añadir nodos (todos los pares variable-valor)
        node_id = 0
        for var in self.csp.variables:
            for value in self.csp.domains[var]:
                node = (var, value)
                self.consistency_graph.add_node(node)
                self.node_to_id[node] = node_id
                self.id_to_node[node_id] = node
                node_id += 1
        
        # Añadir aristas (pares consistentes)
        for constraint in self.csp.constraints:
            if len(constraint.scope) == 2:
                var1, var2 = list(constraint.scope)
                
                for val1 in self.csp.domains[var1]:
                    for val2 in self.csp.domains[var2]:
                        # Verificar si (val1, val2) satisface la restricción
                        try:
                            if constraint.relation(val1, val2):
                                self.consistency_graph.add_edge((var1, val1), (var2, val2))
                        except:
                            # Si la relación falla, asumir inconsistencia
                            pass
        
        return self.consistency_graph
    
    def find_connected_components(self) -> List[Set[Tuple[str, Any]]]:
        """
        Encuentra componentes conexas del grafo de consistencia.
        
        Componentes conexas revelan regiones independientes del espacio de soluciones.
        
        Returns:
            Lista de componentes conexas (cada una es un conjunto de nodos)
        """
        if not self.consistency_graph:
            self.build_consistency_graph()
        
        return list(nx.connected_components(self.consistency_graph))
    
    def compute_graph_metrics(self) -> Dict[str, Any]:
        """
        Calcula métricas del grafo de consistencia.
        
        Returns:
            Diccionario con métricas:
            - num_nodes: Número de nodos
            - num_edges: Número de aristas
            - density: Densidad del grafo
            - num_components: Número de componentes conexas
            - largest_component_size: Tamaño de la componente más grande
            - average_degree: Grado promedio
            - clustering_coefficient: Coeficiente de clustering
        """
        if not self.consistency_graph:
            self.build_consistency_graph()
        
        num_nodes = self.consistency_graph.number_of_nodes()
        num_edges = self.consistency_graph.number_of_edges()
        
        components = list(nx.connected_components(self.consistency_graph))
        num_components = len(components)
        largest_component_size = max(len(c) for c in components) if components else 0
        
        density = nx.density(self.consistency_graph) if num_nodes > 0 else 0.0
        
        degrees = [d for n, d in self.consistency_graph.degree()]
        average_degree = sum(degrees) / len(degrees) if degrees else 0.0
        
        clustering_coefficient = nx.average_clustering(self.consistency_graph) if num_nodes > 0 else 0.0
        
        return {
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'density': density,
            'num_components': num_components,
            'largest_component_size': largest_component_size,
            'average_degree': average_degree,
            'clustering_coefficient': clustering_coefficient
        }
    
    def find_critical_nodes(self, top_k: int = 5) -> List[Tuple[Tuple[str, Any], float]]:
        """
        Encuentra nodos críticos usando centralidad de intermediación (betweenness).
        
        Nodos críticos son aquellos cuya eliminación fragmentaría el grafo.
        
        Args:
            top_k: Número de nodos críticos a retornar
        
        Returns:
            Lista de (nodo, centralidad) ordenada por centralidad descendente
        """
        if not self.consistency_graph:
            self.build_consistency_graph()
        
        if self.consistency_graph.number_of_nodes() == 0:
            return []
        
        betweenness = nx.betweenness_centrality(self.consistency_graph)
        
        sorted_nodes = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_nodes[:top_k]
    
    def analyze_structure(self) -> Dict[str, Any]:
        """
        Análisis estructural completo del CSP.
        
        Returns:
            Diccionario con:
            - metrics: Métricas del grafo
            - components: Componentes conexas
            - critical_nodes: Nodos críticos
            - summary: Resumen textual
        """
        if not self.consistency_graph:
            self.build_consistency_graph()
        
        metrics = self.compute_graph_metrics()
        components = self.find_connected_components()
        critical_nodes = self.find_critical_nodes()
        
        # Generar resumen
        summary_lines = [
            f"Análisis Topológico del CSP:",
            f"  - Variables: {len(self.csp.variables)}",
            f"  - Restricciones: {len(self.csp.constraints)}",
            f"  - Nodos (var-val): {metrics['num_nodes']}",
            f"  - Aristas (consistencias): {metrics['num_edges']}",
            f"  - Densidad: {metrics['density']:.4f}",
            f"  - Componentes conexas: {metrics['num_components']}",
            f"  - Componente más grande: {metrics['largest_component_size']} nodos",
            f"  - Grado promedio: {metrics['average_degree']:.2f}",
            f"  - Coeficiente de clustering: {metrics['clustering_coefficient']:.4f}",
        ]
        
        if critical_nodes:
            summary_lines.append(f"\nNodos Críticos (Top {len(critical_nodes)}):")
            for node, centrality in critical_nodes:
                summary_lines.append(f"  - {node}: {centrality:.4f}")
        
        summary = "\n".join(summary_lines)
        
        return {
            'metrics': metrics,
            'components': components,
            'critical_nodes': critical_nodes,
            'summary': summary
        }


def analyze_csp_topology(csp: CSP) -> Dict[str, Any]:
    """
    Función de conveniencia para analizar la topología de un CSP.
    
    Args:
        csp: Problema CSP a analizar
    
    Returns:
        Diccionario con análisis estructural completo
    """
    adapter = CSPTopologyAdapter(csp)
    return adapter.analyze_structure()

