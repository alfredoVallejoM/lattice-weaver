"""
TopologyAnalyzer: Análisis topológico de problemas CSP.

Este módulo proporciona análisis topológico completo de la microestructura
de un CSP, incluyendo construcción del grafo de consistencia, complejos
simpliciales y cálculo de números de Betti.

Autor: Manus AI
Fecha: 11 de Octubre de 2025
"""

import networkx as nx
from typing import Dict, List, Tuple, Set, Any, Optional
from collections import defaultdict
import itertools


class TopologyAnalyzer:
    """
    Analizador topológico para problemas CSP.
    
    Construye el grafo de consistencia del problema y calcula invariantes
    topológicos como números de Betti, que revelan la estructura del espacio
    de soluciones.
    
    Soporta dos tipos de complejos:
    - Simpliciales: Basados en cliques (triángulos, tetraedros, etc.)
    - Cúbicos: Basados en cubos (cuadrados, cubos, etc.)
    
    Attributes:
        arc_engine: Motor de coherencia del problema
        consistency_graph: Grafo de consistencia (nodos = pares (var, val))
        simplicial_complex: Complejo simplicial construido desde cliques
        cubical_complex: Complejo cúbico construido desde cubos
        betti_numbers: Números de Betti calculados
    """
    
    def __init__(self, arc_engine):
        """
        Inicializa el analizador topológico.
        
        Args:
            arc_engine: Instancia de ArcEngine con el problema
        """
        self.arc_engine = arc_engine
        self.consistency_graph = nx.Graph()
        self.simplicial_complex = None
        self.cubical_complex = None
        self.betti_numbers = {}
        self._node_to_id = {}
        self._id_to_node = {}
    
    def build_consistency_graph(self):
        """
        Construye el grafo de consistencia del problema.
        
        El grafo de consistencia tiene:
        - Nodos: Pares (variable, valor)
        - Aristas: Entre pares consistentes según las restricciones
        
        Este grafo captura la microestructura del CSP.
        """
        # Limpiar grafo anterior
        self.consistency_graph.clear()
        self._node_to_id.clear()
        self._id_to_node.clear()
        
        # Añadir nodos (todos los pares variable-valor)
        node_id = 0
        for var_name, domain in self.arc_engine.variables.items():
            for value in domain.get_values():
                node = (var_name, value)
                self.consistency_graph.add_node(node)
                self._node_to_id[node] = node_id
                self._id_to_node[node_id] = node
                node_id += 1
        
        # Añadir aristas (pares consistentes)
        for cid, constraint in self.arc_engine.constraints.items():
            var1 = constraint.var1
            var2 = constraint.var2
            relation = constraint.relation
            
            domain1 = self.arc_engine.variables[var1]
            domain2 = self.arc_engine.variables[var2]
            
            for val1 in domain1.get_values():
                for val2 in domain2.get_values():
                    if relation(val1, val2):
                        self.consistency_graph.add_edge((var1, val1), (var2, val2))
        
        return self.consistency_graph
    
    def find_maximal_cliques(self, use_fca: bool = False) -> List[List[Tuple]]:
        """
        Encuentra todos los cliques maximales del grafo de consistencia.
        
        Un clique maximal es un conjunto de nodos donde todos están conectados
        entre sí, y no se puede añadir ningún otro nodo manteniendo esta propiedad.
        
        Args:
            use_fca: Si True, usa FCA para acelerar la búsqueda (experimental)
        
        Returns:
            Lista de cliques maximales
        """
        if not self.consistency_graph.nodes:
            self.build_consistency_graph()
        
        if use_fca:
            # Usar aceleración FCA (experimental)
            from .fca_cliques import OptimizedFCACliques
            fca_finder = OptimizedFCACliques(self.consistency_graph)
            cliques_sets = fca_finder.find_cliques()
            # Convertir de sets a listas para compatibilidad
            cliques = [list(c) for c in cliques_sets]
        else:
            # Usar Bron-Kerbosch estándar de NetworkX
            cliques = list(nx.find_cliques(self.consistency_graph))
        
        return cliques
    
    def build_simplicial_complex_simple(self) -> Dict[int, List[List[int]]]:
        """
        Construye un complejo simplicial desde los cliques del grafo.
        
        Un complejo simplicial es una colección de símplices (puntos, aristas,
        triángulos, tetraedros, etc.) que captura la estructura topológica.
        
        Returns:
            Diccionario {dimensión: lista de símplices}
        """
        cliques = self.find_maximal_cliques()
        
        # Convertir cliques a IDs numéricos
        simplicial_complex = defaultdict(list)
        
        for clique in cliques:
            # Convertir nodos a IDs
            clique_ids = [self._node_to_id[node] for node in clique]
            clique_ids.sort()
            
            # Añadir todos los sub-símplices
            for k in range(1, len(clique_ids) + 1):
                for simplex in itertools.combinations(clique_ids, k):
                    simplex_sorted = tuple(sorted(simplex))
                    if simplex_sorted not in simplicial_complex[k-1]:
                        simplicial_complex[k-1].append(list(simplex_sorted))
        
        self.simplicial_complex = dict(simplicial_complex)
        return self.simplicial_complex
    
    def compute_betti_numbers_simple(self, max_dimension: int = 3) -> Dict[str, int]:
        """
        Calcula los números de Betti del complejo simplicial.
        
        Los números de Betti son invariantes topológicos que cuentan "agujeros"
        de diferentes dimensiones:
        - β₀: Número de componentes conexas
        - β₁: Número de ciclos (agujeros 1D)
        - β₂: Número de cavidades (agujeros 2D)
        
        Esta es una implementación simplificada que calcula β₀ y β₁.
        
        Args:
            max_dimension: Dimensión máxima a calcular
            
        Returns:
            Diccionario con números de Betti
        """
        if not self.simplicial_complex:
            self.build_simplicial_complex_simple()
        
        # β₀: Número de componentes conexas
        num_components = nx.number_connected_components(self.consistency_graph)
        
        # β₁: Número de ciclos independientes
        # Fórmula de Euler para grafos: β₁ = |E| - |V| + |C|
        # donde C es el número de componentes
        num_nodes = self.consistency_graph.number_of_nodes()
        num_edges = self.consistency_graph.number_of_edges()
        
        if num_nodes > 0:
            beta_1 = num_edges - num_nodes + num_components
        else:
            beta_1 = 0
        
        # β₂ y superiores: Requieren cálculo de homología completo
        # Por simplicidad, usamos una heurística basada en cliques grandes
        beta_2 = 0
        if 2 in self.simplicial_complex:
            # Contar triángulos que podrían formar cavidades
            triangles = self.simplicial_complex[2]
            if len(triangles) > 10:
                # Heurística: Si hay muchos triángulos, probablemente hay cavidades
                beta_2 = max(0, len(triangles) // 20)
        
        self.betti_numbers = {
            'b0': num_components,
            'b1': max(0, beta_1),  # β₁ no puede ser negativo
            'b2': beta_2,
            'b3': 0  # Requiere cálculo completo
        }
        
        return self.betti_numbers
    
    def get_topology_summary(self) -> dict:
        """
        Obtiene un resumen completo del análisis topológico.
        
        Returns:
            Diccionario con estadísticas topológicas
        """
        if not self.consistency_graph.nodes:
            self.build_consistency_graph()
        
        if not self.betti_numbers:
            self.compute_betti_numbers_simple()
        
        # Estadísticas del grafo
        num_nodes = self.consistency_graph.number_of_nodes()
        num_edges = self.consistency_graph.number_of_edges()
        density = nx.density(self.consistency_graph) if num_nodes > 0 else 0
        
        # Estadísticas de cliques
        cliques = self.find_maximal_cliques()
        clique_sizes = [len(c) for c in cliques]
        
        # Grado promedio
        if num_nodes > 0:
            avg_degree = sum(dict(self.consistency_graph.degree()).values()) / num_nodes
        else:
            avg_degree = 0
        
        return {
            'graph_statistics': {
                'num_nodes': num_nodes,
                'num_edges': num_edges,
                'density': round(density, 4),
                'avg_degree': round(avg_degree, 2),
                'is_connected': nx.is_connected(self.consistency_graph) if num_nodes > 0 else False
            },
            'clique_statistics': {
                'num_maximal_cliques': len(cliques),
                'max_clique_size': max(clique_sizes) if clique_sizes else 0,
                'avg_clique_size': round(sum(clique_sizes) / len(clique_sizes), 2) if clique_sizes else 0
            },
            'betti_numbers': self.betti_numbers,
            'simplicial_complex': {
                'dimensions': list(self.simplicial_complex.keys()) if self.simplicial_complex else [],
                'total_simplices': sum(len(v) for v in self.simplicial_complex.values()) if self.simplicial_complex else 0
            }
        }
    
    def interpret_topology(self) -> dict:
        """
        Interpreta los invariantes topológicos en términos del CSP.
        
        Returns:
            Diccionario con interpretaciones
        """
        if not self.betti_numbers:
            self.compute_betti_numbers_simple()
        
        summary = self.get_topology_summary()
        
        interpretations = []
        
        # Interpretar β₀
        b0 = self.betti_numbers['b0']
        if b0 == 1:
            interpretations.append("El problema es conexo (una sola componente).")
        elif b0 > 1:
            interpretations.append(f"El problema se descompone en {b0} subproblemas independientes.")
        
        # Interpretar β₁
        b1 = self.betti_numbers['b1']
        if b1 == 0:
            interpretations.append("El grafo es un árbol o bosque (sin ciclos).")
        elif b1 > 0:
            interpretations.append(f"Existen {b1} ciclos independientes (posibles simetrías).")
        
        # Interpretar β₂
        b2 = self.betti_numbers['b2']
        if b2 > 0:
            interpretations.append(f"Estructura compleja de alto orden detectada ({b2} cavidades estimadas).")
        
        # Interpretar densidad
        density = summary['graph_statistics']['density']
        if density > 0.7:
            interpretations.append("Grafo muy denso: muchas restricciones entre variables.")
        elif density < 0.1:
            interpretations.append("Grafo disperso: pocas restricciones entre variables.")
        
        # Interpretar conectividad
        if summary['graph_statistics']['is_connected']:
            interpretations.append("Todas las variables están interconectadas.")
        else:
            interpretations.append("Existen variables aisladas o grupos separados.")
        
        return {
            'betti_numbers': self.betti_numbers,
            'interpretations': interpretations,
            'complexity_estimate': self._estimate_complexity(summary)
        }
    
    def _estimate_complexity(self, summary: dict) -> str:
        """
        Estima la complejidad del problema basándose en la topología.
        
        Args:
            summary: Resumen topológico
            
        Returns:
            Estimación de complejidad ('Baja', 'Media', 'Alta')
        """
        b1 = self.betti_numbers.get('b1', 0)
        b2 = self.betti_numbers.get('b2', 0)
        density = summary['graph_statistics']['density']
        num_cliques = summary['clique_statistics']['num_maximal_cliques']
        
        complexity_score = 0
        
        # Factores que aumentan complejidad
        if b1 > 5:
            complexity_score += 2
        elif b1 > 0:
            complexity_score += 1
        
        if b2 > 0:
            complexity_score += 2
        
        if density > 0.5:
            complexity_score += 1
        
        if num_cliques > 100:
            complexity_score += 2
        elif num_cliques > 20:
            complexity_score += 1
        
        # Clasificar
        if complexity_score <= 2:
            return "Baja"
        elif complexity_score <= 4:
            return "Media"
        else:
            return "Alta"
    
    def export_graph(self, filepath: str, format: str = 'gexf'):
        """
        Exporta el grafo de consistencia a un archivo.
        
        Args:
            filepath: Ruta del archivo
            format: Formato ('gexf', 'graphml', 'gml')
        """
        if not self.consistency_graph.nodes:
            self.build_consistency_graph()
        
        if format == 'gexf':
            nx.write_gexf(self.consistency_graph, filepath)
        elif format == 'graphml':
            nx.write_graphml(self.consistency_graph, filepath)
        elif format == 'gml':
            nx.write_gml(self.consistency_graph, filepath)
        else:
            raise ValueError(f"Formato no soportado: {format}")
    
    def visualize_graph(self, output_path: Optional[str] = None):
        """
        Crea una visualización del grafo de consistencia.
        
        Args:
            output_path: Ruta para guardar la imagen (opcional)
        """
        try:
            import matplotlib.pyplot as plt
            
            if not self.consistency_graph.nodes:
                self.build_consistency_graph()
            
            plt.figure(figsize=(12, 8))
            
            # Layout
            if self.consistency_graph.number_of_nodes() < 100:
                pos = nx.spring_layout(self.consistency_graph, k=0.5, iterations=50)
            else:
                pos = nx.kamada_kawai_layout(self.consistency_graph)
            
            # Dibujar
            nx.draw(self.consistency_graph, pos,
                   node_size=50,
                   node_color='lightblue',
                   edge_color='gray',
                   alpha=0.7,
                   with_labels=False)
            
            plt.title("Grafo de Consistencia")
            
            if output_path:
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
            else:
                plt.show()
            
            plt.close()
        
        except ImportError:
            print("matplotlib no está disponible para visualización.")



    def build_cubical_complex(self):
        """
        Construye un complejo cúbico desde el grafo de consistencia.
        
        Los complejos cúbicos son una alternativa a los complejos simpliciales
        que puede ser más eficiente para ciertos tipos de problemas.
        
        Returns:
            Complejo cúbico construido
        """
        if not self.consistency_graph.nodes:
            self.build_consistency_graph()
        
        from .cubical_complex import CubicalComplex
        
        self.cubical_complex = CubicalComplex(self.consistency_graph)
        self.cubical_complex.build_complex()
        
        return self.cubical_complex
    
    def compute_cubical_homology(self) -> Dict[str, int]:
        """
        Calcula la homología usando complejos cúbicos.
        
        Returns:
            Diccionario con números de Betti cúbicos
        """
        if self.cubical_complex is None:
            self.build_cubical_complex()
        
        betti_cubical = self.cubical_complex.compute_cubical_homology()
        
        return betti_cubical
    
    def analyze_with_method(self, method: str = 'simplicial') -> Dict:
        """
        Ejecuta análisis topológico usando el método especificado.
        
        Args:
            method: 'simplicial' o 'cubical'
        
        Returns:
            Diccionario con resultados del análisis
        """
        if method == 'cubical':
            # Análisis con complejos cúbicos
            self.build_cubical_complex()
            betti = self.compute_cubical_homology()
            
            return {
                'method': 'cubical',
                'betti_numbers': betti,
                'complex_stats': self.cubical_complex.get_stats()
            }
        else:
            # Análisis con complejos simpliciales (por defecto)
            self.build_simplicial_complex_simple()
            betti = self.compute_betti_numbers_simple()
            
            return {
                'method': 'simplicial',
                'betti_numbers': betti,
                'complex_stats': {
                    'num_simplices': sum(len(simplices) for simplices in self.simplicial_complex.values())
                }
            }

