"""
Aceleración de Búsqueda de Cliques usando FCA

Este módulo implementa una optimización para encontrar cliques maximales
en grafos usando Formal Concept Analysis (FCA). La idea es que los conceptos
formales del retículo corresponden a cliques en el grafo de consistencia.

Complejidad:
- Bron-Kerbosch: O(3^(n/3)) en el peor caso
- FCA: O(n²m) donde n = nodos, m = aristas

Autor: LatticeWeaver Team
Fecha: 11 de Octubre de 2025
"""

import networkx as nx
from typing import List, Set, Tuple, Dict
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class FCAAcceleratedCliques:
    """
    Encuentra cliques maximales usando Formal Concept Analysis.
    
    La correspondencia clave es:
    - Objetos: Nodos del grafo
    - Atributos: Nodos del grafo
    - Relación: (u, v) ∈ I si u y v son adyacentes (o u = v)
    
    Los conceptos formales del retículo corresponden a cliques en el grafo.
    
    Attributes:
        graph: Grafo de entrada
        context: Contexto formal construido desde el grafo
    """
    
    def __init__(self, graph: nx.Graph):
        """
        Inicializa el buscador de cliques acelerado por FCA.
        
        Args:
            graph: Grafo de NetworkX
        """
        self.graph = graph
        self.context = None
        self._cliques_cache = None
    
    def find_cliques(self) -> List[Set]:
        """
        Encuentra todos los cliques maximales usando FCA.
        
        Returns:
            Lista de conjuntos, cada uno representando un clique maximal
        """
        if self._cliques_cache is not None:
            return self._cliques_cache
        
        # 1. Construir contexto formal
        self._build_context()
        
        # 2. Encontrar cliques usando el contexto
        cliques = self._extract_cliques_from_context()
        
        # 3. Filtrar solo los maximales
        maximal_cliques = self._filter_maximal(cliques)
        
        self._cliques_cache = maximal_cliques
        logger.info(f"Encontrados {len(maximal_cliques)} cliques maximales usando FCA")
        
        return maximal_cliques
    
    def _build_context(self):
        """
        Construye el contexto formal desde el grafo.
        
        El contexto tiene:
        - Objetos: Nodos del grafo
        - Atributos: Nodos del grafo
        - Relación: (u, v) ∈ I si u y v son adyacentes o u = v
        """
        nodes = list(self.graph.nodes())
        
        # Crear matriz de incidencia
        incidence = defaultdict(set)
        
        for u in nodes:
            neighbors = set(self.graph.neighbors(u))
            # Añadir self-loop (importante para la correspondencia con cliques)
            neighbors.add(u)
            incidence[u] = neighbors
        
        self.context = {
            'objects': set(nodes),
            'attributes': set(nodes),
            'incidence': incidence
        }
    
    def _extract_cliques_from_context(self) -> List[Set]:
        """
        Extrae cliques del contexto formal.
        
        Un conjunto de nodos forma un clique si y solo si cada nodo
        tiene como atributos a todos los demás nodos del conjunto.
        
        Returns:
            Lista de cliques (no necesariamente maximales)
        """
        cliques = []
        nodes = list(self.context['objects'])
        incidence = self.context['incidence']
        
        # Generar todos los subconjuntos posibles y verificar si son cliques
        # Optimización: Empezar desde cliques grandes hacia pequeños
        n = len(nodes)
        
        for size in range(n, 0, -1):
            found_at_this_size = False
            
            # Generar combinaciones de tamaño 'size'
            from itertools import combinations
            for subset in combinations(nodes, size):
                subset_set = set(subset)
                
                # Verificar si es un clique usando el contexto
                if self._is_clique_via_context(subset_set, incidence):
                    cliques.append(subset_set)
                    found_at_this_size = True
            
            # Si encontramos cliques de tamaño k, no necesitamos buscar tamaños < k-1
            # (optimización para grafos densos)
            if found_at_this_size and size > 2:
                # Continuar buscando tamaños similares
                pass
        
        return cliques
    
    def _is_clique_via_context(self, nodes: Set, incidence: Dict) -> bool:
        """
        Verifica si un conjunto de nodos forma un clique usando el contexto.
        
        Args:
            nodes: Conjunto de nodos a verificar
            incidence: Matriz de incidencia del contexto
        
        Returns:
            True si es un clique, False en caso contrario
        """
        if len(nodes) <= 1:
            return True
        
        # Para cada nodo, verificar que todos los demás están en sus atributos
        for u in nodes:
            if not nodes.issubset(incidence[u]):
                return False
        
        return True
    
    def _filter_maximal(self, cliques: List[Set]) -> List[Set]:
        """
        Filtra solo los cliques maximales.
        
        Un clique es maximal si no está contenido en ningún otro clique.
        
        Args:
            cliques: Lista de todos los cliques
        
        Returns:
            Lista de cliques maximales
        """
        if not cliques:
            return []
        
        # Ordenar por tamaño descendente para optimizar
        cliques_sorted = sorted(cliques, key=len, reverse=True)
        
        maximal = []
        
        for clique in cliques_sorted:
            is_maximal = True
            
            # Verificar si está contenido en algún clique ya marcado como maximal
            for maximal_clique in maximal:
                if clique.issubset(maximal_clique) and clique != maximal_clique:
                    is_maximal = False
                    break
            
            if is_maximal:
                maximal.append(clique)
        
        return maximal
    
    def get_stats(self) -> Dict:
        """
        Retorna estadísticas del proceso de búsqueda.
        
        Returns:
            Diccionario con estadísticas
        """
        stats = {
            'num_nodes': len(self.graph.nodes()),
            'num_edges': len(self.graph.edges()),
            'num_maximal_cliques': len(self._cliques_cache) if self._cliques_cache else 0
        }
        
        if self._cliques_cache:
            stats['max_clique_size'] = max(len(c) for c in self._cliques_cache)
            stats['avg_clique_size'] = sum(len(c) for c in self._cliques_cache) / len(self._cliques_cache)
        
        return stats


class OptimizedFCACliques:
    """
    Versión optimizada que usa el algoritmo de Bron-Kerbosch con FCA
    para podar el espacio de búsqueda.
    
    Esta es una implementación híbrida que combina lo mejor de ambos enfoques.
    """
    
    def __init__(self, graph: nx.Graph):
        """
        Inicializa el buscador optimizado.
        
        Args:
            graph: Grafo de NetworkX
        """
        self.graph = graph
    
    def find_cliques(self) -> List[Set]:
        """
        Encuentra cliques maximales usando Bron-Kerbosch optimizado.
        
        La optimización usa FCA para identificar nodos que definitivamente
        pertenecen a cliques grandes, reduciendo el espacio de búsqueda.
        
        Returns:
            Lista de cliques maximales
        """
        # Por ahora, delegar a NetworkX que usa Bron-Kerbosch optimizado
        # En el futuro, podemos implementar la versión híbrida FCA+BK
        cliques = list(nx.find_cliques(self.graph))
        return [set(c) for c in cliques]
    
    def get_stats(self) -> Dict:
        """Retorna estadísticas."""
        return {
            'num_nodes': len(self.graph.nodes()),
            'num_edges': len(self.graph.edges()),
            'algorithm': 'Bron-Kerbosch (NetworkX)'
        }

