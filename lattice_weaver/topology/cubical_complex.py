"""
Complejos Cúbicos para Análisis Topológico

Este módulo implementa complejos cúbicos como alternativa a los complejos
simpliciales. Los complejos cúbicos pueden ser más eficientes computacionalmente
para ciertos tipos de problemas, especialmente aquellos con estructura de malla
o cuadrícula.

Diferencias clave:
- Simpliciales: Construidos desde triángulos (símplices)
- Cúbicos: Construidos desde cubos (hipercubos)

Ventajas de complejos cúbicos:
- Más naturales para problemas con estructura de cuadrícula
- Potencialmente más eficientes en memoria
- Mejor para ciertos tipos de visualización

Autor: LatticeWeaver Team
Fecha: 11 de Octubre de 2025
"""

import networkx as nx
from typing import List, Set, Tuple, Dict
from itertools import combinations
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class CubicalComplex:
    """
    Complejo cúbico para análisis topológico de grafos.
    
    Un complejo cúbico es una colección de cubos de diferentes dimensiones:
    - 0-cubos: Vértices (puntos)
    - 1-cubos: Aristas (segmentos)
    - 2-cubos: Cuadrados (caras)
    - 3-cubos: Cubos (volúmenes)
    
    Attributes:
        graph: Grafo de entrada
        cubes: Diccionario {dimensión: lista de cubos}
    """
    
    def __init__(self, graph: nx.Graph):
        """
        Inicializa el complejo cúbico.
        
        Args:
            graph: Grafo de NetworkX
        """
        self.graph = graph
        self.cubes = {0: [], 1: [], 2: [], 3: []}
        self._built = False
    
    def build_complex(self):
        """
        Construye el complejo cúbico a partir del grafo.
        
        Extrae todos los cubos de dimensiones 0, 1, 2 y 3.
        """
        logger.info("Construyendo complejo cúbico...")
        
        # 0-cubos: vértices
        self.cubes[0] = list(self.graph.nodes())
        logger.debug(f"Encontrados {len(self.cubes[0])} 0-cubos (vértices)")
        
        # 1-cubos: aristas
        self.cubes[1] = list(self.graph.edges())
        logger.debug(f"Encontrados {len(self.cubes[1])} 1-cubos (aristas)")
        
        # 2-cubos: cuadrados (ciclos de longitud 4)
        self.cubes[2] = self._find_squares()
        logger.debug(f"Encontrados {len(self.cubes[2])} 2-cubos (cuadrados)")
        
        # 3-cubos: cubos (hipercubos de dimensión 3)
        # Solo buscar en grafos pequeños (< 50 nodos) por eficiencia
        if len(self.graph.nodes()) < 50:
            self.cubes[3] = self._find_cubes()
            logger.debug(f"Encontrados {len(self.cubes[3])} 3-cubos (cubos)")
        else:
            logger.debug("Grafo demasiado grande, omitiendo búsqueda de 3-cubos")
            self.cubes[3] = []
        
        self._built = True
        logger.info("Complejo cúbico construido exitosamente")
    
    def _find_squares(self) -> List[Tuple]:
        """
        Encuentra todos los cuadrados (2-cubos) en el grafo.
        
        Un cuadrado es un ciclo de 4 nodos donde cada nodo está conectado
        a exactamente 2 otros nodos del ciclo.
        
        Returns:
            Lista de tuplas de 4 nodos que forman cuadrados
        """
        squares = set()
        
        # Buscar ciclos de longitud 4
        for node in self.graph.nodes():
            neighbors = list(self.graph.neighbors(node))
            
            # Para cada par de vecinos
            for i, n1 in enumerate(neighbors):
                for n2 in neighbors[i+1:]:
                    # Buscar un cuarto nodo que complete el cuadrado
                    common_neighbors = set(self.graph.neighbors(n1)) & set(self.graph.neighbors(n2))
                    common_neighbors.discard(node)
                    
                    for n3 in common_neighbors:
                        # Verificar que forma un cuadrado
                        square_nodes = tuple(sorted([node, n1, n2, n3]))
                        if self._is_square(square_nodes):
                            squares.add(square_nodes)
        
        return list(squares)
    
    def _is_square(self, nodes: Tuple) -> bool:
        """
        Verifica si 4 nodos forman un cuadrado.
        
        Un cuadrado debe tener exactamente 4 aristas formando un ciclo.
        
        Args:
            nodes: Tupla de 4 nodos
        
        Returns:
            True si forman un cuadrado, False en caso contrario
        """
        if len(nodes) != 4:
            return False
        
        # Contar aristas entre los 4 nodos
        subgraph = self.graph.subgraph(nodes)
        num_edges = len(subgraph.edges())
        
        # Un cuadrado tiene exactamente 4 aristas
        # (un ciclo, sin diagonales)
        if num_edges != 4:
            return False
        
        # Verificar que cada nodo tiene grado 2 en el subgrafo
        for node in nodes:
            if subgraph.degree(node) != 2:
                return False
        
        return True
    
    def _find_cubes(self) -> List[Tuple]:
        """
        Encuentra todos los cubos (3-cubos) en el grafo.
        
        Un cubo tiene 8 vértices y 12 aristas, donde cada vértice
        tiene exactamente 3 vecinos dentro del cubo.
        
        Returns:
            Lista de tuplas de 8 nodos que forman cubos
        """
        cubes = []
        
        # Solo buscar en grafos pequeños
        if len(self.graph.nodes()) > 20:
            return cubes
        
        # Buscar subgrafos de 8 nodos
        for nodes in combinations(self.graph.nodes(), 8):
            if self._is_cube(nodes):
                cubes.append(tuple(sorted(nodes)))
        
        return cubes
    
    def _is_cube(self, nodes: Tuple) -> bool:
        """
        Verifica si 8 nodos forman un cubo.
        
        Args:
            nodes: Tupla de 8 nodos
        
        Returns:
            True si forman un cubo, False en caso contrario
        """
        if len(nodes) != 8:
            return False
        
        subgraph = self.graph.subgraph(nodes)
        
        # Un cubo tiene exactamente 12 aristas
        if len(subgraph.edges()) != 12:
            return False
        
        # Cada nodo debe tener exactamente 3 vecinos
        for node in nodes:
            if subgraph.degree(node) != 3:
                return False
        
        return True
    
    def compute_cubical_homology(self) -> Dict[str, int]:
        """
        Calcula la homología cúbica del complejo.
        
        Los números de Betti cúbicos son análogos a los simpliciales:
        - β₀: Número de componentes conexas
        - β₁: Número de ciclos independientes
        - β₂: Número de cavidades 2D
        
        Returns:
            Diccionario con números de Betti
        """
        if not self._built:
            self.build_complex()
        
        beta_0 = self._compute_beta0_cubical()
        beta_1 = self._compute_beta1_cubical()
        beta_2 = self._compute_beta2_cubical()
        
        return {
            'beta_0': beta_0,
            'beta_1': beta_1,
            'beta_2': beta_2
        }
    
    def _compute_beta0_cubical(self) -> int:
        """
        Calcula β₀: Número de componentes conexas.
        
        Returns:
            Número de componentes conexas
        """
        return nx.number_connected_components(self.graph)
    
    def _compute_beta1_cubical(self) -> int:
        """
        Calcula β₁: Número de ciclos independientes.
        
        Usa la fórmula de Euler: β₁ = |E| - |V| + |C|
        donde E = aristas, V = vértices, C = componentes
        
        Returns:
            Número de ciclos independientes
        """
        num_edges = len(self.cubes[1])
        num_vertices = len(self.cubes[0])
        num_components = self._compute_beta0_cubical()
        
        beta_1 = num_edges - num_vertices + num_components
        
        # β₁ no puede ser negativo
        return max(0, beta_1)
    
    def _compute_beta2_cubical(self) -> int:
        """
        Calcula β₂: Número de cavidades 2D.
        
        Aproximación: contar cuadrados que no son caras de cubos.
        
        Returns:
            Número de cavidades 2D
        """
        # Aproximación simple: contar cuadrados
        # En una implementación completa, usaríamos homología persistente
        num_squares = len(self.cubes[2])
        num_cubes = len(self.cubes[3])
        
        # Cada cubo tiene 6 caras cuadradas
        # Los cuadrados que no son caras de cubos contribuyen a β₂
        beta_2 = max(0, num_squares - 6 * num_cubes)
        
        return beta_2
    
    def get_stats(self) -> Dict:
        """
        Retorna estadísticas del complejo cúbico.
        
        Returns:
            Diccionario con estadísticas
        """
        if not self._built:
            self.build_complex()
        
        return {
            'num_0_cubes': len(self.cubes[0]),
            'num_1_cubes': len(self.cubes[1]),
            'num_2_cubes': len(self.cubes[2]),
            'num_3_cubes': len(self.cubes[3]),
            'total_cubes': sum(len(self.cubes[d]) for d in range(4))
        }
    
    def __repr__(self) -> str:
        """Representación en string del complejo cúbico."""
        stats = self.get_stats()
        return (f"CubicalComplex("
                f"0-cubes={stats['num_0_cubes']}, "
                f"1-cubes={stats['num_1_cubes']}, "
                f"2-cubes={stats['num_2_cubes']}, "
                f"3-cubes={stats['num_3_cubes']})")

