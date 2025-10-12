"""
Generadores de grafos para problemas CSP.

Este módulo proporciona funciones para generar diferentes tipos de grafos
que se usan en problemas como Graph Coloring y Map Coloring.
"""

import random
from typing import List, Tuple, Set
import logging

logger = logging.getLogger(__name__)


def generate_random_graph(n_nodes: int, edge_probability: float, seed: int = None) -> List[Tuple[int, int]]:
    """
    Genera un grafo aleatorio usando el modelo Erdős-Rényi.
    
    Args:
        n_nodes: Número de nodos
        edge_probability: Probabilidad de que exista una arista entre dos nodos (0.0 a 1.0)
        seed: Semilla para reproducibilidad (opcional)
        
    Returns:
        Lista de aristas como tuplas (i, j) donde i < j
        
    Raises:
        ValueError: Si los parámetros son inválidos
    """
    if n_nodes < 1:
        raise ValueError(f"n_nodes debe ser >= 1, recibido {n_nodes}")
    if not 0.0 <= edge_probability <= 1.0:
        raise ValueError(f"edge_probability debe estar en [0.0, 1.0], recibido {edge_probability}")
    
    if seed is not None:
        random.seed(seed)
    
    edges = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if random.random() < edge_probability:
                edges.append((i, j))
    
    logger.debug(f"Generado grafo aleatorio: {n_nodes} nodos, {len(edges)} aristas")
    return edges


def generate_complete_graph(n_nodes: int) -> List[Tuple[int, int]]:
    """
    Genera un grafo completo (todos los nodos conectados entre sí).
    
    Args:
        n_nodes: Número de nodos
        
    Returns:
        Lista de aristas como tuplas (i, j) donde i < j
        
    Raises:
        ValueError: Si n_nodes < 1
    """
    if n_nodes < 1:
        raise ValueError(f"n_nodes debe ser >= 1, recibido {n_nodes}")
    
    edges = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            edges.append((i, j))
    
    logger.debug(f"Generado grafo completo: {n_nodes} nodos, {len(edges)} aristas")
    return edges


def generate_bipartite_graph(n_left: int, n_right: int, edge_probability: float = 0.5, seed: int = None) -> List[Tuple[int, int]]:
    """
    Genera un grafo bipartito aleatorio.
    
    Los nodos se dividen en dos conjuntos: [0, n_left) y [n_left, n_left + n_right).
    Solo hay aristas entre nodos de diferentes conjuntos.
    
    Args:
        n_left: Número de nodos en el conjunto izquierdo
        n_right: Número de nodos en el conjunto derecho
        edge_probability: Probabilidad de arista entre nodos de diferentes conjuntos
        seed: Semilla para reproducibilidad (opcional)
        
    Returns:
        Lista de aristas como tuplas (i, j)
        
    Raises:
        ValueError: Si los parámetros son inválidos
    """
    if n_left < 1 or n_right < 1:
        raise ValueError(f"n_left y n_right deben ser >= 1, recibido {n_left}, {n_right}")
    if not 0.0 <= edge_probability <= 1.0:
        raise ValueError(f"edge_probability debe estar en [0.0, 1.0], recibido {edge_probability}")
    
    if seed is not None:
        random.seed(seed)
    
    edges = []
    for i in range(n_left):
        for j in range(n_left, n_left + n_right):
            if random.random() < edge_probability:
                edges.append((i, j))
    
    logger.debug(f"Generado grafo bipartito: {n_left}+{n_right} nodos, {len(edges)} aristas")
    return edges


def generate_grid_graph(rows: int, cols: int, diagonal: bool = False) -> List[Tuple[int, int]]:
    """
    Genera un grafo de cuadrícula (grid).
    
    Los nodos se numeran de 0 a rows*cols-1, donde el nodo en (row, col)
    tiene índice row*cols + col.
    
    Args:
        rows: Número de filas
        cols: Número de columnas
        diagonal: Si True, incluye conexiones diagonales
        
    Returns:
        Lista de aristas como tuplas (i, j) donde i < j
        
    Raises:
        ValueError: Si rows < 1 o cols < 1
    """
    if rows < 1 or cols < 1:
        raise ValueError(f"rows y cols deben ser >= 1, recibido {rows}, {cols}")
    
    edges = []
    
    def node_index(r: int, c: int) -> int:
        return r * cols + c
    
    for r in range(rows):
        for c in range(cols):
            current = node_index(r, c)
            
            # Conexión derecha
            if c < cols - 1:
                edges.append((current, node_index(r, c + 1)))
            
            # Conexión abajo
            if r < rows - 1:
                edges.append((current, node_index(r + 1, c)))
            
            # Conexiones diagonales
            if diagonal:
                # Diagonal abajo-derecha
                if r < rows - 1 and c < cols - 1:
                    edges.append((current, node_index(r + 1, c + 1)))
                
                # Diagonal abajo-izquierda
                if r < rows - 1 and c > 0:
                    edges.append((current, node_index(r + 1, c - 1)))
    
    logger.debug(f"Generado grafo grid: {rows}x{cols}, {len(edges)} aristas, diagonal={diagonal}")
    return edges


def generate_cycle_graph(n_nodes: int) -> List[Tuple[int, int]]:
    """
    Genera un grafo cíclico (ciclo).
    
    Cada nodo i está conectado a (i+1) % n_nodes.
    
    Args:
        n_nodes: Número de nodos
        
    Returns:
        Lista de aristas como tuplas (i, j)
        
    Raises:
        ValueError: Si n_nodes < 3
    """
    if n_nodes < 3:
        raise ValueError(f"Un ciclo requiere al menos 3 nodos, recibido {n_nodes}")
    
    edges = []
    for i in range(n_nodes):
        j = (i + 1) % n_nodes
        edges.append((min(i, j), max(i, j)))
    
    logger.debug(f"Generado grafo cíclico: {n_nodes} nodos, {len(edges)} aristas")
    return edges


def generate_path_graph(n_nodes: int) -> List[Tuple[int, int]]:
    """
    Genera un grafo de camino (path).
    
    Cada nodo i está conectado a i+1 (excepto el último).
    
    Args:
        n_nodes: Número de nodos
        
    Returns:
        Lista de aristas como tuplas (i, j)
        
    Raises:
        ValueError: Si n_nodes < 2
    """
    if n_nodes < 2:
        raise ValueError(f"Un camino requiere al menos 2 nodos, recibido {n_nodes}")
    
    edges = []
    for i in range(n_nodes - 1):
        edges.append((i, i + 1))
    
    logger.debug(f"Generado grafo de camino: {n_nodes} nodos, {len(edges)} aristas")
    return edges


def generate_star_graph(n_nodes: int) -> List[Tuple[int, int]]:
    """
    Genera un grafo estrella.
    
    Un nodo central (0) conectado a todos los demás.
    
    Args:
        n_nodes: Número total de nodos (incluyendo el centro)
        
    Returns:
        Lista de aristas como tuplas (i, j)
        
    Raises:
        ValueError: Si n_nodes < 2
    """
    if n_nodes < 2:
        raise ValueError(f"Un grafo estrella requiere al menos 2 nodos, recibido {n_nodes}")
    
    edges = []
    for i in range(1, n_nodes):
        edges.append((0, i))
    
    logger.debug(f"Generado grafo estrella: {n_nodes} nodos, {len(edges)} aristas")
    return edges


def generate_wheel_graph(n_nodes: int) -> List[Tuple[int, int]]:
    """
    Genera un grafo rueda (wheel).
    
    Un nodo central (0) conectado a todos los demás, que forman un ciclo.
    
    Args:
        n_nodes: Número total de nodos (incluyendo el centro)
        
    Returns:
        Lista de aristas como tuplas (i, j)
        
    Raises:
        ValueError: Si n_nodes < 4
    """
    if n_nodes < 4:
        raise ValueError(f"Un grafo rueda requiere al menos 4 nodos, recibido {n_nodes}")
    
    edges = []
    
    # Conexiones del centro a todos
    for i in range(1, n_nodes):
        edges.append((0, i))
    
    # Ciclo exterior
    for i in range(1, n_nodes):
        j = i + 1 if i < n_nodes - 1 else 1
        edges.append((min(i, j), max(i, j)))
    
    logger.debug(f"Generado grafo rueda: {n_nodes} nodos, {len(edges)} aristas")
    return edges


def edges_to_adjacency_dict(edges: List[Tuple[int, int]], n_nodes: int = None) -> dict:
    """
    Convierte una lista de aristas a un diccionario de adyacencia.
    
    Args:
        edges: Lista de aristas como tuplas (i, j)
        n_nodes: Número total de nodos (opcional, se infiere si no se proporciona)
        
    Returns:
        Dict donde cada clave es un nodo y el valor es un set de vecinos
    """
    if n_nodes is None:
        n_nodes = max(max(i, j) for i, j in edges) + 1 if edges else 0
    
    adj = {i: set() for i in range(n_nodes)}
    
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)
    
    return adj


def get_graph_chromatic_number_lower_bound(edges: List[Tuple[int, int]], n_nodes: int) -> int:
    """
    Calcula una cota inferior del número cromático del grafo.
    
    Usa el tamaño del clique máximo encontrado mediante heurística.
    
    Args:
        edges: Lista de aristas
        n_nodes: Número de nodos
        
    Returns:
        Cota inferior del número cromático
    """
    if not edges:
        return 1
    
    adj = edges_to_adjacency_dict(edges, n_nodes)
    
    # Encontrar el grado máximo
    max_degree = max(len(neighbors) for neighbors in adj.values())
    
    # El número cromático es al menos max_degree + 1 para grafos regulares
    # Para grafos generales, es al menos el tamaño del clique máximo
    # Como aproximación simple, usamos max_degree + 1
    return max_degree + 1

