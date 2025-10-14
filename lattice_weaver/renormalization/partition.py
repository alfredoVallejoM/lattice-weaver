# lattice_weaver/renormalization/partition.py

"""
Particionamiento de Variables para Renormalización

Este módulo implementa estrategias de particionamiento de variables que
minimizan el edge-cut del grafo de restricciones, preparando el terreno
para la renormalización computacional.

Estrategias implementadas:
- Topológica: Basada en componentes conexas y puentes
- Por simetría: Agrupa variables simétricas
- Adaptativa: Se ajusta según dificultad estimada
- Metis: Usa algoritmo Metis para particionamiento óptimo
"""

from typing import List, Set, Dict, Tuple, Optional, Callable
from collections import defaultdict
import networkx as nx
import numpy as np


class VariablePartitioner:
    """
    Particionador de variables para renormalización.
    
    Implementa múltiples estrategias de particionamiento que minimizan
    el edge-cut del grafo de restricciones.
    """
    
    def __init__(self, strategy: str = 'topological'):
        """
        Inicializa particionador.
        
        Args:
            strategy: Estrategia de particionamiento
                     ('topological', 'symmetry', 'adaptive', 'metis')
        """
        self.strategy = strategy
        
        self.strategies: Dict[str, Callable] = {
            'topological': self._partition_topological,
            'symmetry': self._partition_symmetry,
            'adaptive': self._partition_adaptive,
            'metis': self._partition_metis,
            'simple': self._partition_simple
        }
        
        if strategy not in self.strategies:
            raise ValueError(f"Unknown strategy: {strategy}. "
                           f"Available: {list(self.strategies.keys())}")
    
    def partition(self, csp, k: int) -> List[Set[str]]:
        """
        Particiona variables del CSP en k grupos.
        
        Args:
            csp: CSP a particionar
            k: Número de grupos
        
        Returns:
            Lista de k conjuntos de variables
        """
        return self.strategies[self.strategy](csp, k)
    
    def _partition_simple(self, csp, k: int) -> List[Set[str]]:
        """
        Partición simple: divide variables secuencialmente.
        
        Útil para testing y baseline.
        """
        variables = sorted(list(csp.variables))
        n = len(variables)
        
        if n == 0:
            return []

        # Asegurarse de que k no sea mayor que el número de variables
        # Si k es mayor, cada variable irá en su propio grupo y los grupos restantes estarán vacíos
        k_actual = min(k, n)

        group_size = n // k_actual
        remainder = n % k_actual
        
        partition = []
        current_idx = 0
        for i in range(k_actual):
            size = group_size + (1 if i < remainder else 0)
            partition.append(set(variables[current_idx : current_idx + size]))
            current_idx += size
        
        # Si k original era mayor que n, añadir grupos vacíos para completar k
        while len(partition) < k:
            partition.append(set())

        return partition
    
    def _partition_topological(self, csp, k: int) -> List[Set[str]]:
        """
        Partición topológica: usa estructura del grafo de restricciones.
        
        Estrategia:
        1. Detectar componentes conexas
        2. Si hay suficientes componentes, usarlas
        3. Si no, detectar puentes y removerlos para crear componentes
        4. Balancear tamaños de grupos
        """
        # Construir grafo de restricciones
        G = self._build_constraint_graph(csp)
        
        # Detectar componentes conexas
        components = list(nx.connected_components(G))
        
        if len(components) >= k:
            # Ya hay suficientes componentes
            # Tomar las k más grandes
            components = sorted(components, key=len, reverse=True)[:k]
            return components
        
        # Necesitamos dividir más
        # Detectar puentes (edges cuya remoción desconecta)
        bridges = list(nx.bridges(G))
        
        if bridges:
            # Remover puentes para crear más componentes
            G_copy = G.copy()
            G_copy.remove_edges_from(bridges)
            components = list(nx.connected_components(G_copy))
            
            if len(components) >= k:
                components = sorted(components, key=len, reverse=True)[:k]
                return components
        
        # Si aún no hay suficientes componentes, usar partición simple
        return self._partition_simple(csp, k)

    def _detect_symmetry_groups(self, csp) -> List[Set[str]]:
        """
        Detecta grupos de variables simétricas basándose en una firma heurística.
        """
        signatures = defaultdict(list)
        constraint_graph = self._build_constraint_graph(csp)

        
        for var in sorted(list(csp.variables)):
            domain_size = len(csp.domains[var])
            degree = constraint_graph.degree(var)
            
            # Crear una firma para las restricciones de la variable
            neighbor_signatures = []
            # Ordenar los vecinos para asegurar una firma determinista
            for neighbor in sorted(list(constraint_graph.neighbors(var))):
                # Firma del vecino (simplificada)
                neighbor_sig = (len(csp.domains[neighbor]), constraint_graph.degree(neighbor))
                neighbor_signatures.append(neighbor_sig)
            
            # Usar un frozenset para que el orden no importe
            constraints_signature = frozenset(neighbor_signatures)
            
            # Firma completa de la variable
            signature = (domain_size, degree, constraints_signature)
            signatures[signature].append(var)
            
        # Devolver los grupos de variables con la misma firma
        # print(f"DEBUG (partition.py): Final signatures dict: {dict(signatures)}") # Comentado para evitar output excesivo
        return [set(group) for group in signatures.values()]
    
    def _partition_symmetry(self, csp, k: int) -> List[Set[str]]:
        """
        Partición por simetría: agrupa variables simétricas.
        
        Variables son simétricas si tienen:
        - Mismo dominio
        - Mismas restricciones (módulo permutación)
        """
        # Detectar grupos de simetría
        symmetry_groups = self._detect_symmetry_groups(csp)
        
        # Si no se encuentran grupos de simetría o no hay suficientes, recurrir a _partition_simple
        # Si todos los grupos de simetría son singletons, la detección no fue efectiva.
        # En este caso, es mejor usar una partición simple.
        all_singletons = all(len(g) == 1 for g in symmetry_groups)

        if not symmetry_groups or len(symmetry_groups) < k or all_singletons:
            return self._partition_simple(csp, k)

        # Si hay suficientes grupos de simetría, usarlos
        # Tomar los k grupos más grandes si hay más de k
        if len(symmetry_groups) > k:
            symmetry_groups = sorted(symmetry_groups, key=len, reverse=True)[:k]
        
        return list(symmetry_groups)
    
    def _partition_adaptive(self, csp, k: int) -> List[Set[str]]:
        """
        Partición adaptativa: se ajusta según dificultad estimada.
        
        Estrategia:
        1. Estimar dificultad de cada variable
        2. Asignar variables a grupos balanceando dificultad
        """
        # Estimar dificultad de cada variable
        difficulty_map = self._estimate_difficulty(csp)
        
        # Inicializar k grupos vacíos
        partition = [set() for _ in range(k)]
        group_difficulties = [0.0] * k
        
        # Ordenar variables por dificultad (descendente)
        sorted_vars = sorted(csp.variables, 
                           key=lambda v: difficulty_map.get(v, 0.0), 
                           reverse=True)
        
        # Asignar cada variable al grupo con menor dificultad acumulada
        for var in sorted_vars:
            min_difficulty_idx = np.argmin(group_difficulties)
            partition[min_difficulty_idx].add(var)
            group_difficulties[min_difficulty_idx] += difficulty_map.get(var, 0.0)
        
        return partition
    
    def _partition_metis(self, csp, k: int) -> List[Set[str]]:
        """
        Partición usando algoritmo Metis.
        
        Metis es un algoritmo de particionamiento de grafos que minimiza
        el edge-cut. Es óptimo para muchos casos.
        
        Nota: Requiere librería metis. Si no está disponible, fallback a topológica.
        """
        try:
            import pymetis as metis
        except ImportError:
            print("Warning: metis not available, falling back to topological partitioning")
            return self._partition_topological(csp, k)
        
        # Construir grafo de restricciones
        G = self._build_constraint_graph(csp)
        
        # Convertir a formato de metis
        # metis requiere grafo no dirigido con nodos numerados 0..n-1
        node_to_idx = {node: idx for idx, node in enumerate(G.nodes())}
        idx_to_node = {idx: node for node, idx in node_to_idx.items()}
        
        # Crear lista de adyacencia
        adjacency = [[] for _ in range(len(G.nodes()))]
        for u, v in G.edges():
            adjacency[node_to_idx[u]].append(node_to_idx[v])
            adjacency[node_to_idx[v]].append(node_to_idx[u]) # Metis espera grafo no dirigido
        
        # Llamar a metis
        # nparts: número de particiones
        # xadj, adjncy: representación del grafo
        # vwgt: pesos de los vértices (None para pesos unitarios)
        # adjwgt: pesos de las aristas (None para pesos unitarios)
        # tpwgts: pesos de las particiones (None para particiones balanceadas)
        # ubvec: vector de desbalanceo (None para default)
        
        edge_cut, parts = metis.part_graph(k, adjacency)
        
        # Convertir resultado de metis a formato de sets de variables
        partition = [set() for _ in range(k)]
        for idx, part_id in enumerate(parts):
            partition[part_id].add(idx_to_node[idx])
            
        return partition
    
    def _build_constraint_graph(self, csp) -> nx.Graph:
        """
        Construye un grafo de restricciones donde los nodos son variables
        y las aristas representan restricciones entre ellas.
        """
        G = nx.Graph()
        G.add_nodes_from(csp.variables)
        
        for constraint in csp.constraints:
            # Solo consideramos restricciones binarias para el grafo
            if len(constraint.scope) == 2:
                v1, v2 = tuple(constraint.scope)
                G.add_edge(v1, v2)
            elif len(constraint.scope) > 2:
                # Para restricciones de aridad > 2, añadimos aristas entre todas las parejas
                # de variables en el scope (clique)
                for v1, v2 in itertools.combinations(constraint.scope, 2):
                    G.add_edge(v1, v2)
        return G
    
    def _estimate_difficulty(self, csp) -> Dict[str, float]:
        """
        Estima la dificultad de cada variable.
        
        Heurística simple: número de restricciones en las que participa la variable
        multiplicado por el logaritmo del tamaño de su dominio.
        """
        difficulty_map = {}
        G = self._build_constraint_graph(csp)
        
        for var in csp.variables:
            degree = G.degree(var) # Número de restricciones en las que participa
            domain_size = len(csp.domains[var])
            
            # Evitar log(0) si el dominio está vacío (aunque no debería pasar)
            if domain_size > 0:
                difficulty = degree * np.log(domain_size)
            else:
                difficulty = float('inf') # Variable imposible
            
            difficulty_map[var] = difficulty
            
        return difficulty_map

