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
        variables = list(csp.variables)
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
    
    def _partition_symmetry(self, csp, k: int) -> List[Set[str]]:
        """
        Partición por simetría: agrupa variables simétricas.
        
        Variables son simétricas si tienen:
        - Mismo dominio
        - Mismas restricciones (módulo permutación)
        """
        # Detectar grupos de simetría
        symmetry_groups = self._detect_symmetry_groups(csp)
        
        if len(symmetry_groups) >= k:
            # Ya hay suficientes grupos
            symmetry_groups = sorted(symmetry_groups, key=len, reverse=True)[:k]
            return symmetry_groups
        
        # Fusionar grupos pequeños
        partition = list(symmetry_groups)
        while len(partition) < k:
            # Dividir grupo más grande
            largest = max(partition, key=len)
            partition.remove(largest)
            
            # Dividir en dos
            half = len(largest) // 2
            partition.append(set(list(largest)[:half]))
            partition.append(set(list(largest)[half:]))
        
        # Si hay demasiados grupos, fusionar los más pequeños
        while len(partition) > k:
            smallest1 = min(partition, key=len)
            partition.remove(smallest1)
            smallest2 = min(partition, key=len)
            partition.remove(smallest2)
            partition.append(smallest1.union(smallest2))
        
        return partition
    
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
            u_idx = node_to_idx[u]
            v_idx = node_to_idx[v]
            adjacency[u_idx].append(v_idx)
            adjacency[v_idx].append(u_idx)
        
        # Particionar con metis
        try:
            # pymetis.part_graph can take a NetworkX graph directly.
            # The documentation suggests passing the graph object and nparts as positional arguments.
            # pymetis.part_graph expects nparts as the first argument, then the adjacency.
            # It can directly take a NetworkX graph as the adjacency argument.
            # pymetis expects integer node IDs. Convert NetworkX graph to use integer nodes.
            G_undirected = G.to_undirected()
            node_to_int = {node: i for i, node in enumerate(G_undirected.nodes())}
            int_to_node = {i: node for i, node in enumerate(G_undirected.nodes())}

            # Create a new graph with integer nodes for pymetis
            G_int_nodes = nx.Graph()
            G_int_nodes.add_nodes_from(range(len(G_undirected.nodes())))
            for u, v in G_undirected.edges():
                G_int_nodes.add_edge(node_to_int[u], node_to_int[v])

            edgecuts, parts = metis.part_graph(k, adjacency=G_int_nodes)

            partition = [set() for _ in range(k)]
            for idx, part_id in enumerate(parts):
                original_node = int_to_node[idx]
                partition[part_id].add(original_node)
            return partition
        except Exception as e:
            print(f"Warning: metis failed ({e}), falling back to topological")
            return self._partition_topological(csp, k)
    
    def _build_constraint_graph(self, csp) -> nx.Graph:
        """
        Construye grafo de restricciones del CSP.
        
        Nodos: variables
        Edges: restricciones (peso = fuerza de la restricción)
        """
        G = nx.Graph()
        
        # Añadir nodos (variables)
        G.add_nodes_from(csp.variables)
        
        # Añadir edges (restricciones)
        for constraint in csp.constraints:
            # Obtener variables involucradas
            vars_involved = constraint.scope
            
            # Para restricciones binarias, añadir edge directo
            if len(vars_involved) == 2:
                var1, var2 = vars_involved
                weight = self._estimate_constraint_strength(constraint)
                G.add_edge(var1, var2, weight=weight)
            
            # Para restricciones n-arias, añadir clique
            elif len(vars_involved) > 2:
                for i, var1 in enumerate(vars_involved):
                    for var2 in vars_involved[i+1:]:
                        weight = self._estimate_constraint_strength(constraint)
                        if G.has_edge(var1, var2):
                            G[var1][var2]['weight'] += weight
                        else:
                            G.add_edge(var1, var2, weight=weight)
        
        return G
    
    def _estimate_constraint_strength(self, constraint) -> float:
        """
        Estima fuerza de una restricción.
        
        Fuerza = qué tan restrictiva es la restricción
        """
        # Placeholder: Implementar lógica real para estimar la fuerza de la restricción
        # Por ejemplo, basada en el tamaño del dominio de las variables involucradas
        # o la complejidad de la función de relación.
        return 1.0 # Por ahora, todas las restricciones tienen la misma fuerza

    def _detect_symmetry_groups(self, csp) -> List[Set[str]]:
        """
        Detecta grupos de variables simétricas en el CSP.

        Variables son simétricas si tienen:
        - Mismo dominio
        - Mismas restricciones (módulo permutación)
        """
        # Placeholder: Implementar lógica real para detectar simetrías.
        # Esto es un problema complejo en sí mismo y puede requerir algoritmos
        # de isomorfismo de grafos o análisis de automorfismos.
        
        # Por ahora, una implementación trivial: cada variable es su propio grupo.
        return [{var} for var in csp.variables]

    def _estimate_difficulty(self, csp) -> Dict[str, float]:
        """
        Estima la dificultad de cada variable en el CSP.

        La dificultad puede basarse en:
        - Grado de la variable en el grafo de restricciones
        - Tamaño de su dominio
        - Número y fuerza de las restricciones en las que participa
        """
        difficulty_map: Dict[str, float] = {}
        G = self._build_constraint_graph(csp)

        for var in csp.variables:
            degree = G.degree(var) if G.has_node(var) else 0
            domain_size = len(csp.domains.get(var, []))
            
            # Heurística simple: mayor grado, menor dominio -> mayor dificultad
            difficulty = float(degree) / (domain_size + 1) # Evitar división por cero
            difficulty_map[var] = difficulty
        
        return difficulty_map

    def _balance_partition(self, partition: List[Set[str]], csp) -> List[Set[str]]:
        """
        Balancea el tamaño de los grupos en una partición.
        
        Args:
            partition: La partición a balancear.
            csp: El CSP original (para acceder a las variables).
            
        Returns:
            Partición balanceada
        """
        if not partition:
            return partition
        
        avg_size = sum(len(group) for group in partition) / len(partition)
        
        # Mientras haya desbalance significativo
        max_iterations = 100 # Para evitar bucles infinitos
        iteration = 0
        while iteration < max_iterations:
            largest_group_idx = -1
            smallest_group_idx = -1
            max_size = -1
            min_size = float('inf')

            for i, group in enumerate(partition):
                if len(group) > max_size:
                    max_size = len(group)
                    largest_group_idx = i
                if len(group) < min_size:
                    min_size = len(group)
                    smallest_group_idx = i
            
            # Si el desbalance es aceptable, salir
            if max_size - min_size <= 1 or largest_group_idx == -1 or smallest_group_idx == -1:
                break
            
            # Mover una variable del grupo más grande al más pequeño
            # Elegir la variable que menos impacte las restricciones inter-grupo
            # (esto es una heurística simple, se podría mejorar con análisis de edge-cut)
            
            # Por simplicidad, movemos una variable arbitraria del grupo más grande
            if partition[largest_group_idx]: # Asegurarse de que el grupo no esté vacío
                var_to_move = next(iter(partition[largest_group_idx]))
                partition[largest_group_idx].remove(var_to_move)
                partition[smallest_group_idx].add(var_to_move)
            else:
                # Si el grupo más grande está vacío, algo salió mal o ya está balanceado
                break
            
            iteration += 1
        
        return partition

