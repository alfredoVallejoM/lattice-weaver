"""
Detección y Gestión de Clústeres para el Motor de Coherencia Adaptativa (ACE).

Este módulo implementa la detección automática de clústeres de variables
fuertemente acopladas usando el algoritmo de Louvain.

Autor: LatticeWeaver Team
Fecha: 12 de Octubre de 2025
Versión: 4.2.0
"""

from typing import Dict, Set, List, Tuple, Optional
from dataclasses import dataclass
import networkx as nx
from collections import defaultdict

from .graph_structures import ConstraintGraph, DynamicClusterGraph, Cluster


@dataclass
class ClusteringMetrics:
    """
    Métricas de calidad del clustering.
    
    Attributes:
        modularity: Modularidad del clustering (0-1, mayor es mejor)
        num_clusters: Número de clústeres detectados
        avg_cluster_size: Tamaño promedio de clústeres
        max_cluster_size: Tamaño del clúster más grande
        min_cluster_size: Tamaño del clúster más pequeño
        boundary_density: Densidad de restricciones de frontera
    """
    modularity: float
    num_clusters: int
    avg_cluster_size: float
    max_cluster_size: int
    min_cluster_size: int
    boundary_density: float
    
    def __repr__(self) -> str:
        return (f"ClusteringMetrics(modularity={self.modularity:.3f}, "
                f"clusters={self.num_clusters}, "
                f"avg_size={self.avg_cluster_size:.1f})")


class ClusterDetector:
    """
    Detector de clústeres usando el algoritmo de Louvain.
    
    Identifica comunidades de variables fuertemente acopladas en el
    grafo de restricciones y construye el grafo de clústeres dinámico.
    
    El algoritmo de Louvain maximiza la modularidad del grafo, agrupando
    variables que tienen muchas restricciones entre sí y pocas con el resto.
    
    Examples:
        >>> detector = ClusterDetector(min_cluster_size=2, max_cluster_size=10)
        >>> gcd, metrics = detector.detect_clusters(constraint_graph)
        >>> print(f"Detectados {metrics.num_clusters} clústeres")
    """
    
    def __init__(
        self,
        min_cluster_size: int = 2,
        max_cluster_size: int = 20,
        resolution: float = 1.0,
        randomize: bool = False,
        random_state: Optional[int] = None
    ):
        """
        Inicializa el detector de clústeres.
        
        Args:
            min_cluster_size: Tamaño mínimo de clúster (variables)
            max_cluster_size: Tamaño máximo de clúster (variables)
            resolution: Parámetro de resolución de Louvain (>1 = más clústeres)
            randomize: Si usar inicialización aleatoria
            random_state: Semilla para reproducibilidad
        """
        self.min_cluster_size = min_cluster_size
        self.max_cluster_size = max_cluster_size
        self.resolution = resolution
        self.randomize = randomize
        self.random_state = random_state
    
    def detect_clusters(
        self, 
        cg: ConstraintGraph
    ) -> Tuple[DynamicClusterGraph, ClusteringMetrics]:
        """
        Detecta clústeres en el grafo de restricciones.
        
        Args:
            cg: Grafo de restricciones
        
        Returns:
            Tupla con (DynamicClusterGraph, ClusteringMetrics)
        
        Raises:
            ValueError: Si el grafo está vacío
        """
        if len(cg.get_all_variables()) == 0:
            raise ValueError("Cannot detect clusters in empty graph")
        
        # Paso 1: Ejecutar algoritmo de Louvain
        communities = self._run_louvain(cg)
        
        # Paso 2: Post-procesar comunidades (merge/split según tamaños)
        communities = self._postprocess_communities(communities, cg)
        
        # Paso 3: Construir grafo de clústeres dinámico
        gcd = self._build_cluster_graph(communities, cg)
        
        # Paso 4: Calcular métricas
        metrics = self._compute_metrics(gcd, cg, communities)
        
        return gcd, metrics
    
    def _run_louvain(self, cg: ConstraintGraph) -> Dict[str, int]:
        """
        Ejecuta el algoritmo de Louvain sobre el grafo de restricciones.
        
        Args:
            cg: Grafo de restricciones
        
        Returns:
            Diccionario {variable: community_id}
        """
        # Usar NetworkX para Louvain (implementación simplificada)
        # En producción, usar python-louvain para mejor rendimiento
        
        try:
            import community as community_louvain
            # Usar librería community si está disponible
            communities = community_louvain.best_partition(
                cg.graph.to_undirected(),
                resolution=self.resolution,
                randomize=self.randomize,
                random_state=self.random_state
            )
        except ImportError:
            # Fallback: usar greedy_modularity_communities de NetworkX
            from networkx.algorithms import community as nx_community
            
            communities_sets = nx_community.greedy_modularity_communities(
                cg.graph.to_undirected(),
                resolution=self.resolution
            )
            
            # Convertir a formato {variable: community_id}
            communities = {}
            for comm_id, comm_set in enumerate(communities_sets):
                for var in comm_set:
                    communities[var] = comm_id
        
        return communities
    
    def _postprocess_communities(
        self, 
        communities: Dict[str, int],
        cg: ConstraintGraph
    ) -> Dict[str, int]:
        """
        Post-procesa comunidades para respetar límites de tamaño.
        
        Args:
            communities: Diccionario {variable: community_id}
            cg: Grafo de restricciones
        
        Returns:
            Diccionario actualizado {variable: community_id}
        """
        # Agrupar variables por comunidad
        comm_to_vars: Dict[int, Set[str]] = defaultdict(set)
        for var, comm_id in communities.items():
            comm_to_vars[comm_id].add(var)
        
        # Procesar comunidades iterativamente hasta que todas cumplan límites
        max_iterations = 10
        iteration = 0
        
        while iteration < max_iterations:
            needs_processing = False
            new_comm_to_vars: Dict[int, Set[str]] = {}
            next_comm_id = max(comm_to_vars.keys()) + 1 if comm_to_vars else 0
            
            for comm_id, variables in comm_to_vars.items():
                size = len(variables)
                
                if size < self.min_cluster_size:
                    # Comunidad muy pequeña: fusionar con vecina más conectada
                    needs_processing = True
                    merged_id = self._merge_small_community_id(
                        variables, comm_to_vars, cg
                    )
                    if merged_id in new_comm_to_vars:
                        new_comm_to_vars[merged_id] |= variables
                    else:
                        new_comm_to_vars[merged_id] = set(variables)
                
                elif size > self.max_cluster_size:
                    # Comunidad muy grande: dividir recursivamente
                    needs_processing = True
                    sub_communities = self._split_large_community(
                        variables, cg, next_comm_id
                    )
                    # Agrupar por sub_comm_id
                    for var, sub_comm_id in sub_communities.items():
                        if sub_comm_id not in new_comm_to_vars:
                            new_comm_to_vars[sub_comm_id] = set()
                        new_comm_to_vars[sub_comm_id].add(var)
                    next_comm_id += len(set(sub_communities.values()))
                
                else:
                    # Comunidad de tamaño adecuado
                    new_comm_to_vars[comm_id] = set(variables)
            
            comm_to_vars = new_comm_to_vars
            iteration += 1
            
            if not needs_processing:
                break
        
        # Convertir de vuelta a formato {variable: community_id}
        new_communities = {}
        for comm_id, variables in comm_to_vars.items():
            for var in variables:
                new_communities[var] = comm_id
        
        return new_communities
    
    def _merge_small_community_id(
        self,
        variables: Set[str],
        comm_to_vars: Dict[int, Set[str]],
        cg: ConstraintGraph
    ) -> int:
        """
        Encuentra el ID de comunidad con la que fusionar una comunidad pequeña.
        
        Args:
            variables: Variables de la comunidad pequeña
            comm_to_vars: Mapa de comunidades a variables
            cg: Grafo de restricciones
        
        Returns:
            ID de la comunidad con la que fusionar
        """
        # Contar conexiones con otras comunidades
        connections: Dict[int, int] = defaultdict(int)
        
        for var in variables:
            for neighbor in cg.get_neighbors(var):
                # Encontrar comunidad del vecino
                for comm_id, comm_vars in comm_to_vars.items():
                    if neighbor in comm_vars and neighbor not in variables:
                        connections[comm_id] += 1
        
        if not connections:
            # Sin conexiones: crear comunidad nueva
            return max(comm_to_vars.keys()) + 1 if comm_to_vars else 0
        
        # Fusionar con la comunidad más conectada
        return max(connections.items(), key=lambda x: x[1])[0]
    
    def _merge_small_community(
        self,
        variables: Set[str],
        comm_to_vars: Dict[int, Set[str]],
        cg: ConstraintGraph
    ) -> int:
        """
        Fusiona una comunidad pequeña con su vecina más conectada.
        
        Args:
            variables: Variables de la comunidad pequeña
            comm_to_vars: Mapa de comunidades a variables
            cg: Grafo de restricciones
        
        Returns:
            ID de la comunidad con la que fusionar
        """
        # Contar conexiones con otras comunidades
        connections: Dict[int, int] = defaultdict(int)
        
        for var in variables:
            for neighbor in cg.get_neighbors(var):
                # Encontrar comunidad del vecino
                for comm_id, comm_vars in comm_to_vars.items():
                    if neighbor in comm_vars and neighbor not in variables:
                        connections[comm_id] += 1
        
        if not connections:
            # Sin conexiones: crear comunidad singleton
            return max(comm_to_vars.keys()) + 1
        
        # Fusionar con la comunidad más conectada
        return max(connections.items(), key=lambda x: x[1])[0]
    
    def _split_large_community(
        self,
        variables: Set[str],
        cg: ConstraintGraph,
        start_id: int
    ) -> Dict[str, int]:
        """
        Divide una comunidad grande recursivamente.
        
        Args:
            variables: Variables de la comunidad grande
            cg: Grafo de restricciones
            start_id: ID inicial para nuevas comunidades
        
        Returns:
            Diccionario {variable: new_community_id}
        """
        # Crear subgrafo con solo estas variables
        subgraph = cg.graph.subgraph(variables).copy()
        
        # Ejecutar Louvain en el subgrafo
        try:
            import community as community_louvain
            sub_communities = community_louvain.best_partition(
                subgraph.to_undirected(),
                resolution=self.resolution * 1.5  # Mayor resolución para dividir
            )
        except ImportError:
            from networkx.algorithms import community as nx_community
            communities_sets = nx_community.greedy_modularity_communities(
                subgraph.to_undirected()
            )
            sub_communities = {}
            for i, comm_set in enumerate(communities_sets):
                for var in comm_set:
                    sub_communities[var] = i
        
        # Renumerar IDs
        unique_ids = sorted(set(sub_communities.values()))
        id_mapping = {old_id: start_id + i for i, old_id in enumerate(unique_ids)}
        
        return {var: id_mapping[comm_id] for var, comm_id in sub_communities.items()}
    
    def _build_cluster_graph(
        self,
        communities: Dict[str, int],
        cg: ConstraintGraph
    ) -> DynamicClusterGraph:
        """
        Construye el grafo de clústeres dinámico a partir de comunidades.
        
        Args:
            communities: Diccionario {variable: community_id}
            cg: Grafo de restricciones
        
        Returns:
            DynamicClusterGraph construido
        """
        gcd = DynamicClusterGraph()
        
        # Agrupar variables por comunidad
        comm_to_vars: Dict[int, Set[str]] = defaultdict(set)
        for var, comm_id in communities.items():
            comm_to_vars[comm_id].add(var)
        
        # Crear clústeres
        comm_id_to_cluster_id: Dict[int, int] = {}
        for comm_id, variables in comm_to_vars.items():
            cluster_id = gcd.add_cluster(variables, state="ACTIVE")
            comm_id_to_cluster_id[comm_id] = cluster_id
        
        # Detectar y añadir fronteras
        boundaries = self._detect_boundaries(communities, cg)
        for comm1, comm2 in boundaries:
            cluster1 = comm_id_to_cluster_id[comm1]
            cluster2 = comm_id_to_cluster_id[comm2]
            gcd.add_boundary_constraint(cluster1, cluster2)
        
        return gcd
    
    def _detect_boundaries(
        self,
        communities: Dict[str, int],
        cg: ConstraintGraph
    ) -> Set[Tuple[int, int]]:
        """
        Detecta restricciones de frontera entre comunidades.
        
        Args:
            communities: Diccionario {variable: community_id}
            cg: Grafo de restricciones
        
        Returns:
            Conjunto de pares (comm1, comm2) con restricciones entre ellos
        """
        boundaries = set()
        
        for var in cg.get_all_variables():
            var_comm = communities[var]
            
            for neighbor in cg.get_neighbors(var):
                neighbor_comm = communities[neighbor]
                
                if var_comm != neighbor_comm:
                    # Restricción de frontera
                    boundary = tuple(sorted([var_comm, neighbor_comm]))
                    boundaries.add(boundary)
        
        return boundaries
    
    def _compute_metrics(
        self,
        gcd: DynamicClusterGraph,
        cg: ConstraintGraph,
        communities: Dict[str, int]
    ) -> ClusteringMetrics:
        """
        Calcula métricas de calidad del clustering.
        
        Args:
            gcd: Grafo de clústeres dinámico
            cg: Grafo de restricciones
            communities: Diccionario {variable: community_id}
        
        Returns:
            ClusteringMetrics con métricas calculadas
        """
        # Calcular modularidad
        modularity = self._compute_modularity(cg, communities)
        
        # Estadísticas de clústeres
        clusters = gcd.get_all_clusters()
        num_clusters = len(clusters)
        cluster_sizes = [len(c.variables) for c in clusters]
        
        avg_size = sum(cluster_sizes) / num_clusters if num_clusters > 0 else 0
        max_size = max(cluster_sizes) if cluster_sizes else 0
        min_size = min(cluster_sizes) if cluster_sizes else 0
        
        # Densidad de fronteras
        num_boundaries = len(gcd.graph.edges())
        max_boundaries = num_clusters * (num_clusters - 1) / 2
        boundary_density = num_boundaries / max_boundaries if max_boundaries > 0 else 0
        
        return ClusteringMetrics(
            modularity=modularity,
            num_clusters=num_clusters,
            avg_cluster_size=avg_size,
            max_cluster_size=max_size,
            min_cluster_size=min_size,
            boundary_density=boundary_density
        )
    
    def _compute_modularity(
        self,
        cg: ConstraintGraph,
        communities: Dict[str, int]
    ) -> float:
        """
        Calcula la modularidad del clustering.
        
        La modularidad mide qué tan bien el clustering agrupa variables
        fuertemente conectadas. Valores cercanos a 1 son mejores.
        
        Args:
            cg: Grafo de restricciones
            communities: Diccionario {variable: community_id}
        
        Returns:
            Modularidad (0-1)
        """
        try:
            import community as community_louvain
            return community_louvain.modularity(communities, cg.graph.to_undirected())
        except ImportError:
            # Fallback: cálculo manual de modularidad
            from networkx.algorithms import community as nx_community
            
            # Convertir a formato de conjuntos
            comm_to_vars: Dict[int, Set[str]] = defaultdict(set)
            for var, comm_id in communities.items():
                comm_to_vars[comm_id].add(var)
            
            communities_sets = [frozenset(vars) for vars in comm_to_vars.values()]
            
            return nx_community.modularity(
                cg.graph.to_undirected(),
                communities_sets
            )


class BoundaryManager:
    """
    Gestor de restricciones de frontera entre clústeres.
    
    Identifica y gestiona las restricciones que cruzan fronteras entre
    clústeres, permitiendo la propagación eficiente de cambios.
    
    Examples:
        >>> manager = BoundaryManager()
        >>> boundaries = manager.get_boundary_constraints(gcd, cg)
        >>> for (c1, c2), constraints in boundaries.items():
        ...     print(f"Frontera {c1}-{c2}: {len(constraints)} restricciones")
    """
    
    def __init__(self):
        """Inicializa el gestor de fronteras."""
        pass
    
    def get_boundary_constraints(
        self,
        gcd: DynamicClusterGraph,
        cg: ConstraintGraph
    ) -> Dict[Tuple[int, int], List[Tuple[str, str]]]:
        """
        Obtiene todas las restricciones de frontera entre clústeres.
        
        Args:
            gcd: Grafo de clústeres dinámico
            cg: Grafo de restricciones
        
        Returns:
            Diccionario {(cluster1_id, cluster2_id): [(var1, var2), ...]}
        """
        boundaries: Dict[Tuple[int, int], List[Tuple[str, str]]] = defaultdict(list)
        
        # Crear mapa de variable a clúster
        var_to_cluster: Dict[str, int] = {}
        for cluster in gcd.get_all_clusters():
            for var in cluster.variables:
                var_to_cluster[var] = cluster.id
        
        # Identificar restricciones de frontera
        for var1 in cg.get_all_variables():
            cluster1 = var_to_cluster.get(var1)
            if cluster1 is None:
                continue
            
            for var2 in cg.get_neighbors(var1):
                cluster2 = var_to_cluster.get(var2)
                if cluster2 is None or cluster1 == cluster2:
                    continue
                
                # Restricción de frontera
                boundary_key = tuple(sorted([cluster1, cluster2]))
                boundaries[boundary_key].append((var1, var2))
        
        return boundaries
    
    def get_cluster_boundary_vars(
        self,
        cluster_id: int,
        gcd: DynamicClusterGraph,
        cg: ConstraintGraph
    ) -> Set[str]:
        """
        Obtiene las variables de frontera de un clúster.
        
        Variables de frontera son aquellas que tienen restricciones
        con variables de otros clústeres.
        
        Args:
            cluster_id: ID del clúster
            gcd: Grafo de clústeres dinámico
            cg: Grafo de restricciones
        
        Returns:
            Conjunto de variables de frontera
        """
        cluster = gcd.get_cluster(cluster_id)
        boundary_vars = set()
        
        # Crear mapa de variable a clúster
        var_to_cluster: Dict[str, int] = {}
        for c in gcd.get_all_clusters():
            for var in c.variables:
                var_to_cluster[var] = c.id
        
        # Identificar variables de frontera
        for var in cluster.variables:
            for neighbor in cg.get_neighbors(var):
                neighbor_cluster = var_to_cluster.get(neighbor)
                if neighbor_cluster is not None and neighbor_cluster != cluster_id:
                    boundary_vars.add(var)
                    break
        
        return boundary_vars
    
    def compute_boundary_density(
        self,
        cluster_id: int,
        gcd: DynamicClusterGraph,
        cg: ConstraintGraph
    ) -> float:
        """
        Calcula la densidad de frontera de un clúster.
        
        Densidad = (restricciones de frontera) / (restricciones totales)
        
        Args:
            cluster_id: ID del clúster
            gcd: Grafo de clústeres dinámico
            cg: Grafo de restricciones
        
        Returns:
            Densidad de frontera (0-1)
        """
        cluster = gcd.get_cluster(cluster_id)
        
        # Contar restricciones internas y de frontera
        internal_constraints = 0
        boundary_constraints = 0
        
        # Crear mapa de variable a clúster
        var_to_cluster: Dict[str, int] = {}
        for c in gcd.get_all_clusters():
            for var in c.variables:
                var_to_cluster[var] = c.id
        
        for var in cluster.variables:
            for neighbor in cg.get_neighbors(var):
                neighbor_cluster = var_to_cluster.get(neighbor)
                if neighbor_cluster == cluster_id:
                    internal_constraints += 1
                elif neighbor_cluster is not None:
                    boundary_constraints += 1
        
        total_constraints = internal_constraints + boundary_constraints
        
        if total_constraints == 0:
            return 0.0
        
        return boundary_constraints / total_constraints

