"""
Estructuras de Grafo para el Motor de Coherencia Adaptativa (ACE).

Este módulo implementa las dos representaciones del problema:
1. Grafo de Restricciones (GR) - Nivel 1
2. Grafo de Clústeres Dinámico (GCD) - Nivel 2

Autor: LatticeWeaver Team
Fecha: 12 de Octubre de 2025
Versión: 4.2.0
"""

from dataclasses import dataclass, field
from typing import Set, Dict, List, Tuple, Optional, Callable
import networkx as nx


@dataclass
class ConstraintEdge:
    """
    Arista del grafo de restricciones.
    
    Representa una restricción binaria entre dos variables.
    
    Attributes:
        var1: Primera variable
        var2: Segunda variable
        constraint_func: Función que verifica la restricción
        weight: Peso de la restricción (basado en selectividad)
    """
    var1: str
    var2: str
    constraint_func: Callable[[any, any], bool]
    weight: float = 1.0
    
    def evaluate(self, val1: any, val2: any) -> bool:
        """Evalúa la restricción con valores específicos."""
        return self.constraint_func(val1, val2)


class ConstraintGraph:
    """
    Grafo de Restricciones (GR) - Nivel 1.
    
    Representa el problema CSP como un grafo donde:
    - Nodos = Variables
    - Aristas = Restricciones binarias
    
    Esta es la representación base del problema, sin cambios
    respecto a la formulación original.
    
    Examples:
        >>> cg = ConstraintGraph()
        >>> cg.add_variable('X', {1, 2, 3})
        >>> cg.add_variable('Y', {1, 2, 3})
        >>> cg.add_constraint('X', 'Y', lambda x, y: x != y)
        >>> cg.get_domain('X')
        {1, 2, 3}
    """
    
    def __init__(self):
        """Inicializa un grafo de restricciones vacío."""
        self.graph = nx.Graph()
        self.constraints: Dict[Tuple[str, str], ConstraintEdge] = {}
        self._initial_domains: Dict[str, set] = {}
    
    def add_variable(self, var: str, domain: set):
        """
        Añade una variable al grafo.
        
        Args:
            var: Nombre de la variable
            domain: Conjunto de valores posibles
        
        Raises:
            ValueError: Si la variable ya existe
        """
        if var in self.graph:
            raise ValueError(f"Variable '{var}' already exists")
        
        self.graph.add_node(var, domain=set(domain))
        self._initial_domains[var] = set(domain)
    
    def add_constraint(
        self, 
        var1: str, 
        var2: str, 
        func: Callable[[any, any], bool],
        weight: float = 1.0
    ):
        """
        Añade una restricción (arista) al grafo.
        
        Args:
            var1: Primera variable
            var2: Segunda variable
            func: Función de restricción (var1_val, var2_val) -> bool
            weight: Peso de la restricción (opcional)
        
        Raises:
            ValueError: Si alguna variable no existe
        """
        if var1 not in self.graph or var2 not in self.graph:
            raise ValueError(f"Variables must exist before adding constraint")
        
        edge = ConstraintEdge(var1, var2, func, weight)
        self.graph.add_edge(var1, var2, constraint=edge)
        
        # Almacenar en ambas direcciones para acceso rápido
        self.constraints[(var1, var2)] = edge
        self.constraints[(var2, var1)] = edge
    
    def get_domain(self, var: str) -> set:
        """
        Obtiene el dominio actual de una variable.
        
        Args:
            var: Nombre de la variable
        
        Returns:
            Conjunto de valores posibles
        
        Raises:
            KeyError: Si la variable no existe
        """
        if var not in self.graph:
            raise KeyError(f"Variable '{var}' does not exist")
        
        return self.graph.nodes[var]['domain']
    
    def update_domain(self, var: str, new_domain: set):
        """
        Actualiza el dominio de una variable.
        
        Args:
            var: Nombre de la variable
            new_domain: Nuevo conjunto de valores
        
        Raises:
            KeyError: Si la variable no existe
            ValueError: Si el nuevo dominio está vacío
        """
        if var not in self.graph:
            raise KeyError(f"Variable '{var}' does not exist")
        
        if not new_domain:
            raise ValueError(f"Domain of '{var}' cannot be empty")
        
        self.graph.nodes[var]['domain'] = set(new_domain)
    
    def get_neighbors(self, var: str) -> List[str]:
        """
        Obtiene las variables vecinas (conectadas por restricciones).
        
        Args:
            var: Nombre de la variable
        
        Returns:
            Lista de nombres de variables vecinas
        """
        if var not in self.graph:
            return []
        
        return list(self.graph.neighbors(var))
    
    def get_constraint(self, var1: str, var2: str) -> Optional[ConstraintEdge]:
        """
        Obtiene la restricción entre dos variables.
        
        Args:
            var1: Primera variable
            var2: Segunda variable
        
        Returns:
            ConstraintEdge si existe, None en caso contrario
        """
        return self.constraints.get((var1, var2))
    
    def has_constraint(self, var1: str, var2: str) -> bool:
        """Verifica si existe una restricción entre dos variables."""
        return (var1, var2) in self.constraints
    
    def get_all_variables(self) -> List[str]:
        """Obtiene todas las variables del grafo."""
        return list(self.graph.nodes())
    
    def get_all_constraints(self) -> List[ConstraintEdge]:
        """Obtiene todas las restricciones del grafo."""
        return [edge for _, _, edge in self.graph.edges(data='constraint')]
    
    def reset_domains(self):
        """Restaura todos los dominios a sus valores iniciales."""
        for var, initial_domain in self._initial_domains.items():
            self.graph.nodes[var]['domain'] = set(initial_domain)
    
    def is_consistent(self) -> bool:
        """
        Verifica si el grafo está en un estado consistente.
        
        Returns:
            True si todos los dominios son no vacíos
        """
        return all(
            len(self.get_domain(var)) > 0 
            for var in self.get_all_variables()
        )
    
    def __repr__(self) -> str:
        num_vars = len(self.graph.nodes())
        num_constraints = len(self.graph.edges())
        return f"ConstraintGraph(vars={num_vars}, constraints={num_constraints})"


@dataclass
class Cluster:
    """
    Un clúster de variables fuertemente acopladas.
    
    Representa un subconjunto de variables que se resuelven juntas
    como una unidad coherente.
    
    Attributes:
        id: Identificador único del clúster
        variables: Conjunto de variables en el clúster
        state: Estado actual (ACTIVE, STABLE, SOLVED, INCONSISTENT)
        last_update_iteration: Última iteración en que cambió
        metadata: Metadatos adicionales
    """
    id: int
    variables: Set[str]
    state: str = "ACTIVE"
    last_update_iteration: int = 0
    metadata: Dict = field(default_factory=dict)
    
    def is_active(self) -> bool:
        """Verifica si el clúster está activo."""
        return self.state == "ACTIVE"
    
    def is_stable(self) -> bool:
        """Verifica si el clúster está estable."""
        return self.state == "STABLE"
    
    def is_solved(self) -> bool:
        """Verifica si el clúster está resuelto."""
        return self.state == "SOLVED"
    
    def is_inconsistent(self) -> bool:
        """Verifica si el clúster es inconsistente."""
        return self.state == "INCONSISTENT"
    
    def mark_active(self, iteration: int):
        """Marca el clúster como activo."""
        self.state = "ACTIVE"
        self.last_update_iteration = iteration
    
    def mark_stable(self):
        """Marca el clúster como estable."""
        self.state = "STABLE"
    
    def mark_solved(self):
        """Marca el clúster como resuelto."""
        self.state = "SOLVED"
    
    def mark_inconsistent(self):
        """Marca el clúster como inconsistente."""
        self.state = "INCONSISTENT"
    
    def __repr__(self) -> str:
        return f"Cluster(id={self.id}, vars={len(self.variables)}, state={self.state})"


class DynamicClusterGraph:
    """
    Grafo de Clústeres Dinámico (GCD) - Nivel 2.
    
    Representa una vista de alto nivel del problema donde:
    - Nodos = Clústeres de variables fuertemente acopladas
    - Aristas = Restricciones de frontera entre clústeres
    
    Este grafo evoluciona durante la ejecución mediante operaciones
    de renormalización (MERGE, SPLIT, PRUNE).
    
    Examples:
        >>> gcd = DynamicClusterGraph()
        >>> c1 = gcd.add_cluster({'X', 'Y'})
        >>> c2 = gcd.add_cluster({'Z', 'W'})
        >>> gcd.add_boundary_constraint(c1, c2)
        >>> gcd.get_active_clusters()
        [Cluster(id=0, ...), Cluster(id=1, ...)]
    """
    
    def __init__(self):
        """Inicializa un grafo de clústeres vacío."""
        self.graph = nx.Graph()
        self.clusters: Dict[int, Cluster] = {}
        self.next_cluster_id = 0
    
    def add_cluster(self, variables: Set[str], state: str = "ACTIVE") -> int:
        """
        Crea un nuevo clúster.
        
        Args:
            variables: Conjunto de variables del clúster
            state: Estado inicial del clúster
        
        Returns:
            ID del clúster creado
        
        Raises:
            ValueError: Si el conjunto de variables está vacío
        """
        if not variables:
            raise ValueError("Cluster cannot be empty")
        
        cluster_id = self.next_cluster_id
        self.next_cluster_id += 1
        
        cluster = Cluster(id=cluster_id, variables=set(variables), state=state)
        self.clusters[cluster_id] = cluster
        self.graph.add_node(cluster_id, cluster=cluster)
        
        return cluster_id
    
    def add_boundary_constraint(self, cluster1_id: int, cluster2_id: int):
        """
        Añade una arista de frontera entre clústeres.
        
        Args:
            cluster1_id: ID del primer clúster
            cluster2_id: ID del segundo clúster
        
        Raises:
            KeyError: Si algún clúster no existe
        """
        if cluster1_id not in self.clusters or cluster2_id not in self.clusters:
            raise KeyError("Both clusters must exist")
        
        self.graph.add_edge(cluster1_id, cluster2_id)
    
    def get_cluster(self, cluster_id: int) -> Cluster:
        """
        Obtiene un clúster por su ID.
        
        Args:
            cluster_id: ID del clúster
        
        Returns:
            Objeto Cluster
        
        Raises:
            KeyError: Si el clúster no existe
        """
        if cluster_id not in self.clusters:
            raise KeyError(f"Cluster {cluster_id} does not exist")
        
        return self.clusters[cluster_id]
    
    def get_active_clusters(self) -> List[Cluster]:
        """
        Retorna los clústeres activos.
        
        Returns:
            Lista de clústeres en estado ACTIVE
        """
        return [c for c in self.clusters.values() if c.is_active()]
    
    def get_all_clusters(self) -> List[Cluster]:
        """Retorna todos los clústeres."""
        return list(self.clusters.values())
    
    def get_neighbors(self, cluster_id: int) -> List[int]:
        """
        Obtiene los clústeres vecinos.
        
        Args:
            cluster_id: ID del clúster
        
        Returns:
            Lista de IDs de clústeres vecinos
        """
        if cluster_id not in self.graph:
            return []
        
        return list(self.graph.neighbors(cluster_id))
    
    def merge_clusters(self, cluster1_id: int, cluster2_id: int) -> int:
        """
        Fusiona dos clústeres en uno nuevo.
        
        Operación de RENORMALIZACIÓN: MERGE
        
        Args:
            cluster1_id: ID del primer clúster
            cluster2_id: ID del segundo clúster
        
        Returns:
            ID del nuevo meta-clúster
        
        Raises:
            KeyError: Si algún clúster no existe
        """
        c1 = self.get_cluster(cluster1_id)
        c2 = self.get_cluster(cluster2_id)
        
        # Crear nuevo meta-clúster
        merged_vars = c1.variables | c2.variables
        new_id = self.add_cluster(merged_vars, state="ACTIVE")
        
        # Transferir aristas de frontera
        neighbors1 = set(self.get_neighbors(cluster1_id))
        neighbors2 = set(self.get_neighbors(cluster2_id))
        
        # Unir vecinos (excluyendo los clústeres que se están fusionando)
        all_neighbors = (neighbors1 | neighbors2) - {cluster1_id, cluster2_id}
        
        for neighbor in all_neighbors:
            self.add_boundary_constraint(new_id, neighbor)
        
        # Eliminar clústeres antiguos
        self.remove_cluster(cluster1_id)
        self.remove_cluster(cluster2_id)
        
        return new_id
    
    def split_cluster(
        self, 
        cluster_id: int, 
        partition1: Set[str], 
        partition2: Set[str]
    ) -> Tuple[int, int]:
        """
        Divide un clúster en dos.
        
        Operación de RENORMALIZACIÓN: SPLIT
        
        Args:
            cluster_id: ID del clúster a dividir
            partition1: Primer conjunto de variables
            partition2: Segundo conjunto de variables
        
        Returns:
            Tupla con los IDs de los dos nuevos clústeres
        
        Raises:
            KeyError: Si el clúster no existe
            ValueError: Si las particiones no son válidas
        """
        cluster = self.get_cluster(cluster_id)
        
        # Validar particiones
        if partition1 | partition2 != cluster.variables:
            raise ValueError("Partitions must cover all variables")
        
        if partition1 & partition2:
            raise ValueError("Partitions must be disjoint")
        
        # Crear nuevos clústeres
        new_id1 = self.add_cluster(partition1)
        new_id2 = self.add_cluster(partition2)
        
        # Añadir arista entre los nuevos clústeres
        self.add_boundary_constraint(new_id1, new_id2)
        
        # Transferir aristas de frontera
        for neighbor in self.get_neighbors(cluster_id):
            self.add_boundary_constraint(new_id1, neighbor)
            self.add_boundary_constraint(new_id2, neighbor)
        
        # Eliminar clúster antiguo
        self.remove_cluster(cluster_id)
        
        return new_id1, new_id2
    
    def prune_cluster(self, cluster_id: int):
        """
        Elimina un clúster resuelto o inconsistente.
        
        Operación de RENORMALIZACIÓN: PRUNE
        
        Args:
            cluster_id: ID del clúster a eliminar
        
        Raises:
            KeyError: Si el clúster no existe
        """
        self.remove_cluster(cluster_id)
    
    def remove_cluster(self, cluster_id: int):
        """
        Elimina un clúster del grafo.
        
        Args:
            cluster_id: ID del clúster a eliminar
        
        Raises:
            KeyError: Si el clúster no existe
        """
        if cluster_id not in self.clusters:
            raise KeyError(f"Cluster {cluster_id} does not exist")
        
        self.graph.remove_node(cluster_id)
        del self.clusters[cluster_id]
    
    def is_empty(self) -> bool:
        """Verifica si el grafo está vacío."""
        return len(self.clusters) == 0
    
    def __repr__(self) -> str:
        num_clusters = len(self.clusters)
        num_boundaries = len(self.graph.edges())
        return f"DynamicClusterGraph(clusters={num_clusters}, boundaries={num_boundaries})"

