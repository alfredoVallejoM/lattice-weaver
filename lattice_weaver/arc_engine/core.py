# lattice_weaver/arc_engine/core.py

from typing import Iterable, Callable, Any, Optional, Dict, Tuple, Set, List
import networkx as nx

from .domains import create_optimal_domain, Domain
from .constraints import Constraint
from .ac31 import revise_with_last_support

class ArcEngine:
    """
    @i18n:key arc_engine_class
    @i18n:category core_engine
    
    @i18n:desc_es Motor de consistencia de arcos de alto rendimiento y optimizado, basado en AC-3.1. Esta es la Capa 0 de la arquitectura LatticeWeaver.
    @i18n:desc_en High-performance, optimized arc consistency engine based on AC-3.1. This is Layer 0 of the LatticeWeaver architecture.
    @i18n:desc_fr Moteur de cohérence d'arc optimisé et à haute performance, basé sur AC-3.1. C'est la couche 0 de l'architecture LatticeWeaver.
    """

    def __init__(self, parallel: bool = False, parallel_mode: str = 'thread', use_tms: bool = False):
        """
        @i18n:key arc_engine_init
        @i18n:desc_es Inicializa el ArcEngine.
        @i18n:desc_en Initializes the ArcEngine.
        @i18n:desc_fr Initialise le ArcEngine.

        Parameters
        ----------
        parallel : bool, default=False
            @i18n:param parallel
            @i18n:type bool
            @i18n:desc_es Si es True, habilita la ejecución paralela.
            @i18n:desc_en If True, enables parallel execution.
            @i18n:desc_fr Si True, active l'exécution parallèle.
        parallel_mode : str, default='thread'
            @i18n:param parallel_mode
            @i18n:type str
            @i18n:values ["thread", "topological"]
            @i18n:desc_es Tipo de paralelización ('thread', 'topological'). 'thread' usa ThreadPoolExecutor. 'topological' usa multiprocessing con grupos independientes.
            @i18n:desc_en Type of parallelization ('thread', 'topological'). 'thread' uses ThreadPoolExecutor. 'topological' uses multiprocessing with independent groups.
            @i18n:desc_fr Type de parallélisation ('thread', 'topological'). 'thread' utilise ThreadPoolExecutor. 'topological' utilise multiprocessing avec des groupes indépendants.
        use_tms : bool, default=False
            @i18n:param use_tms
            @i18n:type bool
            @i18n:desc_es Si es True, habilita el Truth Maintenance System para seguimiento de dependencias.
            @i18n:desc_en If True, enables the Truth Maintenance System for dependency tracking.
            @i18n:desc_fr Si True, active le Truth Maintenance System pour le suivi des dépendances.
        """
        self.variables: Dict[str, Domain] = {}
        self.constraints: Dict[str, Constraint] = {}
        self.graph = nx.Graph()  # Constraint graph
        self.parallel = parallel
        self.parallel_mode = parallel_mode
        self.use_tms = use_tms

        # Data structure for AC-3.1 last support optimization
        self.last_support: Dict[Tuple[str, str, Any], Any] = {}
        
        # Truth Maintenance System (optional)
        self.tms = None
        if use_tms:
            from .tms import create_tms
            self.tms = create_tms()

    def add_variable(self, name: str, domain: Iterable[Any]):
        """
        @i18n:key add_variable
        @i18n:desc_es Añade una variable con su dominio inicial. El motor seleccionará automáticamente la estructura de datos más eficiente para representar el dominio.
        @i18n:desc_en Adds a variable with its initial domain. The engine will automatically select the most efficient data structure to represent the domain.
        @i18n:desc_fr Ajoute une variable avec son domaine initial. Le moteur sélectionnera automatiquement la structure de données la plus efficace pour représenter le domaine.

        Parameters
        ----------
        name : str
            @i18n:param name
            @i18n:type str
            @i18n:desc_es El nombre de la variable.
            @i18n:desc_en The name of the variable.
            @i18n:desc_fr Le nom de la variable.
        domain : Iterable[Any]
            @i18n:param domain
            @i18n:type Iterable[Any]
            @i18n:desc_es Un iterable de valores posibles para la variable.
            @i18n:desc_en An iterable of possible values for the variable.
            @i18n:desc_fr Un itérable de valeurs possibles pour la variable.
        """
        if name in self.variables:
            raise ValueError(f"Variable '{name}' already exists.")
        self.variables[name] = create_optimal_domain(domain)
        self.graph.add_node(name)

    def add_constraint(self, var1: str, var2: str, relation: Callable[[Any, Any], bool], cid: Optional[str] = None):
        """
        @i18n:key add_constraint
        @i18n:desc_es Añade una restricción binaria entre dos variables.
        @i18n:desc_en Adds a binary constraint between two variables.
        @i18n:desc_fr Ajoute une contrainte binaire entre deux variables.

        Parameters
        ----------
        var1 : str
            @i18n:param var1
            @i18n:type str
            @i18n:desc_es Nombre de la primera variable.
            @i18n:desc_en Name of the first variable.
            @i18n:desc_fr Nom de la première variable.
        var2 : str
            @i18n:param var2
            @i18n:type str
            @i18n:desc_es Nombre de la segunda variable.
            @i18n:desc_en Name of the second variable.
            @i18n:desc_fr Nom de la deuxième variable.
        relation : Callable[[Any, Any], bool]
            @i18n:param relation
            @i18n:type Callable[[Any, Any], bool]
            @i18n:desc_es Una función que devuelve True si dos valores son consistentes.
            @i18n:desc_en A callable that returns True if two values are consistent.
            @i18n:desc_fr Une fonction qui renvoie True si deux valeurs sont cohérentes.
        cid : Optional[str], default=None
            @i18n:param cid
            @i18n:type Optional[str]
            @i18n:desc_es ID opcional para la restricción.
            @i18n:desc_en Optional ID for the constraint.
            @i18n:desc_fr ID optionnel pour la contrainte.
        """
        if cid is None:
            cid = f"{var1}_{var2}"
        if cid in self.constraints:
            raise ValueError(f"Constraint ID '{cid}' already exists.")
        
        self.constraints[cid] = Constraint(var1, var2, relation)
        self.graph.add_edge(var1, var2, cid=cid)

    def enforce_arc_consistency(self) -> bool:
        """
        @i18n:key enforce_arc_consistency
        @i18n:desc_es Aplica consistencia de arcos en todo el CSP usando un algoritmo AC-3.1 optimizado. Si el modo paralelo está habilitado, usa la estrategia de paralelización especificada.
        @i18n:desc_en Enforces arc consistency on the entire CSP using an optimized AC-3.1 algorithm. If parallel mode is enabled, uses the specified parallelization strategy.
        @i18n:desc_fr Applique la cohérence d'arc sur l'ensemble du CSP en utilisant un algorithme AC-3.1 optimisé. Si le mode parallèle est activé, utilise la stratégie de parallélisation spécifiée.

        Returns
        -------
        bool
            @i18n:return bool
            @i18n:desc_es False si se encuentra una inconsistencia (un dominio se vacía), True en caso contrario.
            @i18n:desc_en False if an inconsistency is found (a domain becomes empty), True otherwise.
            @i18n:desc_fr False si une incohérence est trouvée (un domaine devient vide), True sinon.
        """
        if self.parallel:
            if self.parallel_mode == 'topological':
                from .topological_parallel import TopologicalParallelAC3
                topological_ac3 = TopologicalParallelAC3(self)
                return topological_ac3.enforce_arc_consistency_topological()
            else:  # 'thread' mode
                from .parallel_ac3 import ParallelAC3
                parallel_ac3 = ParallelAC3(self)
                return parallel_ac3.enforce_arc_consistency_parallel()
        
        queue: list[tuple[str, str, str]] = []
        for cid, c in self.constraints.items():
            queue.append((c.var1, c.var2, cid))
            queue.append((c.var2, c.var1, cid))

        while queue:
            xi, xj, constraint_id = queue.pop(0)

            revised, removed_values = revise_with_last_support(self, xi, xj, constraint_id)

            if revised:
                if self.use_tms and self.tms and removed_values:
                    for removed_val in removed_values:
                        self.tms.record_removal(
                            variable=xi,
                            value=removed_val,
                            constraint_id=constraint_id,
                            supporting_values={xj: list(self.variables[xj].get_values())}
                        )
                
                if not self.variables[xi]:
                    if self.use_tms and self.tms:
                        explanations = self.tms.explain_inconsistency(xi)
                        suggested = self.tms.suggest_constraint_to_relax(xi)
                        if suggested:
                            print(f"⚠️ Sugerencia: relajar restricción '{suggested}'")
                    
                    return False

                for neighbor in self.graph.neighbors(xi):
                    if neighbor != xj:
                        c_id = self.graph.get_edge_data(neighbor, xi)['cid']
                        queue.append((neighbor, xi, c_id))
        
        return True

    def build_consistency_graph(self) -> nx.Graph:
        """
        @i18n:key build_consistency_graph
        @i18n:note not_implemented
        @i18n:desc_es Construye el grafo de consistencia (o micro-estructura) del CSP. Nodos son pares (variable, valor), aristas conectan asignaciones consistentes. (Será implementado en Fase 3).
        @i18n:desc_en Builds the consistency graph (or micro-structure) of the CSP. Nodes are (variable, value) pairs, edges connect consistent assignments. (To be implemented in Phase 3).
        @i18n:desc_fr Construit le graphe de cohérence (ou micro-structure) du CSP. Les nœuds sont des paires (variable, valeur), les arêtes relient des affectations cohérentes. (Sera implémenté en Phase 3).
        """
        raise NotImplementedError("build_consistency_graph will be implemented in Phase 3.")

    def analyze_simplicial_topology(self, concept_lattice: Optional[Any] = None) -> Dict[str, int]:
        """
        @i18n:key analyze_simplicial_topology
        @i18n:note not_implemented
        @i18n:desc_es Realiza análisis topológico en el grafo de consistencia calculando los números de Betti de su complejo de cliques. (Será implementado en Fase 3).
        @i18n:desc_en Performs topological analysis on the consistency graph by computing the Betti numbers of its clique complex. (To be implemented in Phase 3).
        @i18n:desc_fr Effectue une analyse topologique sur le graphe de cohérence en calculant les nombres de Betti de son complexe de cliques. (Sera implémenté en Phase 3).
        """
        raise NotImplementedError("analyze_simplicial_topology will be implemented in Phase 3.")

    def remove_constraint(self, constraint_id: str):
        """
        @i18n:key remove_constraint
        @i18n:desc_es Elimina una restricción y restaura eficientemente la consistencia usando TMS. Si TMS está habilitado, los valores eliminados debido a esta restricción se restauran si son consistentes con las restricciones restantes.
        @i18n:desc_en Removes a constraint and efficiently restores consistency using TMS. If TMS is enabled, values removed due to this constraint are restored if they are consistent with remaining constraints.
        @i18n:desc_fr Supprime une contrainte et restaure efficacement la cohérence en utilisant TMS. Si TMS est activé, les valeurs supprimées à cause de cette contrainte sont restaurées si elles sont cohérentes avec les contraintes restantes.

        Parameters
        ----------
        constraint_id : str
            @i18n:param constraint_id
            @i18n:type str
            @i18n:desc_es ID de la restricción a eliminar.
            @i18n:desc_en ID of the constraint to remove.
            @i18n:desc_fr ID de la contrainte à supprimer.
        """
        if constraint_id not in self.constraints:
            raise ValueError(f"Constraint '{constraint_id}' not found")
        
        constraint = self.constraints[constraint_id]
        var1, var2 = constraint.var1, constraint.var2
        
        del self.constraints[constraint_id]
        self.graph.remove_edge(var1, var2)
        
        if self.use_tms and self.tms:
            restorable = self.tms.get_restorable_values(constraint_id)
            
            for var, values in restorable.items():
                for val in values:
                    if self._is_value_consistent(var, val):
                        self.variables[var].add(val)
                        print(f"✅ Restaurado: {var}={val}")
            
            self.tms.remove_constraint_justifications(constraint_id)
        
        print(f"Restricción '{constraint_id}' eliminada")
    
    def _is_value_consistent(self, variable: str, value: Any) -> bool:
        """
        @i1e:key _is_value_consistent
        @i18n:desc_es Verifica si un valor es consistente con todas las restricciones actuales.
        @i18n:desc_en Checks if a value is consistent with all current constraints.
        @i18n:desc_fr Vérifie si une valeur est cohérente avec toutes les contraintes actuelles.
        """
        for neighbor in self.graph.neighbors(variable):
            cid = self.graph.get_edge_data(variable, neighbor)['cid']
            constraint = self.constraints[cid]
            
            has_support = False
            for neighbor_val in self.variables[neighbor].get_values():
                if constraint.var1 == variable:
                    if constraint.relation(value, neighbor_val):
                        has_support = True
                        break
                else:
                    if constraint.relation(neighbor_val, value):
                        has_support = True
                        break
            
            if not has_support:
                return False
        
        return True

    def __repr__(self):
        return f"ArcEngine(variables={len(self.variables)}, constraints={len(self.constraints)})"

