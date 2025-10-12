# lattice_weaver/arc_engine/core.py

from typing import Iterable, Callable, Any, Optional, Dict, Tuple, Set, List
import networkx as nx

from .domains import create_optimal_domain, Domain
from .constraints import Constraint
from .ac31 import revise_with_last_support

class ArcEngine:
    """
    High-performance, optimized arc consistency engine based on AC-3.1.
    This is Layer 0 of the LatticeWeaver architecture.
    """

    def __init__(self, parallel: bool = False, parallel_mode: str = 'thread', use_tms: bool = False):
        """
        Initializes the ArcEngine.

        :param parallel: If True, enables parallel execution.
        :param parallel_mode: Type of parallelization ('thread', 'topological').
                             'thread' uses ThreadPoolExecutor (limited by GIL).
                             'topological' uses multiprocessing with independent groups.
        :param use_tms: If True, enables Truth Maintenance System for dependency tracking.
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
        Adds a variable with its initial domain.

        The engine will automatically select the most efficient data structure
        to represent the domain.

        :param name: The name of the variable.
        :param domain: An iterable of possible values for the variable.
        """
        if name in self.variables:
            raise ValueError(f"Variable '{name}' already exists.")
        self.variables[name] = create_optimal_domain(domain)
        self.graph.add_node(name)

    def add_constraint(self, var1: str, var2: str, relation: Callable[[Any, Any], bool], cid: Optional[str] = None):
        """
        Adds a binary constraint between two variables.

        :param var1: Name of the first variable.
        :param var2: Name of the second variable.
        :param relation: A callable that returns True if two values are consistent.
        :param cid: Optional ID for the constraint.
        """
        if cid is None:
            cid = f"{var1}_{var2}"
        if cid in self.constraints:
            raise ValueError(f"Constraint ID '{cid}' already exists.")
        
        self.constraints[cid] = Constraint(var1, var2, relation)
        self.graph.add_edge(var1, var2, cid=cid)

    def enforce_arc_consistency(self) -> bool:
        """
        Enforces arc consistency on the entire CSP using an optimized AC-3.1 algorithm.
        
        If parallel mode is enabled, uses the specified parallelization strategy.

        :return: False if an inconsistency is found (a domain becomes empty), True otherwise.
        """
        # Use parallel version if enabled
        if self.parallel:
            if self.parallel_mode == 'topological':
                from .topological_parallel import TopologicalParallelAC3
                topological_ac3 = TopologicalParallelAC3(self)
                return topological_ac3.enforce_arc_consistency_topological()
            else:  # 'thread' mode
                from .parallel_ac3 import ParallelAC3
                parallel_ac3 = ParallelAC3(self)
                return parallel_ac3.enforce_arc_consistency_parallel()
        
        # Sequential AC-3.1 algorithm
        # The queue contains tuples of (variable_to_revise, constraining_variable, constraint_id)
        queue: list[tuple[str, str, str]] = []
        for cid, c in self.constraints.items():
            queue.append((c.var1, c.var2, cid))
            queue.append((c.var2, c.var1, cid))

        while queue:
            xi, xj, constraint_id = queue.pop(0)

            # The core of the AC-3.1 algorithm
            revised, removed_values = revise_with_last_support(self, xi, xj, constraint_id)

            if revised:
                # Register removals in TMS if enabled
                if self.use_tms and self.tms and removed_values:
                    for removed_val in removed_values:
                        self.tms.record_removal(
                            variable=xi,
                            value=removed_val,
                            constraint_id=constraint_id,
                            supporting_values={xj: list(self.variables[xj].get_values())}
                        )
                
                if not self.variables[xi]:
                    # Inconsistency detected
                    if self.use_tms and self.tms:
                        # Explain inconsistency
                        explanations = self.tms.explain_inconsistency(xi)
                        suggested = self.tms.suggest_constraint_to_relax(xi)
                        if suggested:
                            print(f"⚠️ Sugerencia: relajar restricción '{suggested}'")
                    
                    return False  # Inconsistency found

                # Add affected arcs back to the queue
                for neighbor in self.graph.neighbors(xi):
                    if neighbor != xj:
                        # Find the constraint ID for the (neighbor, xi) arc
                        c_id = self.graph.get_edge_data(neighbor, xi)['cid']
                        queue.append((neighbor, xi, c_id))
        
        return True

    def build_consistency_graph(self) -> nx.Graph:
        """
        Builds the consistency graph (or micro-structure) of the CSP.
        Nodes are (variable, value) pairs, edges connect consistent assignments.
        
        (To be implemented in Phase 3)
        """
        # Placeholder for Phase 3
        raise NotImplementedError("build_consistency_graph will be implemented in Phase 3.")

    def analyze_simplicial_topology(self, concept_lattice: Optional[Any] = None) -> Dict[str, int]:
        """
        Performs topological analysis on the consistency graph by computing
        the Betti numbers of its clique complex.

        (To be implemented in Phase 3)

        :param concept_lattice: (Optional) A pre-computed concept lattice from Layer 1
                                  to accelerate clique finding.
        :return: A dictionary of Betti numbers (e.g., {'b0': 1, 'b1': 3}).
        """
        # Placeholder for Phase 3
        raise NotImplementedError("analyze_simplicial_topology will be implemented in Phase 3.")

    def remove_constraint(self, constraint_id: str):
        """
        Removes a constraint and efficiently restores consistency using TMS.
        
        If TMS is enabled, values removed due to this constraint are restored
        if they are consistent with remaining constraints.
        
        :param constraint_id: ID of the constraint to remove
        """
        if constraint_id not in self.constraints:
            raise ValueError(f"Constraint '{constraint_id}' not found")
        
        # Get constraint info before removal
        constraint = self.constraints[constraint_id]
        var1, var2 = constraint.var1, constraint.var2
        
        # Remove constraint
        del self.constraints[constraint_id]
        self.graph.remove_edge(var1, var2)
        
        # If TMS enabled, restore values
        if self.use_tms and self.tms:
            restorable = self.tms.get_restorable_values(constraint_id)
            
            for var, values in restorable.items():
                for val in values:
                    # Check if value is consistent with remaining constraints
                    if self._is_value_consistent(var, val):
                        self.variables[var].add(val)
                        print(f"✅ Restaurado: {var}={val}")
            
            # Clean up TMS
            self.tms.remove_constraint_justifications(constraint_id)
        
        print(f"Restricción '{constraint_id}' eliminada")
    
    def _is_value_consistent(self, variable: str, value: Any) -> bool:
        """
        Verifica si un valor es consistente con todas las restricciones actuales.
        
        :param variable: Variable
        :param value: Valor a verificar
        :return: True si el valor es consistente
        """
        for neighbor in self.graph.neighbors(variable):
            cid = self.graph.get_edge_data(variable, neighbor)['cid']
            constraint = self.constraints[cid]
            
            # Check if there's support
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

