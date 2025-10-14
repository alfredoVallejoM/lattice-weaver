# lattice_weaver/homotopy/analyzer.py

"""
Homotopy Analyzer - Layer 2 of LatticeWeaver

This module implements the HomotopyAnalyzer class, which detects and analyzes
homotopic equivalences in the space of specifications. It integrates with the
ArcEngine (Layer 0) to provide a high-level view of the problem's structure.

This is an evolution of the original Locale class from LocalePy, refactored to
work with the new optimized ArcEngine.
"""

import networkx as nx
from typing import FrozenSet, Any, Tuple, List, Optional, Dict, Set
from collections import defaultdict
import json

from lattice_weaver.core.csp_problem import CSP, Constraint
from .rules import HomotopyRules  # 🆕 AÑADIR

# Type aliases
Constraint = Tuple[Any, ...]
State = FrozenSet[Constraint]


class HomotopyAnalyzer:
    """
    Analyzes the homotopy type of a problem's specification space.
    
    This class maintains a graph of states (coherent sets of constraints) and
    detects when two different paths of specification lead to equivalent states,
    which indicates a homotopic equivalence (commutativity of constraints).
    
    Attributes:
        graph (nx.DiGraph): Graph of states and specification transitions
        TOP (State): State of total indetermination (top element)
        BOTTOM (State): Contradictory state (bottom element)
        arc_engine (ArcEngine): The underlying coherence engine
        homotopy_counter (int): Number of homotopies detected
    """
    
    def __init__(self, csp_instance: Optional[CSP] = None):
        """
        Initializes the HomotopyAnalyzer.
        
        Args:
            arc_engine: An optional ArcEngine instance. If None, a new one is created.
        """
        # Layer 0: Arc consistency engine
        self.csp_instance = csp_instance if csp_instance is not None else CSP()
        
        # 🆕 Layer 2: Homotopy rules (precomputed)
        self.rules = HomotopyRules()
        if len(self.csp_instance.constraints) > 0:
            self.rules.precompute_from_constraints(self.csp_instance.constraints)
        
        # Layer 2: Graph of states and specifications
        self.graph = nx.DiGraph()
        
        # Special elements of the Heyting algebra
        self.TOP: State = frozenset()
        self.BOTTOM: State = frozenset([('__CONTRADICTION__',)])
        
        # Initialize graph
        self.graph.add_node(self.TOP, type='state', label='TOP', level=0)
        self.graph.add_node(self.BOTTOM, type='state', label='BOTTOM', level=-1)
        
        # Statistics
        self.homotopy_counter = 0
        self.state_counter = 2
        self.specification_counter = 0
        
        # Cache for optimization
        self._meet_cache: Dict[Tuple[State, State], State] = {}
        
        # Homotopy registry: stores detected commutative squares
        self.homotopies: List[Dict[str, State]] = []
    
    def _meet(self, state_a: State, state_b: State) -> State:
        """
        Computes the meet (conjunction) of two states in the Heyting algebra.
        
        The meet is the union of constraints if coherent, or BOTTOM if not.
        
        Args:
            state_a: First state
            state_b: Second state
        
        Returns:
            The meet of both states (state_a ∧ state_b)
        """
        # Check cache
        cache_key = (state_a, state_b) if state_a <= state_b else (state_b, state_a)
        if cache_key in self._meet_cache:
            return self._meet_cache[cache_key]
        
        # Handle special cases
        if state_a == self.BOTTOM or state_b == self.BOTTOM:
            result = self.BOTTOM
        elif state_a == self.TOP:
            result = state_b
        elif state_b == self.TOP:
            result = state_a
        else:
            # Compute union
            new_state_candidate = state_a.union(state_b)
            
            # Check coherence using the ArcEngine
            is_coherent = self._check_coherence(new_state_candidate)
            
            result = new_state_candidate if is_coherent else self.BOTTOM
        
        # Cache result
        self._meet_cache[cache_key] = result
        return result
    
    def _check_coherence(self, state: State) -> bool:
        """
        Checks if a state is coherent using the ArcEngine.
        
        This method translates the state (set of constraints) into the ArcEngine's
        format and runs arc consistency.
        
        Args:
            state: The state to check
        
        Returns:
            True if coherent, False otherwise
        """
        if state == self.TOP:
            return True
        if state == self.BOTTOM:
            return False
        
        # Create a temporary ArcEngine for this check
        temp_engine = CSP()
        
        # Parse constraints and populate the engine
        variables = {}
        constraints_list = []
        
        for constraint in state:
            if constraint[0] == '__CONTRADICTION__':
                return False
            elif constraint[0] == 'var':
                # Format: ('var', 'A', 'in', {1, 2, 3})
                _, var_name, _, domain = constraint
                variables[var_name] = domain
            elif constraint[0] == 'constraint':
                # Format: ('constraint', 'A', 'B', 'neq')
                constraints_list.append(constraint)
        
        # Add variables to engine
        for var_name, domain in variables.items():
            temp_engine.variables[var_name] = domain
        
        # Add constraints
        constraint_counter = 0
        for constraint in constraints_list:
            _, var1, var2, rel_type = constraint
            cid = f"c_{constraint_counter}_{var1}_{var2}_{rel_type}"
            constraint_counter += 1
            
            if rel_type == 'neq':
                temp_engine.constraints.append(Constraint(scope=[var1, var2], relation=lambda a, b: a != b, name=cid))
            elif rel_type == 'eq':
                temp_engine.constraints.append(Constraint(scope=[var1, var2], relation=lambda a, b: a == b, name=cid))
            elif rel_type == 'lt':
                temp_engine.constraints.append(Constraint(scope=[var1, var2], relation=lambda a, b: a < b, name=cid))
            elif rel_type == 'gt':
                temp_engine.constraints.append(Constraint(scope=[var1, var2], relation=lambda a, b: a > b, name=cid))
            # Add more relation types as needed
        
        # Run arc consistency
        from ..core.csp_engine.solver import AC3Solver
        solver = AC3Solver(temp_engine)
        return solver.enforce_arc_consistency()
    
    def specify(self, current_state: State, constraint: Constraint) -> State:
        """
        Applies a specification (adds a constraint) to the current state.
        
        This is the main operation for evolving the problem. It computes the meet
        of the current state with the new constraint, updates the graph, and
        triggers the cascade to detect homotopies.
        
        Args:
            current_state: The current state
            constraint: The constraint to add
        
        Returns:
            The new state after specification
        """
        # Create constraint state
        constraint_state = frozenset([constraint])
        
        # Compute new state
        new_state = self._meet(current_state, constraint_state)
        
        # Update graph
        if new_state not in self.graph:
            level = self.graph.nodes[current_state].get('level', 0) + 1
            self.graph.add_node(new_state, type='state', level=level)
            self.state_counter += 1
        
        # Add edge
        self.graph.add_edge(current_state, new_state, constraint=constraint)
        self.specification_counter += 1
        
        # Trigger homotopy detection cascade
        if new_state != self.BOTTOM:
            self._cascade_recalculation(current_state, constraint, new_state)
        
        return new_state
    
    def _cascade_recalculation(self, base_state: State, new_constraint: Constraint, new_state: State):
        """
        Detects homotopies by checking if the new specification commutes with
        existing specifications from the base state.
        
        🆕 OPTIMIZADO: Usa reglas precomputadas para verificación O(1) en lugar de O(k²).
        
        A homotopy is detected when:
        (base_state ∧ constraint_A) ∧ constraint_B == (base_state ∧ constraint_B) ∧ constraint_A
        
        This forms a commutative square in the specification graph.
        
        Args:
            base_state: The state from which the new specification was made
            new_constraint: The constraint that was just added
            new_state: The resulting state (base_state ∧ new_constraint)
        """
        # Extraer el ID de la restricción del constraint
        new_constraint_id = self._extract_constraint_id(new_constraint)
        
        # Get all outgoing edges from base_state (except the one we just added)
        for _, target_state, edge_data in self.graph.out_edges(base_state, data=True):
            if target_state == new_state:
                continue  # Skip the edge we just added
            
            other_constraint = edge_data.get('constraint')
            if other_constraint is None:
                continue
            
            other_constraint_id = self._extract_constraint_id(other_constraint)
            
            # 🆕 OPTIMIZACIÓN: Verificación O(1) con reglas precomputadas
            if not self.rules.is_commutative(new_constraint_id, other_constraint_id):
                continue  # No conmutan, skip
            
            # Las restricciones conmutan según las reglas precomputadas
            # Verificar que efectivamente llegan al mismo estado
            # Path 1: base_state -> new_state -> final_state_1
            constraint_state_other = frozenset([other_constraint])
            final_state_1 = self._meet(new_state, constraint_state_other)
            
            # Path 2: base_state -> target_state -> final_state_2
            final_state_2 = self._meet(target_state, frozenset([new_constraint]))
            
            # Check if they lead to the same state (commutativity)
            if final_state_1 == final_state_2 and final_state_1 != self.BOTTOM:
                # Homotopy detected!
                self._register_homotopy(base_state, new_state, target_state, final_state_1,
                                       new_constraint, other_constraint)
    
    def _extract_constraint_id(self, constraint: Constraint) -> str:
        """
        Extrae el ID de una restricción.
        
        Args:
            constraint: Tupla que representa una restricción
        
        Returns:
            ID de la restricción como string
        """
        # Las restricciones en el grafo son tuplas
        # El formato depende de cómo se agregaron
        # Asumimos que el primer elemento es el ID o podemos usar hash
        if isinstance(constraint, tuple) and len(constraint) > 0:
            return str(constraint[0]) if not isinstance(constraint[0], tuple) else str(hash(constraint))
        return str(hash(constraint))
    
    def _register_homotopy(self, base: State, state_a: State, state_b: State, final: State,
                          constraint_a: Constraint, constraint_b: Constraint):
        """
        Registers a detected homotopy (commutative square).
        
        Args:
            base: The base state
            state_a: State after applying constraint_a
            state_b: State after applying constraint_b
            final: The final state (same via both paths)
            constraint_a: First constraint
            constraint_b: Second constraint
        """
        # Ensure all edges exist in the graph
        if not self.graph.has_edge(state_a, final):
            self.graph.add_edge(state_a, final, constraint=constraint_b)
        if not self.graph.has_edge(state_b, final):
            self.graph.add_edge(state_b, final, constraint=constraint_a)
        
        # Record the homotopy
        homotopy = {
            'id': self.homotopy_counter,
            'base': base,
            'state_a': state_a,
            'state_b': state_b,
            'final': final,
            'constraint_a': constraint_a,
            'constraint_b': constraint_b
        }
        self.homotopies.append(homotopy)
        self.homotopy_counter += 1
        
        # Mark the homotopy in the graph (for visualization)
        self.graph.add_node(f"homotopy_{self.homotopy_counter}", 
                           type='homotopy', 
                           square=[base, state_a, state_b, final])
    
    def get_statistics(self) -> Dict[str, int]:
        """Returns statistics about the analysis."""
        return {
            'states': self.state_counter,
            'specifications': self.specification_counter,
            'homotopies': self.homotopy_counter,
            'graph_nodes': self.graph.number_of_nodes(),
            'graph_edges': self.graph.number_of_edges()
        }
    
    def export_to_json(self, filepath: str):
        """
        Exports the specification graph to JSON format.
        
        Args:
            filepath: Path to the output JSON file
        """
        def state_to_list(state: State) -> List[List]:
            result = []
            for c in state:
                # Convert each constraint tuple, handling frozensets
                constraint_list = []
                for item in c:
                    if isinstance(item, frozenset):
                        constraint_list.append(list(item))
                    else:
                        constraint_list.append(item)
                result.append(constraint_list)
            return result
        
        data = {
            'nodes': [],
            'edges': [],
            'homotopies': [],
            'statistics': self.get_statistics()
        }
        
        # Export nodes
        for node, attrs in self.graph.nodes(data=True):
            if isinstance(node, frozenset):
                node_data = {
                    'id': str(hash(node)),
                    'type': attrs.get('type', 'state'),
                    'label': attrs.get('label', ''),
                    'level': attrs.get('level', 0),
                    'constraints': state_to_list(node)
                }
                data['nodes'].append(node_data)
        
        # Export edges
        for source, target, attrs in self.graph.edges(data=True):
            if isinstance(source, frozenset) and isinstance(target, frozenset):
                constraint = attrs.get('constraint', ())
                # Convert constraint tuple, handling frozensets
                constraint_list = []
                for item in constraint:
                    if isinstance(item, frozenset):
                        constraint_list.append(list(item))
                    else:
                        constraint_list.append(item)
                
                edge_data = {
                    'source': str(hash(source)),
                    'target': str(hash(target)),
                    'constraint': constraint_list
                }
                data['edges'].append(edge_data)
        
        # Export homotopies
        for h in self.homotopies:
            homotopy_data = {
                'id': h['id'],
                'base': str(hash(h['base'])),
                'state_a': str(hash(h['state_a'])),
                'state_b': str(hash(h['state_b'])),
                'final': str(hash(h['final']))
            }
            data['homotopies'].append(homotopy_data)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def __repr__(self):
        stats = self.get_statistics()
        return (f"HomotopyAnalyzer(states={stats['states']}, "
                f"specifications={stats['specifications']}, "
                f"homotopies={stats['homotopies']})")

