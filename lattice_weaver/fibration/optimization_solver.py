"""
Optimization Solver - Para problemas con restricciones SOFT

Este solver está diseñado específicamente para problemas con restricciones SOFT.
A diferencia del CoherenceSolverOptimized (que se detiene en la primera solución),
este solver explora múltiples soluciones y devuelve la de menor energía.

Estrategias:
1. Branch & Bound guiado por energía
2. Exploración de k-mejores valores por variable
3. Beam search con ancho configurable
"""

from typing import Dict, List, Optional, Any, Tuple
import heapq
from dataclasses import dataclass, field
from .constraint_hierarchy import ConstraintHierarchy, Hardness, ConstraintLevel
from .energy_landscape_optimized import EnergyLandscapeOptimized


@dataclass(order=True)
class SearchState:
    """Estado en el espacio de búsqueda."""
    energy: Tuple[bool, float] # (all_hard_satisfied, total_energy)
    assignment: Dict[str, Any] = field(compare=False)
    domains: Dict[str, List[Any]] = field(compare=False)
    depth: int = field(compare=False)




class OptimizationSolver:
    """
    Solver para optimización con restricciones SOFT.
    
    Explora múltiples soluciones y devuelve la de menor energía.
    """
    
    def __init__(self, variables: List[str], domains: Dict[str, List[Any]],
                 hierarchy: ConstraintHierarchy):
        """
        Inicializa el solver de optimización.
        
        Args:
            variables: Lista de variables del problema
            domains: Dominios iniciales de las variables
            hierarchy: Jerarquía de restricciones (HARD + SOFT)
        """
        self.variables = variables
        self.initial_domains = {var: list(domain) for var, domain in domains.items()}
        self.hierarchy = hierarchy
        self.landscape = EnergyLandscapeOptimized(hierarchy)
        
        # Configuración
        self.beam_width = 10  # Número de estados a mantener en cada nivel
        self.k_best_values = 3  # Número de mejores valores a explorar por variable
        
        # Estadísticas
        self.nodes_explored = 0
        self.solutions_found = 0
        self.best_energy = float('inf')
        
    def solve(self, max_nodes: int = 100000, 
             strategy: str = "beam_search") -> Optional[Dict[str, Any]]:
        """
        Resuelve el problema buscando la solución de menor energía.
        
        Args:
            max_nodes: Número máximo de nodos a explorar
            strategy: Estrategia de búsqueda ("beam_search", "branch_bound", "k_best")
            
        Returns:
            Mejor solución encontrada o None
        """
        self.nodes_explored = 0
        self.solutions_found = 0
        self.best_energy = float('inf')
        
        if strategy == "beam_search":
            return self._beam_search(max_nodes)
        elif strategy == "branch_bound":
            return self._branch_and_bound(max_nodes)
        elif strategy == "k_best":
            return self._k_best_search(max_nodes)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _beam_search(self, max_nodes: int) -> Optional[Dict[str, Any]]:
        """
        Beam search: mantiene los k mejores estados en cada nivel.
        
        Explora en anchura pero solo mantiene los estados más prometedores.
        """
        # Estado inicial
        initial_satisfied, initial_total_energy = self.landscape.compute_energy({})
        beam = [SearchState(
            energy=(initial_satisfied, initial_total_energy),
            assignment={},
            domains=self.initial_domains.copy(),
            depth=0
        )]
        
        best_solution = None
        best_energy = float('inf')
        
        while beam and self.nodes_explored < max_nodes:
            # Seleccionar siguiente variable (la misma para todos los estados del nivel)
            var = self._select_variable_mrv(beam[0].assignment, beam[0].domains)
            
            if var is None:
                # Todos los estados están completos
                break
            
            # Expandir todos los estados del beam
            next_beam = []
            
            for state in beam:
                if self.nodes_explored >= max_nodes:
                    break
                
                # Calcular energía base
                base_satisfied, base_total_energy = self.landscape.compute_energy(state.assignment)
                base_energy = (base_satisfied, base_total_energy)
                
                # Calcular gradiente
                gradient = self.landscape.compute_energy_gradient_optimized(
                    state.assignment, base_energy, var, state.domains[var]
                )
                
                # Explorar k mejores valores
                sorted_values = sorted(gradient.items(), key=lambda x: x[1])
                k_best = sorted_values[:self.k_best_values]
                
                for value, energy in k_best:
                    self.nodes_explored += 1
                    
                    # Crear nueva asignación
                    new_assignment = state.assignment.copy()
                    new_assignment[var] = value
                    
                    # Verificar si es solución completa
                    if len(new_assignment) == len(self.variables):
                        self.solutions_found += 1
                        
                        # Verificar restricciones HARD
                        if self._check_hard_constraints(new_assignment):
                            if energy < best_energy:
                                best_energy = energy
                                best_solution = new_assignment
                        continue
                    
                    # Propagar restricciones
                    new_domains = self._propagate_constraints(new_assignment, state.domains)
                    
                    if new_domains is not None:
                        next_beam.append(SearchState(
                            energy=(self.landscape.compute_energy(new_assignment)), # Recalcular para obtener (satisfied, total_energy)
                            assignment=new_assignment,
                            domains=new_domains,
                            depth=state.depth + 1
                        ))
            
            # Mantener solo los beam_width mejores estados
            next_beam.sort()
            beam = next_beam[:self.beam_width]
        
        return best_solution
    
    def _branch_and_bound(self, max_nodes: int) -> Optional[Dict[str, Any]]:
        """
        Branch & Bound: poda basada en cota superior de energía.
        
        Explora exhaustivamente pero poda ramas que no pueden mejorar la mejor solución.
        """
        best_solution = None
        best_energy = float('inf')
        
        def backtrack(assignment: Dict, domains: Dict, current_energy: Tuple[bool, float]):
            nonlocal best_solution, best_energy
            
            if self.nodes_explored >= max_nodes:
                return
            
            self.nodes_explored += 1
            
            # Poda: si la energía actual ya supera la mejor, no continuar
            if current_energy[1] >= best_energy:
                return
            
            # Solución completa
            if len(assignment) == len(self.variables):
                if self._check_hard_constraints(assignment):
                    self.solutions_found += 1
                    if current_energy[1] < best_energy:
                        best_energy = current_energy[1]
                        best_solution = assignment.copy()
                return
            
            # Seleccionar variable
            var = self._select_variable_mrv(assignment, domains)
            if var is None:
                return
            
            # Calcular gradiente
            gradient = self.landscape.compute_energy_gradient_optimized(
                assignment, current_energy, var, domains[var]
            )
            
            # Explorar valores ordenados por energía
            sorted_values = sorted(gradient.items(), key=lambda x: x[1])
            
            for value, energy in sorted_values:
                # Poda: si este valor ya supera la mejor solución, no explorar
                if energy >= best_energy:
                    break
                
                assignment[var] = value
                
                # Propagar restricciones
                new_domains = self._propagate_constraints(assignment, domains)
                
                if new_domains is not None:
                    new_energy = self.landscape.compute_energy_incremental(
                        {k: v for k, v in assignment.items() if k != var},
                        current_energy,
                        var,
                        value
                    )
                    backtrack(assignment, new_domains, new_energy)
                
                del assignment[var]
        
        backtrack({}, self.initial_domains.copy(), self.landscape.compute_energy({}))
        return best_solution
    
    def _k_best_search(self, max_nodes: int) -> Optional[Dict[str, Any]]:
        """
        K-Best Search: explora los k mejores valores de cada variable.
        
        Similar a backtracking pero solo explora los k valores más prometedores.
        """
        best_solution = None
        best_energy = float('inf')
        
        def backtrack(assignment: Dict, domains: Dict):
            nonlocal best_solution, best_energy
            
            if self.nodes_explored >= max_nodes:
                return
            
            self.nodes_explored += 1
            
            # Solución completa
            if len(assignment) == len(self.variables):
                all_hard_satisfied, total_energy = self.landscape.compute_energy(assignment)
                if all_hard_satisfied:
                    self.solutions_found += 1
                    if total_energy < best_energy:
                        best_energy = total_energy
                        best_solution = assignment.copy()
                return
            
            # Seleccionar variable
            var = self._select_variable_mrv(assignment, domains)
            if var is None:
                return
            
            # Calcular gradiente
            base_satisfied, base_total_energy = self.landscape.compute_energy(assignment)
            base_energy = (base_satisfied, base_total_energy)
            gradient = self.landscape.compute_energy_gradient_optimized(
                assignment, base_energy, var, domains[var]
            )
            
            # Explorar solo los k mejores valores
            sorted_values = sorted(gradient.items(), key=lambda x: x[1])
            k_best = sorted_values[:self.k_best_values]
            
            for value, energy in k_best:
                assignment[var] = value
                
                # Propagar restricciones
                new_domains = self._propagate_constraints(assignment, domains)
                
                if new_domains is not None:
                    backtrack(assignment, new_domains)
                
                del assignment[var]
        
        backtrack({}, self.initial_domains.copy())
        return best_solution
    
    def _select_variable_mrv(self, 
                            assignment: Dict[str, Any],
                            domains: Dict[str, List[Any]]) -> Optional[str]:
        """
        Selecciona variable usando heurística MRV (Minimum Remaining Values).
        """
        unassigned = [v for v in self.variables if v not in assignment]
        
        if not unassigned:
            return None
        
        # MRV: variable con menor dominio
        min_domain_size = min(len(domains[v]) for v in unassigned)
        mrv_vars = [v for v in unassigned if len(domains[v]) == min_domain_size]
        
        if len(mrv_vars) == 1:
            return mrv_vars[0]
        
        # Tie-breaker: Degree (más restricciones con variables no asignadas)
        degrees = {}
        for var in mrv_vars:
            constraints = self.hierarchy.get_constraints_involving(var)
            degree = sum(
                1 for c in constraints 
                if any(v not in assignment for v in c.variables if v != var)
            )
            degrees[var] = degree
        
        return max(degrees, key=degrees.get)

    def _propagate_constraints(self,
                              assignment: Dict[str, Any],
                              domains: Dict[str, List[Any]]) -> Optional[Dict[str, List[Any]]]:
        """
        Propaga restricciones para reducir dominios.
        """
        new_domains = {var: list(domain) for var, domain in domains.items()}
        changed = True
        
        while changed:
            changed = False
            
            for var in self.variables:
                if var in assignment:
                    continue
                
                # Filtrar valores inconsistentes
                consistent_values = []
                
                for value in new_domains[var]:
                    # Crear asignación temporal
                    temp_assignment = assignment.copy()
                    temp_assignment[var] = value
                    
                    # Verificar si es consistente con restricciones HARD
                    if self._check_hard_constraints(temp_assignment):
                        consistent_values.append(value)
                
                # Si el dominio cambió, marcar como changed
                if len(consistent_values) < len(new_domains[var]):
                    changed = True
                    new_domains[var] = consistent_values
                
                # Detección temprana de conflicto
                if not new_domains[var]:
                    return None  # Dominio vacío -> conflicto
        
        return new_domains

    def _check_hard_constraints(self, assignment: Dict[str, Any]) -> bool:
        """
        Verifica que todas las restricciones HARD estén satisfechas.
        """
        # Usar el método compute_energy de EnergyLandscapeOptimized para verificar hard constraints
        all_hard_satisfied, _ = self.landscape.compute_energy(assignment, use_cache=False)
        return all_hard_satisfied

    def get_statistics(self) -> Dict:
        """Devuelve estadísticas de la búsqueda."""
        landscape_stats = self.landscape.get_cache_statistics()
        
        return {
            'nodes_explored': self.nodes_explored,
            'solutions_found': self.solutions_found,
            'best_energy': self.best_energy,
            'landscape_stats': landscape_stats
        }

