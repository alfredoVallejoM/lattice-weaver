"""
Simple Optimization Solver - Versión simplificada y robusta

Usa backtracking con ordenamiento por energía para encontrar múltiples soluciones
y devolver la mejor.
"""

from typing import Dict, List, Optional, Any
from .constraint_hierarchy import ConstraintHierarchy, Hardness, ConstraintLevel
from .energy_landscape_optimized import EnergyLandscapeOptimized


class SimpleOptimizationSolver:
    """
    Solver de optimización simplificado.
    
    Explora el espacio de búsqueda ordenando valores por energía y mantiene
    la mejor solución encontrada.
    """
    
    def __init__(self, variables: List[str], domains: Dict[str, List[Any]],
                 hierarchy: ConstraintHierarchy):
        self.variables = variables
        self.initial_domains = {var: list(domain) for var, domain in domains.items()}
        self.hierarchy = hierarchy
        self.landscape = EnergyLandscapeOptimized(hierarchy)
        
        # Configuración
        self.max_solutions = 10  # Número máximo de soluciones a encontrar
        
        # Estadísticas
        self.nodes_explored = 0
        self.solutions_found = 0
        self.best_solution = None
        self.best_energy = float('inf')
        self.all_solutions = []
        
    def solve(self, max_nodes: int = 10000) -> Optional[Dict[str, Any]]:
        """
        Resuelve el problema buscando la solución de menor energía.
        
        Args:
            max_nodes: Número máximo de nodos a explorar
            
        Returns:
            Mejor solución encontrada o None
        """
        self.nodes_explored = 0
        self.solutions_found = 0
        self.best_solution = None
        self.best_energy = float('inf')
        self.all_solutions = []
        self.max_nodes = max_nodes
        
        self._backtrack({})
        
        return self.best_solution
    
    def _backtrack(self, assignment: Dict) -> None:
        """Backtracking con ordenamiento por energía."""
        if self.nodes_explored >= self.max_nodes:
            return
        
        if self.solutions_found >= self.max_solutions:
            return
        
        self.nodes_explored += 1
        
        # Solución completa
        if len(assignment) == len(self.variables):
            # Verificar restricciones HARD
            if self._check_hard_constraints(assignment):
                self.solutions_found += 1
                
                # Calcular energía
                energy = self.landscape.compute_energy(assignment).total_energy
                
                # Guardar solución
                self.all_solutions.append((assignment.copy(), energy))
                
                # Actualizar mejor solución
                if energy < self.best_energy:
                    self.best_energy = energy
                    self.best_solution = assignment.copy()
            
            return
        
        # Seleccionar variable
        var = self._select_variable_mrv(assignment)
        if var is None:
            return
        
        # Calcular energía base
        base_energy = self.landscape.compute_energy(assignment)
        
        # Calcular gradiente para ordenar valores
        gradient = self.landscape.compute_energy_gradient_optimized(
            assignment, base_energy, var, self.initial_domains[var]
        )
        
        # Ordenar valores por energía (explorar primero los más prometedores)
        sorted_values = sorted(gradient.items(), key=lambda x: x[1])
        
        # Explorar valores
        for value, energy in sorted_values:
            # Poda: si la energía ya supera la mejor solución, no continuar
            if energy >= self.best_energy:
                continue
            
            assignment[var] = value
            
            # Verificar consistencia con restricciones HARD
            if self._is_consistent_hard(assignment):
                self._backtrack(assignment)
            
            del assignment[var]
    
    def _select_variable_mrv(self, assignment: Dict) -> Optional[str]:
        """Selecciona variable usando MRV."""
        unassigned = [v for v in self.variables if v not in assignment]
        
        if not unassigned:
            return None
        
        # Seleccionar la primera variable no asignada
        # (podríamos usar MRV pero para simplificar usamos orden)
        return unassigned[0]
    
    def _is_consistent_hard(self, assignment: Dict) -> bool:
        """Verifica consistencia con restricciones HARD."""
        local_constraints = self.hierarchy.get_constraints_at_level(ConstraintLevel.LOCAL)
        
        for constraint in local_constraints:
            if constraint.hardness != Hardness.HARD:
                continue
            
            # Solo verificar si todas las variables están asignadas
            if not all(var in assignment for var in constraint.variables):
                continue
            
            satisfied, violation = constraint.evaluate(assignment)
            
            if not satisfied or violation > 0:
                return False
        
        return True
    
    def _check_hard_constraints(self, assignment: Dict) -> bool:
        """Verifica que todas las restricciones HARD estén satisfechas."""
        for level in ConstraintLevel:
            constraints = self.hierarchy.get_constraints_at_level(level)
            
            for constraint in constraints:
                if constraint.hardness != Hardness.HARD:
                    continue
                
                satisfied, violation = constraint.evaluate(assignment)
                
                if not satisfied or violation > 0:
                    return False
        
        return True
    
    def get_statistics(self) -> Dict:
        """Devuelve estadísticas de la búsqueda."""
        return {
            'nodes_explored': self.nodes_explored,
            'solutions_found': self.solutions_found,
            'best_energy': self.best_energy,
            'landscape_stats': self.landscape.get_cache_statistics()
        }

