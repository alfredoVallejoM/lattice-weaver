"""
Coherence Solver - Optimized Version

Solver optimizado que implementa:
1. Propagación de restricciones (constraint propagation)
2. Heurística MRV (Minimum Remaining Values)
3. Poda agresiva basada en energía
4. Detección temprana de conflictos
5. Uso del paisaje de energía optimizado

Parte de la implementación del Flujo de Fibración (Propuesta 2) - Fase 1 Optimizada.
"""

from typing import Dict, List, Optional, Any, Tuple
import heapq
from dataclasses import dataclass
from .constraint_hierarchy import ConstraintHierarchy, Hardness, ConstraintLevel
from .energy_landscape_optimized import EnergyLandscapeOptimized



@dataclass
class SearchNode:
    """Nodo en el árbol de búsqueda."""
    assignment: Dict[str, Any]
    domains: Dict[str, List[Any]]  # Dominios reducidos
    energy: Tuple[bool, float] # (all_hard_satisfied, total_energy)
    depth: int
    
    def __lt__(self, other):
        # Comparar por energía total (segundo elemento de la tupla)
        return self.energy[1] < other.energy[1]


class CoherenceSolverOptimized:
    """
    Solver optimizado que usa Flujo de Fibración con propagación de restricciones.
    
    Optimizaciones principales:
    1. Propagación de restricciones (reduce dominios)
    2. Heurística MRV para selección de variables
    3. Poda agresiva (solo valores con energía 0 para restricciones HARD)
    4. Detección temprana de conflictos
    5. Uso de paisaje de energía optimizado
    """
    
    def __init__(self, variables: List[str], domains: Dict[str, List[Any]]):
        """
        Inicializa el solver optimizado.
        
        Args:
            variables: Lista de variables del problema
            domains: Dominios iniciales de las variables
        """
        self.variables = variables
        self.initial_domains = {var: list(domain) for var, domain in domains.items()}
        
        # Componentes del sistema de fibración
        self.hierarchy = ConstraintHierarchy()
        self.landscape = EnergyLandscapeOptimized(self.hierarchy)
        
        # Estadísticas
        self.nodes_explored = 0
        self.nodes_pruned = 0
        self.propagations = 0
        self.conflicts_detected = 0
        
    def solve(self, max_nodes: int = 100000) -> Optional[Dict[str, Any]]:
        """
        Resuelve el problema usando búsqueda guiada por energía con propagación.
        
        Args:
            max_nodes: Número máximo de nodos a explorar
            
        Returns:
            Solución encontrada o None
        """
        self.nodes_explored = 0
        self.nodes_pruned = 0
        self.propagations = 0
        self.conflicts_detected = 0
        
        # Propagación inicial
        initial_domains = self._propagate_constraints({}, self.initial_domains)
        
        if initial_domains is None:
            return None  # Problema inconsistente
        
        return self._backtrack({}, initial_domains, max_nodes)
    
    def _backtrack(self, 
                  assignment: Dict[str, Any],
                  domains: Dict[str, List[Any]],
                  max_nodes: int) -> Optional[Dict[str, Any]]:
        """
        Backtracking con propagación de restricciones.
        
        Args:
            assignment: Asignación parcial actual
            domains: Dominios reducidos
            max_nodes: Máximo de nodos a explorar
            
        Returns:
            Solución o None
        """
        self.nodes_explored += 1
        
        if self.nodes_explored > max_nodes:
            return None
        
        # Verificar si es solución completa
        if len(assignment) == len(self.variables):
            # Verificar que todas las restricciones HARD están satisfechas
            all_hard_satisfied, total_energy = self.landscape.compute_energy(assignment)
            
            if all_hard_satisfied:
                return assignment
            return None
        
        # OPTIMIZACIÓN: Heurística MRV (Minimum Remaining Values)
        var = self._select_variable_mrv(assignment, domains)
        
        if var is None:
            return None
        
        # OPTIMIZACIÓN: Detección temprana de conflictos
        if not domains[var]:
            self.conflicts_detected += 1
            return None
        
        # Calcular energía base para cálculo incremental
        base_satisfied, base_total_energy = self.landscape.compute_energy(assignment, use_cache=True)
        base_energy = (base_satisfied, base_total_energy)
        
        # OPTIMIZACIÓN: Calcular gradiente de forma optimizada
        gradient = self.landscape.compute_energy_gradient_optimized(
            assignment, base_energy, var, domains[var]
        )
        
        # OPTIMIZACIÓN: Poda agresiva basada en tipo de restricciones
        pruned_values = self._prune_values(gradient, assignment)
        
        if not pruned_values:
            self.nodes_pruned += 1
            return None
        
        # Explorar valores en orden de energía creciente
        for value in pruned_values:
            assignment[var] = value
            
            # OPTIMIZACIÓN: Propagación de restricciones
            new_domains = self._propagate_constraints(assignment, domains)
            
            if new_domains is not None:
                result = self._backtrack(assignment, new_domains, max_nodes)
                if result is not None:
                    return result
            else:
                self.propagations += 1
            
            del assignment[var]
        
        return None
    
    def _select_variable_mrv(self, 
                            assignment: Dict[str, Any],
                            domains: Dict[str, List[Any]]) -> Optional[str]:
        """
        Selecciona variable usando heurística MRV (Minimum Remaining Values).
        
        OPTIMIZACIÓN: Selecciona la variable con menor dominio.
        Tie-breaker: variable con más restricciones (degree heuristic).
        
        Args:
            assignment: Asignación parcial
            domains: Dominios actuales
            
        Returns:
            Variable seleccionada o None
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
    
    def _prune_values(self,
                     gradient: Dict[Any, float],
                     assignment: Dict[str, Any]) -> List[Any]:
        """
        Poda valores basándose en el gradiente de energía.
        
        OPTIMIZACIÓN: Poda agresiva para restricciones HARD.
        - Si solo hay restricciones HARD: solo valores con energía 0
        - Si hay restricciones SOFT: tolerar pequeño aumento de energía
        
        Args:
            gradient: Gradiente de energía {valor: energía}
            assignment: Asignación actual
            
        Returns:
            Lista de valores ordenados por energía
        """
        if not gradient:
            return []
        
        # Ordenar valores por energía
        sorted_values = sorted(gradient.items(), key=lambda x: x[1])
        
        # Determinar si hay restricciones SOFT
        has_soft = any(
            len(self.hierarchy.get_constraints_at_level(level)) > 0
            for level in [ConstraintLevel.PATTERN, ConstraintLevel.GLOBAL]
        )
        
        if not has_soft:
            # OPTIMIZACIÓN: Solo restricciones HARD -> solo valores con energía 0
            pruned = [v for v, e in sorted_values if e == 0.0]
        else:
            # Hay restricciones SOFT -> tolerar pequeño aumento
            min_energy = sorted_values[0][1]
            
            # Si la mínima energía es 0, ser estricto
            # Aquí `e` es la energía total, que incluye soft constraints.
            # Si `min_energy` es 0, significa que todas las soft constraints también están satisfechas.
            # Si hay soft constraints, podemos tolerar un pequeño aumento de energía.
            # La lógica de poda debe ser más sofisticada para manejar HARD y SOFT de forma correcta.
            # Por ahora, si hay soft constraints, permitimos cualquier valor que no viole HARD constraints
            # y ordenamos por energía total.
            # La función `compute_energy_gradient_optimized` ya devuelve la energía total.
            # La `pruned_values` debería considerar `all_hard_satisfied` también.
            # Por simplicidad, si hay soft constraints, se devuelven todos los valores ordenados por energía.
            # Si no hay soft constraints, solo los que tienen energía 0.
            # Si hay restricciones HARD, solo se consideran los valores que las satisfacen (energía 0 para HARD)
            # Si no hay restricciones HARD violadas, entonces se ordenan por energía total (incluyendo SOFT)
            # La función compute_energy_gradient_optimized ya devuelve la energía total.
            # Aquí, `e` es la energía total para la asignación con ese valor.
            # Si `all_hard_satisfied` es False para un valor, ese valor no debe ser considerado.
            # La `compute_energy_gradient_optimized` ya debería filtrar esto, pero para mayor seguridad:
            # La lógica de `_prune_values` debería recibir el gradiente y el estado de `all_hard_satisfied`.
            # Por ahora, asumimos que el gradiente ya refleja la energía total, y que los valores con energía alta
            # (debido a violaciones HARD) ya están penalizados.
            # Simplificamos a devolver todos los valores ordenados por energía, y el solver se encargará de las HARD.
            pruned = [v for v, e in sorted_values]
        
        return pruned
    
    def _propagate_constraints(self,
                              assignment: Dict[str, Any],
                              domains: Dict[str, List[Any]]) -> Optional[Dict[str, List[Any]]]:
        """
        Propaga restricciones para reducir dominios.
        
        OPTIMIZACIÓN CRÍTICA: Implementa constraint propagation.
        Reduce dominios de variables no asignadas eliminando valores inconsistentes.
        
        Args:
            assignment: Asignación parcial
            domains: Dominios actuales
            
        Returns:
            Nuevos dominios reducidos o None si hay conflicto
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
                    if self._is_consistent_hard(temp_assignment):
                        consistent_values.append(value)
                
                # Si el dominio cambió, marcar como changed
                if len(consistent_values) < len(new_domains[var]):
                    changed = True
                    new_domains[var] = consistent_values
                
                # OPTIMIZACIÓN: Detección temprana de conflicto
                if not new_domains[var]:
                    return None  # Dominio vacío -> conflicto
        
        return new_domains
    
    def _is_consistent_hard(self, assignment: Dict[str, Any]) -> bool:
        """
        Verifica si una asignación es consistente con restricciones HARD.
        
        Args:
            assignment: Asignación a verificar
            
        Returns:
            True si es consistente
        """
        # Usar el método evaluate_solution de EnergyLandscapeOptimized para verificar hard constraints
        all_hard_satisfied, _ = self.landscape.compute_energy(assignment, use_cache=False)
        return all_hard_satisfied
    
    def _check_hard_constraints(self, assignment: Dict[str, Any]) -> bool:
        """
        Verifica que todas las restricciones HARD estén satisfechas.
        
        Args:
            assignment: Asignación completa
            
        Returns:
            True si todas las HARD están satisfechas
        """
        # Usar el método evaluate_solution de EnergyLandscapeOptimized para verificar hard constraints
        all_hard_satisfied, _ = self.landscape.compute_energy(assignment, use_cache=False)
        return all_hard_satisfied
    
    def get_statistics(self) -> Dict:
        """Devuelve estadísticas de la búsqueda."""
        landscape_stats = self.landscape.get_cache_statistics()
        
        return {
            'nodes_explored': self.nodes_explored,
            'nodes_pruned': self.nodes_pruned,
            'propagations': self.propagations,
            'conflicts_detected': self.conflicts_detected,
            'pruning_rate': self.nodes_pruned / max(self.nodes_explored, 1),
            'landscape_stats': landscape_stats
        }

