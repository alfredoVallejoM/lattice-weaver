from typing import Dict, List, Any, Optional, Tuple
import random

from .constraint_hierarchy import ConstraintHierarchy
from .energy_landscape_optimized import EnergyLandscapeOptimized
from .hacification_engine import HacificationEngine

class HillClimbingFibrationSolver:
    """
    Un solver de búsqueda local que utiliza una estrategia de Hill Climbing para encontrar soluciones
    en problemas con restricciones HARD y SOFT. La búsqueda se guía por el paisaje de energía.
    
    Mejoras:
    - Inicialización más robusta: Genera soluciones aleatorias que satisfacen restricciones HARD.
    - Búsqueda de vecinos más eficiente: Considera solo un subconjunto aleatorio de vecinos o variables.
    """
    def __init__(self, variables: List[str], domains: Dict[str, List[Any]], hierarchy: ConstraintHierarchy, max_iterations: int = 1000, num_restarts: int = 10, neighbor_sampling_rate: float = 0.1):
        self.variables = variables
        self.domains = domains
        self.hierarchy = hierarchy
        self.landscape = EnergyLandscapeOptimized(hierarchy)
        self.hacification_engine = HacificationEngine(hierarchy, self.landscape)
        self.max_iterations = max_iterations
        self.num_restarts = num_restarts
        self.neighbor_sampling_rate = neighbor_sampling_rate # Tasa de muestreo para vecinos

    def solve(self) -> Optional[Dict[str, Any]]:
        """
        Inicia el proceso de búsqueda con múltiples reinicios para encontrar la mejor solución.
        """
        best_solution: Optional[Dict[str, Any]] = None
        best_energy: float = float("inf")

        for restart_idx in range(self.num_restarts):
            current_solution = self._generate_random_valid_solution()
            if current_solution is None: 
                # print(f"No se pudo generar una solución inicial válida en el reinicio {restart_idx+1}")
                continue

            current_energy = self.landscape.compute_energy(current_solution).total_energy

            for iter_idx in range(self.max_iterations):
                neighbor, neighbor_energy = self._get_best_neighbor(current_solution)
                
                # Si no hay un vecino mejor o la energía no mejora, hemos llegado a un óptimo local
                if neighbor is None or neighbor_energy >= current_energy:
                    break 
                
                current_solution = neighbor
                current_energy = neighbor_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_solution = current_solution

        return best_solution

    def _generate_random_valid_solution(self) -> Optional[Dict[str, Any]]:
        """
        Genera una solución aleatoria que satisface las restricciones HARD.
        Retorna None si no se puede encontrar una solución válida en un número razonable de intentos.
        """
        # Intentar encontrar una solución válida de forma aleatoria
        for _ in range(500): # Aumentar el número de intentos
            assignment = {}
            # Asignar valores aleatorios a las variables
            for var in self.variables:
                assignment[var] = random.choice(self.domains[var])
            
            # Verificar si la asignación satisface las restricciones HARD
            h_result = self.hacification_engine.hacify(assignment, strict=True)
            if not h_result.has_hard_violation:
                return assignment
        return None

    def _get_best_neighbor(self, current_solution: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], float]:
        """
        Encuentra el mejor vecino de la solución actual.
        Un vecino se define como una solución con un solo valor de variable cambiado.
        Se muestrea un subconjunto de variables para mejorar la eficiencia.
        """
        best_neighbor: Optional[Dict[str, Any]] = None
        best_neighbor_energy: float = float("inf")

        # Muestrear un subconjunto de variables para considerar como vecinos
        # Asegurarse de que al menos una variable sea muestreada
        num_variables_to_sample = max(1, int(len(self.variables) * self.neighbor_sampling_rate))
        variables_to_sample = random.sample(self.variables, num_variables_to_sample)

        for var in variables_to_sample:
            original_value = current_solution[var]
            for value in self.domains[var]:
                if value == original_value: continue

                neighbor = current_solution.copy()
                neighbor[var] = value

                # Verificar restricciones HARD para el vecino
                h_result = self.hacification_engine.hacify(neighbor, strict=True)
                if h_result.has_hard_violation: continue

                neighbor_energy = self.landscape.compute_energy(neighbor).total_energy
                if neighbor_energy < best_neighbor_energy:
                    best_neighbor_energy = neighbor_energy
                    best_neighbor = neighbor
        
        return best_neighbor, best_neighbor_energy

