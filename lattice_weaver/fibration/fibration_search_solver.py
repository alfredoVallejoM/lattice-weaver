import time
from typing import Dict, List, Any, Optional, Tuple
import random

from .constraint_hierarchy import ConstraintHierarchy, ConstraintLevel, Hardness
from .energy_landscape_optimized import EnergyLandscapeOptimized, EnergyComponents
from .hacification_engine import HacificationEngine, HacificationResult
from .landscape_modulator import LandscapeModulator, ModulationStrategy, AdaptiveStrategy

class FibrationSearchSolver:
    """
    Un solver de búsqueda que integra el Flujo de Fibración (HacificationEngine y LandscapeModulator)
    para encontrar soluciones óptimas en problemas con restricciones HARD y SOFT.

    Este solver utiliza una estrategia de búsqueda heurística guiada por el paisaje de energía
    modulado y la poda de dominios mediante hacificación.
    """
    def __init__(self, variables: List[str], domains: Dict[str, List[Any]], hierarchy: ConstraintHierarchy, max_iterations: int = 10000, max_backtracks: int = 50000):
        self.variables = variables
        self.domains = domains
        self.hierarchy = hierarchy
        self.landscape = EnergyLandscapeOptimized(hierarchy)
        self.hacification_engine = HacificationEngine(hierarchy, self.landscape)
        self.modulator = LandscapeModulator(self.landscape)
        self.modulator.set_strategy(AdaptiveStrategy()) # Estrategia adaptativa por defecto

        self.best_solution: Optional[Dict[str, Any]] = None
        self.best_energy: float = float("inf")
        self.num_solutions_found: int = 0
        self.max_solutions: int = 100 # Buscar múltiples soluciones para optimización
        self.max_iterations = max_iterations
        self.max_backtracks = max_backtracks
        self.backtracks_count: int = 0



    def solve(self, time_limit_seconds: int = 60) -> Optional[Dict[str, Any]]:
        """
        Inicia el proceso de búsqueda para encontrar la mejor solución.
        """
        self.best_solution = None
        self.best_energy = float("inf")
        self.num_solutions_found = 0
        self.backtracks_count = 0
        self.start_time = time.time()
        self.time_limit_seconds = time_limit_seconds

        initial_assignment: Dict[str, Any] = {}
        self._search(initial_assignment, 0)
        return self.best_solution

    def _search(self, assignment: Dict[str, Any], iteration: int) -> None:
        if self.backtracks_count > self.max_backtracks or iteration > self.max_iterations or (time.time() - self.start_time > self.time_limit_seconds):
            return

        if len(assignment) == len(self.variables):
            current_energy = self.landscape.compute_energy(assignment).total_energy
            if current_energy < self.best_energy:
                self.best_energy = current_energy
                self.best_solution = assignment.copy()
            self.num_solutions_found += 1
            return

        var = self._select_next_variable(assignment)
        if var is None:
            return

        ordered_values = self._get_ordered_domain_values(var, assignment)

        # Ordenar los valores por la energía que producen, priorizando los que minimizan las violaciones SOFT
        # LCV: Least Constraining Value - elegir el valor que deja más opciones para las variables futuras.
        # Aquí, lo interpretamos como el valor que minimiza la energía total (HARD + SOFT) de la asignación parcial.
        for value in ordered_values:
            new_assignment = assignment.copy()
            new_assignment[var] = value

            h_result = self.hacification_engine.hacify(new_assignment, strict=False)
            
            if h_result.has_hard_violation:
                self.backtracks_count += 1
                continue # Podar si hay violaciones HARD

            # Branch & Bound principal: Podar si la energía actual de la asignación parcial ya es peor que la mejor solución encontrada.
            # Solo podar si ya tenemos una solución completa para comparar.
            if self.best_solution is not None and h_result.energy.total_energy >= self.best_energy:
                self.backtracks_count += 1
                continue # Podar esta rama, no puede llevar a una solución mejor

            # Poda más flexible para SOFT constraints: Podar si la energía es significativamente peor en una asignación parcial
            # y no es una asignación completa. El umbral (ej. 1.1 para 10% peor) puede ser ajustado.
            # Para una exploración más exhaustiva de SOFT constraints, reducimos el factor de poda.
            # Poda más flexible para SOFT constraints: Solo podar si la energía actual de la asignación parcial
            # es significativamente peor que la mejor solución encontrada Y la asignación parcial ya es muy larga.
            # Esto permite explorar caminos con energía temporalmente más alta que podrían llevar a un óptimo global.
            # El factor de 1.2 (20% peor) y la condición de len(new_assignment) > len(self.variables) / 2 son heurísticas.
            # Poda más agresiva: si la energía de la asignación parcial ya es peor que la mejor solución encontrada
            # y no hay posibilidad de mejorar, podar.
            # Considerar un umbral más estricto o un análisis de cotas inferiores si es posible.
            if self.best_solution is not None and h_result.energy.total_energy >= self.best_energy:
                self.backtracks_count += 1
                continue # Podar esta rama, no puede llevar a una solución mejor

            # Poda heurística para SOFT constraints: si la energía es significativamente peor en una asignación parcial
            # y ya hemos avanzado bastante en la asignación, podar.
            # El factor 1.05 (5% peor) y el 50% de variables asignadas son heurísticas ajustables.
            if self.best_solution is not None and h_result.energy.total_energy > self.best_energy * 1.05 and len(new_assignment) > len(self.variables) * 0.5:
                self.backtracks_count += 1
                continue # Podar si la energía es significativamente peor en una asignación parcial avanzada.

            self._search(new_assignment, iteration + 1)

        self.backtracks_count += 1

    def _select_next_variable(self, assignment: Dict[str, Any]) -> Optional[str]:
        unassigned_vars = [v for v in self.variables if v not in assignment]
        if not unassigned_vars: return None

        context = {
            "progress": len(assignment) / len(self.variables),
            "local_violations": self.landscape.compute_energy(assignment).local_energy,
            "global_violations": self.landscape.compute_energy(assignment).global_energy
        }
        self.modulator.apply_modulation(context)

        min_domain_size = float("inf")
        best_var = None
        for var in unassigned_vars:
            # MRV: Usar strict=True para filtrar el dominio solo por restricciones HARD
            filtered_domain = self.hacification_engine.filter_coherent_extensions(assignment, var, self.domains[var], strict=True)
            # MRV: Minimum Remaining Values - seleccionar la variable con el dominio más pequeño.
            # Esto ayuda a detectar fallos antes.
            if len(filtered_domain) < min_domain_size or (len(filtered_domain) == min_domain_size and random.random() < 0.5):
                min_domain_size = len(filtered_domain)
                best_var = var
            elif len(filtered_domain) == min_domain_size:
                if random.random() < 0.5:
                    best_var = var
        return best_var

    def _get_ordered_domain_values(self, variable: str, assignment: Dict[str, Any]) -> List[Any]:
        # LCV: Primero, filtrar el dominio por restricciones HARD (strict=True)
        filtered_domain = self.hacification_engine.filter_coherent_extensions(assignment, variable, self.domains[variable], strict=True)
        
        # Luego, ordenar los valores factibles por la energía que producen (incluyendo SOFT)
        def calculate_value_cost(value):
            temp_assignment = assignment.copy()
            temp_assignment[variable] = value
            return self.landscape.compute_energy(temp_assignment).total_energy
        
        return sorted(filtered_domain, key=calculate_value_cost)

    def set_modulation_strategy(self, strategy: ModulationStrategy):
        self.modulator.set_strategy(strategy)

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "best_energy": self.best_energy,
            "num_solutions_found": self.num_solutions_found,
            "backtracks_count": self.backtracks_count,
            "landscape_stats": self.landscape.get_cache_statistics(),
            "hacification_stats": self.hacification_engine.get_statistics(),
            "modulator_stats": self.modulator.get_statistics()
        }

