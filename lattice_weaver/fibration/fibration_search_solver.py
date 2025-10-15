import time
from typing import Dict, List, Any, Optional, Tuple
import random

from .constraint_hierarchy import ConstraintHierarchy, ConstraintLevel, Hardness
from .energy_landscape_optimized import EnergyLandscapeOptimized
from .hacification_engine import HacificationEngine, HacificationResult
from .landscape_modulator import LandscapeModulator, ModulationStrategy, AdaptiveStrategy
from .autoperturbation_system import AutoperturbationSystem
from .multiscale_compiler_api import MultiscaleCompilerAPI
from .simple_multiscale_compiler import SimpleMultiscaleCompiler
from .fibration_search_solver_api import FibrationSearchSolverAPI

class FibrationSearchSolver(FibrationSearchSolverAPI):
    """
    Implementa un solver de búsqueda avanzado que integra el Flujo de Fibración para
    resolver problemas de satisfacción de restricciones (CSP) y optimización (COP)
    con restricciones HARD y SOFT organizadas jerárquicamente.

    Este solver combina técnicas de búsqueda heurística (MRV, LCV), poda (Branch & Bound,
    hacificación) y modulación del paisaje de energía para encontrar soluciones óptimas
    de manera eficiente. Soporta la compilación multiescala de problemas para mejorar
    la escalabilidad.

    Attributes:
        variables (List[str]): Lista de nombres de variables en el problema.
        domains (Dict[str, List[Any]]): Diccionario que mapea cada variable a su dominio de valores posibles.
        hierarchy (ConstraintHierarchy): La jerarquía de restricciones del problema.
        multiscale_compiler (Optional[MultiscaleCompilerAPI]): Un compilador multiescala opcional para transformar el problema.
        landscape (EnergyLandscapeOptimized): El paisaje de energía utilizado para evaluar asignaciones.
        hacification_engine (HacificationEngine): Motor de hacificación para la poda de dominios.
        modulator (LandscapeModulator): Modulador del paisaje de energía para adaptar la estrategia de búsqueda.
        autoperturbation_system (AutoperturbationSystem): Sistema para perturbar la búsqueda y escapar de mínimos locales.
        best_solution (Optional[Dict[str, Any]]): La mejor solución encontrada hasta el momento.
        best_energy (float): La energía de la mejor solución encontrada.
        num_solutions_found (int): Número de soluciones completas encontradas.
        max_solutions (int): Límite de soluciones a buscar (actualmente no implementado para detener la búsqueda).
        max_iterations (int): Límite de nodos a visitar en el árbol de búsqueda.
        max_backtracks (int): Límite de retrocesos permitidos.
        backtracks_count (int): Contador de retrocesos realizados.
        nodes_visited (int): Contador de nodos visitados en el árbol de búsqueda.
        start_time (float): Marca de tiempo del inicio de la búsqueda.
        time_limit_seconds (int): Límite de tiempo en segundos para la búsqueda.
        internal_hierarchy (ConstraintHierarchy): Jerarquía de restricciones después de la compilación (si aplica).
        internal_domains (Dict[str, List[Any]]): Dominios de variables después de la compilación (si aplica).
        original_variables (List[str]): Variables originales antes de la compilación (si aplica).
        original_domains (Dict[str, List[Any]]): Dominios originales antes de la compilación (si aplica).
        compilation_metadata (Dict[str, Any]): Metadatos generados por el compilador multiescala.
    """
    def __init__(self,
                 variables: List[str],
                 domains: Dict[str, List[Any]],
                 hierarchy: ConstraintHierarchy,
                 multiscale_compiler: Optional[MultiscaleCompilerAPI] = None,
                 max_iterations: int = 10000, max_backtracks: int = 50000):
        """
        Inicializa el FibrationSearchSolver.

        Args:
            variables (List[str]): Nombres de las variables del problema.
            domains (Dict[str, List[Any]]): Dominios de cada variable.
            hierarchy (ConstraintHierarchy): Jerarquía de restricciones del problema.
            multiscale_compiler (Optional[MultiscaleCompilerAPI]): Compilador para transformar el problema.
            max_iterations (int): Número máximo de nodos a visitar en el árbol de búsqueda.
            max_backtracks (int): Número máximo de retrocesos permitidos.
        """
        self.multiscale_compiler = multiscale_compiler
        self.compilation_metadata: Dict[str, Any] = {}

        # Si se proporciona un compilador, el problema se transforma a una representación interna.
        if self.multiscale_compiler:
            print("Compilando problema con MultiscaleCompiler...")
            self.internal_hierarchy, self.internal_domains, self.compilation_metadata = \
                self.multiscale_compiler.compile_problem(hierarchy, domains)
            self.original_variables = variables # Se guardan las variables originales para la descompilación.
            self.original_domains = domains
        else:
            self.internal_hierarchy = hierarchy
            self.internal_domains = domains
            self.original_variables = variables
            self.original_domains = domains

        self.variables = list(self.internal_domains.keys()) # Las variables son las del problema compilado.
        self.domains = self.internal_domains # Los dominios son los del problema compilado.
        self.hierarchy = self.internal_hierarchy # La jerarquía es la del problema compilado.

        self.landscape = EnergyLandscapeOptimized(self.internal_hierarchy)
        self.hacification_engine = HacificationEngine(self.internal_hierarchy, self.landscape, self.internal_domains)
        self.modulator = LandscapeModulator(self.landscape)
        self.modulator.set_strategy(AdaptiveStrategy()) # Estrategia adaptativa por defecto para la modulación.
        self.autoperturbation_system = AutoperturbationSystem(self.internal_hierarchy, self.landscape, self.modulator, self.internal_domains)

        self.best_solution: Optional[Dict[str, Any]] = None
        self.best_energy: float = float("inf")
        self.num_solutions_found: int = 0
        self.max_solutions: int = 100 # Límite de soluciones a buscar (no detiene la búsqueda, solo cuenta).
        self.max_iterations = max_iterations
        self.max_backtracks = max_backtracks
        self.backtracks_count: int = 0
        self.nodes_visited: int = 0 # Contador de nodos visitados en el árbol de búsqueda.
        self.start_time: float = 0.0
        self.time_limit_seconds: int = 0

    def solve(self, time_limit_seconds: int = 60) -> Optional[Dict[str, Any]]:
        """
        Inicia el proceso de búsqueda para encontrar la mejor solución que satisfaga
        todas las restricciones HARD y minimice la energía de las restricciones SOFT.

        Args:
            time_limit_seconds (int): Límite de tiempo en segundos para la ejecución del solver.

        Returns:
            Optional[Dict[str, Any]]: La mejor solución encontrada como un diccionario de asignaciones
                                      (variable -> valor), o None si no se encuentra ninguna solución.
        """
        self.best_solution = None
        self.best_energy = float("inf")
        self.num_solutions_found = 0
        self.backtracks_count = 0
        self.nodes_visited = 0
        self.start_time = time.time()
        self.time_limit_seconds = time_limit_seconds

        initial_assignment: Dict[str, Any] = {}
        self._search(initial_assignment, 0)
        
        # Si se usó un compilador, la mejor solución encontrada se descompila a la representación original.
        if self.best_solution and self.multiscale_compiler:
            print("Descompilando la solución encontrada...")
            return self.multiscale_compiler.decompile_solution(self.best_solution, self.compilation_metadata)
        
        return self.best_solution

    def _search(self, assignment: Dict[str, Any], iteration: int) -> None:
        """
        Método recursivo de búsqueda con retroceso (backtracking) y poda.

        Args:
            assignment (Dict[str, Any]): La asignación parcial actual de variables.
            iteration (int): El número de iteración actual (profundidad en el árbol de búsqueda).
        """
        # Criterios de parada para la búsqueda (límites de retrocesos, nodos visitados, tiempo).
        if self.backtracks_count > self.max_backtracks or self.nodes_visited > self.max_iterations or (time.time() - self.start_time > self.time_limit_seconds):
            return

        # Si todas las variables han sido asignadas, se ha encontrado una solución completa.
        if len(assignment) == len(self.variables):
            all_hard_satisfied, current_energy, _, _, _ = self.landscape.compute_energy(assignment)
            if all_hard_satisfied and current_energy < self.best_energy:
                self.best_energy = current_energy
                self.best_solution = assignment.copy()
            self.num_solutions_found += 1
            self.nodes_visited += 1
            return

        self.nodes_visited += 1
        # Selección de la siguiente variable a asignar (MRV + heurística de grado).
        var = self._select_next_variable(assignment)
        if var is None:
            return # No hay más variables por asignar (debería ser capturado por len(assignment) == len(self.variables))
        if var not in self.domains or not self.domains[var]: # Dominio vacío para la variable seleccionada.
            self.backtracks_count += 1
            return # Retroceso si el dominio está vacío.

        # Obtener y ordenar los valores del dominio de la variable actual (LCV).
        ordered_values = self._get_ordered_domain_values(var, assignment)

        for value in ordered_values:
            new_assignment = assignment.copy()
            new_assignment[var] = value

            # Hacificación: Poda de dominios y verificación de coherencia para restricciones HARD.
            h_result = self.hacification_engine.hacify(new_assignment, strict=False)
            
            if h_result.has_hard_violation:
                self.backtracks_count += 1
                continue # Podar esta rama si hay violaciones HARD.

            # Branch & Bound: Poda si la energía actual de la asignación parcial ya es peor que la mejor solución encontrada.
            if self.best_solution is not None and h_result.energy[1] >= self.best_energy:
                self.backtracks_count += 1
                continue # Podar esta rama, no puede llevar a una solución mejor.

            # Poda heurística para SOFT constraints: si la energía es significativamente peor en una asignación parcial
            # y ya se ha avanzado bastante en la asignación, se poda la rama. Los umbrales son heurísticos y ajustables.
            if self.best_solution is not None and h_result.energy[1] > self.best_energy * 1.05 and len(new_assignment) > len(self.variables) * 0.5:
                self.backtracks_count += 1
                continue # Podar si la energía es significativamente peor en una asignación parcial avanzada.

            # Llamada recursiva para la siguiente variable.
            self._search(new_assignment, iteration + 1)

    def _select_next_variable(self, assignment: Dict[str, Any]) -> Optional[str]:
        """
        Selecciona la siguiente variable no asignada utilizando la heurística MRV (Minimum Remaining Values)
        y, en caso de empate, la heurística de grado (Degree Heuristic).

        Args:
            assignment (Dict[str, Any]): La asignación parcial actual.

        Returns:
            Optional[str]: El nombre de la siguiente variable a asignar, o None si todas están asignadas.
        """
        unassigned_vars = [v for v in self.variables if v not in assignment]
        if not unassigned_vars: return None

        # El modulador puede ajustar los pesos de nivel del paisaje de energía basado en el progreso.
        context = {
            "progress": len(assignment) / len(self.variables),
            # Los valores de violaciones locales/globales ahora se obtienen del desglose de compute_energy.
            # Para la selección de variable, se pueden usar los valores de la última evaluación de energía.
            # Sin embargo, para la modulación, el LandscapeModulator accede directamente al landscape.
            "local_violations": 0.0, # Placeholder, el modulador accede directamente
            "global_violations": 0.0 # Placeholder, el modulador accede directamente
        }
        self.modulator.apply_modulation(context)

        best_var = None
        min_domain_size = float("inf")
        max_degree = -1

        for var in unassigned_vars:
            # Obtener el dominio filtrado por restricciones HARD (AC-3 ya aplicado en hacification_engine).
            filtered_domain = self.hacification_engine.filter_coherent_extensions(assignment, var, self.domains[var], strict=True)
            current_domain_size = len(filtered_domain)

            if current_domain_size == 0: # Si el dominio está vacío, esta rama no tiene solución.
                return var # Devolver esta variable para forzar un backtrack.

            # Calcular el grado (número de restricciones HARD que involucran a 'var' y a otras variables no asignadas).
            current_degree = 0
            for const in self.hierarchy.get_constraints_by_level(ConstraintLevel.LOCAL) + \
                         self.hierarchy.get_constraints_by_level(ConstraintLevel.PATTERN) + \
                         self.hierarchy.get_constraints_by_level(ConstraintLevel.GLOBAL):
                if const.hardness == Hardness.HARD and var in const.variables:
                    for other_var in const.variables:
                        if other_var != var and other_var in unassigned_vars:
                            current_degree += 1
            
            # Aplicar MRV (Minimum Remaining Values), y en caso de empate, la heurística de grado.
            if current_domain_size < min_domain_size:
                min_domain_size = current_domain_size
                max_degree = current_degree
                best_var = var
            elif current_domain_size == min_domain_size:
                if current_degree > max_degree:
                    max_degree = current_degree
                    best_var = var
                elif current_degree == max_degree and random.random() < 0.5: # Desempate aleatorio para evitar sesgos.
                    best_var = var
        return best_var

    def _get_ordered_domain_values(self, variable: str, assignment: Dict[str, Any]) -> List[Any]:
        """
        Obtiene los valores del dominio de una variable, ordenados por la heurística LCV (Least Constraining Value).
        Los valores se ordenan según la energía total que producen en la asignación parcial extendida.

        Args:
            variable (str): La variable para la cual se ordenarán los valores del dominio.
            assignment (Dict[str, Any]): La asignación parcial actual.

        Returns:
            List[Any]: Una lista de valores del dominio, ordenados de menor a mayor energía.
        """
        # Primero, filtrar el dominio por restricciones HARD para asegurar la coherencia.
        filtered_domain = self.hacification_engine.filter_coherent_extensions(assignment, variable, self.domains[variable], strict=True)
        
        # Luego, ordenar los valores factibles por la energía que producen (incluyendo SOFT).
        def calculate_value_cost(value):
            temp_assignment = assignment.copy()
            temp_assignment[variable] = value
            _, total_energy, _, _, _ = self.landscape.compute_energy(temp_assignment)
            return total_energy
        
        return sorted(filtered_domain, key=calculate_value_cost)

    def set_modulation_strategy(self, strategy: ModulationStrategy):
        """
        Establece la estrategia de modulación del paisaje de energía.

        Args:
            strategy (ModulationStrategy): La nueva estrategia de modulación a aplicar.
        """
        self.modulator.set_strategy(strategy)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Devuelve un diccionario con estadísticas de rendimiento y estado del solver.

        Returns:
            Dict[str, Any]: Un diccionario que contiene métricas como la mejor energía,
                            número de soluciones encontradas, retrocesos, nodos visitados,
                            y estadísticas de los componentes internos (paisaje de energía, hacificación, modulador).
        """
        return {
            "best_energy": self.best_energy,
            "num_solutions_found": self.num_solutions_found,
            "backtracks_count": self.backtracks_count,
            "nodes_visited": self.nodes_visited,
            "landscape_stats": self.landscape.get_cache_statistics(),
            "hacification_stats": self.hacification_engine.get_statistics(),
            "modulator_stats": self.modulator.get_statistics()
        }

