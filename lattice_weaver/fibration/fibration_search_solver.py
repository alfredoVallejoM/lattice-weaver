import time
import uuid
from typing import Dict, List, Any, Optional, Tuple
import random

from .constraint_hierarchy import ConstraintHierarchy, ConstraintLevel, Hardness
from .energy_landscape_optimized import EnergyLandscapeOptimized, EnergyComponents
from ..arc_engine.core import ArcEngine
from .hacification_engine import HacificationEngine, HacificationResult
from .landscape_modulator import LandscapeModulator, ModulationStrategy, AdaptiveStrategy
from ..homotopy.rules import HomotopyRules

class FibrationSearchSolver:
    """
    Un solver de búsqueda que integra el Flujo de Fibración (HacificationEngine y LandscapeModulator)
    para encontrar soluciones óptimas en problemas con restricciones HARD y SOFT.

    Este solver utiliza una estrategia de búsqueda heurística guiada por el paisaje de energía
    modulado y la poda de dominios mediante hacificación.
    """
    def __init__(self, variables: List[str], domains: Dict[str, List[Any]], hierarchy: ConstraintHierarchy, max_iterations: int = 10000, max_backtracks: int = 50000, use_tms: bool = False, parallel: bool = False, parallel_mode: str = 'thread', use_homotopy: bool = False):
        self.variables = variables
        self.domains = domains
        self.hierarchy = hierarchy
        self.landscape = EnergyLandscapeOptimized(hierarchy)
        # Crear una instancia de ArcEngine para pasar al HacificationEngine
        self.arc_engine = ArcEngine(use_tms=use_tms, parallel=parallel, parallel_mode=parallel_mode, use_homotopy=use_homotopy)
        self.homotopy_rules: Optional[HomotopyRules] = None
        if use_homotopy:
            self.homotopy_rules = HomotopyRules()
            # La precomputación se hará después de que todas las restricciones estén en el ArcEngine
            # Esto se hará en el método solve o antes de la primera llamada a hacify/filter_coherent_extensions
            # Por ahora, solo inicializamos la clase.
        self.hacification_engine = HacificationEngine(hierarchy, self.landscape, self.arc_engine)
        self.modulator = LandscapeModulator(self.landscape)
        self.modulator.set_strategy(AdaptiveStrategy()) # Estrategia adaptativa por defecto

        self.best_solution: Optional[Dict[str, Any]] = None
        self.best_energy: float = float("inf")
        self.num_solutions_found: int = 0
        self.max_solutions: int = 100 # Buscar múltiples soluciones para optimización
        self.max_iterations = max_iterations
        self.max_backtracks = max_backtracks
        self.backtracks_count: int = 0
        self.nodes_visited: int = 0 # Para contar los nodos del árbol de búsqueda


    def solve(self, time_limit_seconds: int = 60) -> Optional[Dict[str, Any]]:
        """
        Inicia el proceso de búsqueda para encontrar la mejor solución.
        """
        self.best_solution = None
        self.best_energy = float("inf")
        self.num_solutions_found = 0
        self.backtracks_count = 0
        self.nodes_visited = 0
        self.start_time = time.time()
        self.time_limit_seconds = time_limit_seconds

        if self.homotopy_rules and not self.homotopy_rules._precomputed:
            # Añadir todas las restricciones HARD al ArcEngine para la precomputación
            temp_arc_engine_for_homotopy = ArcEngine()
            for level in ConstraintLevel:
                for constraint in self.hierarchy.get_constraints_at_level(level):
                    if constraint.hardness == Hardness.HARD and len(constraint.variables) == 2:
                        # Generar un nombre de relación único para cada restricción
                        base_relation_name = constraint.metadata.get("name", f"constraint_{id(constraint.predicate)}")
                        relation_name = f"{base_relation_name}_{id(constraint)}_{uuid.uuid4()}" # Asegurar unicidad global
                        def adapted_predicate(val1, val2, meta, constraint=constraint):
                            # El predicado original espera argumentos posicionales (val1, val2) o (val1)
                            if len(constraint.variables) == 2:
                                return constraint.predicate(val1, val2)
                            elif len(constraint.variables) == 1:
                                # Para restricciones unarias, ArcEngine pasa val1 y val2 (que es None)
                                # El predicado original solo espera val1
                                return constraint.predicate(val1)
                            else:
                                raise ValueError("Unsupported constraint arity for adapted_predicate")
                        temp_arc_engine_for_homotopy.register_relation(relation_name, adapted_predicate)
                        if len(constraint.variables) == 2:
                            temp_arc_engine_for_homotopy.add_constraint(constraint.variables[0], constraint.variables[1], relation_name, metadata=constraint.metadata)
                        elif len(constraint.variables) == 1:
                            # Para restricciones unarias, solo se añade una variable
                            # ArcEngine.add_constraint para unarias espera (variable, None, relation_name, metadata)
                            temp_arc_engine_for_homotopy.add_constraint(constraint.variables[0], None, relation_name, metadata=constraint.metadata)
                        else:
                            raise ValueError("Unsupported constraint arity for ArcEngine.add_constraint")
            
            # Añadir restricciones unarias HARD al ArcEngine para la precomputación si es necesario
            # HomotopyRules actualmente solo usa restricciones binarias para construir el grafo de dependencias.
            # Si en el futuro se extiende para unarias, se añadiría aquí.

            if temp_arc_engine_for_homotopy.constraints:
                self.homotopy_rules.precompute_from_engine(temp_arc_engine_for_homotopy)

        initial_assignment: Dict[str, Any] = {}
        self._search(initial_assignment, 0)
        return self.best_solution

    def _search(self, assignment: Dict[str, Any], iteration: int) -> None:
        if self.backtracks_count > self.max_backtracks or self.nodes_visited > self.max_iterations or (time.time() - self.start_time > self.time_limit_seconds):
            return

        if len(assignment) == len(self.variables):
            current_energy = self.landscape.compute_energy(assignment).total_energy
            if current_energy < self.best_energy:
                self.best_energy = current_energy
                self.best_solution = assignment.copy()
            self.num_solutions_found += 1
            self.nodes_visited += 1
            return

        self.nodes_visited += 1
        var = self._select_next_variable(assignment)
        if var is None:
            return
        if var not in self.domains or not self.domains[var]: # Dominio vacío para la variable seleccionada
            self.backtracks_count += 1
            return

        ordered_values = self._get_ordered_domain_values(var, assignment)

        # Ordenar los valores por la energía que producen, priorizando los que minimizan las violaciones SOFT
        # LCV: Least Constraining Value - elegir el valor que deja más opciones para las variables futuras.
        # Aquí, lo interpretamos como el valor que minimiza la energía total (HARD + SOFT) de la asignación parcial.
        for value in ordered_values:
            # Forward Checking implícito: el hacification_engine ya poda el dominio de la variable actual
            # con AC-3, lo que es una forma de forward checking para las restricciones HARD.
            # Aquí, la coherencia se verifica con la asignación parcial extendida.
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

            # Poda heurística para SOFT constraints: si la energía es significativamente peor en una asignación parcial
            # y ya hemos avanzado bastante en la asignación, podar.
            # El factor 1.05 (5% peor) y el 50% de variables asignadas son heurísticas ajustables.
            if self.best_solution is not None and h_result.energy.total_energy > self.best_energy * 1.05 and len(new_assignment) > len(self.variables) * 0.5:
                self.backtracks_count += 1
                continue # Podar si la energía es significativamente peor en una asignación parcial avanzada.

            self._search(new_assignment, iteration + 1)

        # self.backtracks_count += 1 # Se incrementa en el bucle for si se poda una rama

    def _select_next_variable(self, assignment: Dict[str, Any]) -> Optional[str]:
        if self.homotopy_rules and self.homotopy_rules._precomputed:
            # Usar el orden de variables optimizado de HomotopyRules
            # Esto requiere una nueva función en HomotopyRules para obtener el orden de variables
            # Por ahora, asumimos que existe y la llamamos
            try:
                ordered_vars = self.homotopy_rules.get_optimal_variable_order()
                unassigned_vars = [v for v in ordered_vars if v not in assignment]
            except (NotImplementedError, AttributeError):
                unassigned_vars = [v for v in self.variables if v not in assignment]
        else:
            unassigned_vars = [v for v in self.variables if v not in assignment]
        if not unassigned_vars: return None

        context = {
            "progress": len(assignment) / len(self.variables),
            "local_violations": self.landscape.compute_energy(assignment).local_energy,
            "global_violations": self.landscape.compute_energy(assignment).global_energy
        }
        self.modulator.apply_modulation(context)

        # MRV (Minimum Remaining Values) + Degree Heuristic
        # Seleccionar la variable no asignada con el dominio más pequeño.
        # En caso de empate, seleccionar la variable que participa en el mayor número de restricciones HARD
        # con otras variables no asignadas (heurística de grado).
        
        best_var = None
        min_domain_size = float('inf')
        max_degree = -1

        for var in unassigned_vars:
            # Obtener el dominio filtrado por restricciones HARD (AC-3 ya aplicado en hacification_engine)
            filtered_domain = self.hacification_engine.filter_coherent_extensions(assignment, var, self.domains[var], strict=True)
            current_domain_size = len(filtered_domain)

            if current_domain_size == 0: # Si el dominio está vacío, esta rama no tiene solución
                return var # Devolver esta variable para forzar un backtrack

            # Calcular el grado (número de restricciones HARD que involucran a 'var' y a otras variables no asignadas)
            current_degree = 0
            for const in self.hierarchy.get_constraints_at_level(ConstraintLevel.LOCAL) + \
                         self.hierarchy.get_constraints_at_level(ConstraintLevel.PATTERN) + \
                         self.hierarchy.get_constraints_at_level(ConstraintLevel.GLOBAL):
                if const.hardness == Hardness.HARD and var in const.variables:
                    for other_var in const.variables:
                        if other_var != var and other_var in unassigned_vars:
                            current_degree += 1
            
            # Aplicar MRV, y en caso de empate, la heurística de grado
            if current_domain_size < min_domain_size:
                min_domain_size = current_domain_size
                max_degree = current_degree
                best_var = var
            elif current_domain_size == min_domain_size:
                if current_degree > max_degree:
                    max_degree = current_degree
                    best_var = var
                elif current_degree == max_degree and random.random() < 0.5: # Desempate aleatorio
                    best_var = var
        return best_var

    def _get_ordered_domain_values(self, variable: str, assignment: Dict[str, Any]) -> List[Any]:
        # LCV: Primero, filtrar el dominio por restricciones HARD (strict=True)
        filtered_domain = self.hacification_engine.filter_coherent_extensions(assignment, variable, self.domains[variable], strict=True)
        
        # Si HomotopyRules está activo y precomputado, usarlo para ordenar los valores
        if self.homotopy_rules and self.homotopy_rules._precomputed:
            # Aquí se podría implementar una heurística de ordenamiento de valores basada en homotopía
            # Por ahora, se mantiene el ordenamiento por energía, pero se podría refinar.
            pass

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
            "nodes_visited": self.nodes_visited,
            "landscape_stats": self.landscape.get_cache_statistics(),
            "hacification_stats": self.hacification_engine.get_statistics(),
            "modulator_stats": self.modulator.get_statistics()
        }

