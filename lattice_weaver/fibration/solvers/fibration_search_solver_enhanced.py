from typing import Dict, List, Optional, Any, Set
import time
import logging
import networkx as nx

from ..hacification_engine import HacificationEngine # Usar el refactorizado
from ..constraint_hierarchy import ConstraintHierarchy
from ..energy_landscape_optimized import EnergyLandscapeOptimized
from lattice_weaver.arc_engine.core import ArcEngine
# from lattice_weaver.homotopy.rules import HomotopyRules # Comentar por ahora

logger = logging.getLogger(__name__)


class FibrationSearchSolverEnhanced:
    """
    Solver de búsqueda con heurísticas mejoradas y backtracking inteligente.
    
    Mejoras sobre la versión original:
    - MRV mejorado: Combina tamaño de dominio + grado + centralidad + impacto
    - LCV mejorado: Combina energía + restricciones afectadas + conmutatividad
    - TMS integrado: Backjumping inteligente, explicaciones de conflictos
    - Sin copias de assignment: Gestión eficiente de estado
    - Configuración flexible: Pesos ajustables para heurísticas
    """
    
    def __init__(
        self,
        hierarchy: ConstraintHierarchy,
        landscape: EnergyLandscapeOptimized,
        arc_engine: ArcEngine,
        hacification_engine: HacificationEngine,
        variables: List[str],
        domains: Dict[str, List[Any]],
        use_homotopy: bool = False, # Deshabilitar por ahora
        use_tms: bool = True,
        use_enhanced_heuristics: bool = True,
        heuristic_weights: Optional[Dict[str, float]] = None,
        max_backtracks: int = 10000,
        max_iterations: int = 10000,
        time_limit_seconds: float = 60.0
    ):
        """
        Inicializa el solver mejorado.
        
        Args:
            hierarchy: Jerarquía de restricciones
            landscape: Paisaje de energía
            arc_engine: Motor de consistencia de arcos
            variables: Lista de variables del problema
            domains: Dominios de cada variable
            use_homotopy: Si True, usa HomotopyRules
            use_tms: Si True, usa TMS para backjumping
            use_enhanced_heuristics: Si True, usa heurísticas mejoradas
            heuristic_weights: Pesos para las heurísticas (opcional)
            max_backtracks: Límite de backtracks
            max_iterations: Límite de iteraciones
            time_limit_seconds: Límite de tiempo en segundos
        """
        self.hierarchy = hierarchy
        self.landscape = landscape
        self.arc_engine = arc_engine
        self.variables = variables
        self.domains = domains
        
        # Configuración de optimizaciones
        self.use_homotopy = use_homotopy
        self.use_tms = use_tms
        self.use_enhanced_heuristics = use_enhanced_heuristics
        
        # Pesos para heurísticas (configurables)
        self.heuristic_weights = heuristic_weights or {
            # MRV weights
            'mrv_domain_size': 0.4,
            'mrv_degree': 0.3,
            'mrv_centrality': 0.2,
            'mrv_impact': 0.1,
            # LCV weights
            'lcv_energy': 0.5,
            'lcv_affected': 0.3,
            'lcv_commutative': 0.2
        }
        
        # Límites de búsqueda
        self.max_backtracks = max_backtracks
        self.max_iterations = max_iterations
        self.time_limit_seconds = time_limit_seconds
        
        # Motor de hacificación optimizado
        self.hacification_engine = hacification_engine
        
        # HomotopyRules (si está habilitado)
        self.homotopy_rules: Optional[Any] = None # Cambiar a Any por ahora
        # if use_homotopy:
        #     self.homotopy_rules = HomotopyRules()
        
        # Estado de búsqueda
        self.best_solution: Optional[Dict[str, Any]] = None
        self.best_energy: float = float('inf')
        self.num_solutions_found: int = 0
        self.backtracks_count: int = 0
        self.start_time: float = 0.0
        
        # Estadísticas extendidas
        self.stats = {
            'nodes_explored': 0,
            'conflicts_analyzed': 0,
            'backjumps_performed': 0,
            'max_depth_reached': 0,
            'heuristic_calls': {
                'mrv': 0,
                'lcv': 0
            }
        }
        
        # Mapeo de variables a niveles de decisión (para TMS)
        self.variable_to_decision_level: Dict[str, int] = {}
        
    def solve(self) -> Optional[Dict[str, Any]]:
        """
        Resuelve el problema CSP usando búsqueda con backtracking mejorado.
        
        Returns:
            Mejor solución encontrada o None si no hay solución
        """
        logger.info("[FibrationSearchSolverEnhanced] Iniciando búsqueda...")
        
        # Inicializar tiempo
        self.start_time = time.time()
        
        # Precomputar HomotopyRules si está habilitado
        # if self.homotopy_rules and self.use_homotopy:
        #     logger.info("[FibrationSearchSolverEnhanced] Precomputando HomotopyRules...")
        #     try:
        #         self.homotopy_rules.precompute_from_engine(self.arc_engine)
        #         logger.info(f"  Pares conmutativos: {len(self.homotopy_rules.commutative_pairs)}")
        #         logger.info(f"  Grupos independientes: {len(self.homotopy_rules.independent_groups)}")
        #     except Exception as e:
        #         logger.warning(f"  Error precomputando HomotopyRules: {e}")
        #         self.use_homotopy = False
        
        # Iniciar búsqueda
        assignment = {}
        self._search(assignment, decision_level=0)
        
        # Reportar resultados
        elapsed = time.time() - self.start_time
        logger.info(f"[FibrationSearchSolverEnhanced] Búsqueda completada en {elapsed:.2f}s")
        logger.info(f"  Soluciones encontradas: {self.num_solutions_found}")
        logger.info(f"  Mejor energía: {self.best_energy:.4f}")
        logger.info(f"  Backtracks: {self.backtracks_count}")
        logger.info(f"  Nodos explorados: {self.stats['nodes_explored']}")
        logger.info(f"  Backjumps realizados: {self.stats['backjumps_performed']}")
        
        return self.best_solution
    
    def _search(self, assignment: Dict[str, Any], decision_level: int) -> None:
        """
        Búsqueda recursiva con backtracking inteligente.
        
        Args:
            assignment: Asignación actual (modificada in-place)
            decision_level: Nivel de decisión actual
        """
        self.stats['nodes_explored'] += 1
        self.stats['max_depth_reached'] = max(self.stats['max_depth_reached'], decision_level)
        
        # Verificar límites
        if self._should_stop():
            return
        
        # Caso base: solución completa
        if len(assignment) == len(self.variables):
            self._process_solution(assignment)
            return
        
        # Seleccionar próxima variable (MRV mejorado)
        var = self._select_next_variable(assignment)
        if var is None:
            return
        
        # Obtener valores ordenados (LCV mejorado)
        ordered_values = self._get_ordered_domain_values(var, assignment)
        
        # Probar cada valor
        for value in ordered_values:
            # Registrar decisión en TMS
            if self.use_tms and self.arc_engine.tms:
                self.arc_engine.tms.record_decision(var, value)
            
            # Hacer asignación (sin copia)
            assignment[var] = value
            self.variable_to_decision_level[var] = decision_level
            
            # Verificar coherencia
            h_result = self.hacification_engine.hacify(assignment, strict=False)
            
            if h_result.has_hard_violation:
                # Analizar conflicto con TMS
                if self.use_tms and self.arc_engine.tms:
                    conflict_level = self._analyze_conflict(var, assignment, decision_level)
                    
                    if conflict_level < decision_level:
                        # Backjumping: retroceder múltiples niveles
                        del assignment[var]
                        del self.variable_to_decision_level[var]
                        self.backtracks_count += 1
                        self.stats['backjumps_performed'] += 1
                        return  # Salir de este nivel
                
                # Backtrack normal
                del assignment[var]
                del self.variable_to_decision_level[var]
                self.backtracks_count += 1
                continue
            
            # Poda por energía (branch & bound)
            if self.best_solution is not None and h_result.energy.total_energy >= self.best_energy:
                del assignment[var]
                del self.variable_to_decision_level[var]
                self.backtracks_count += 1
                continue
            
            # Recursión al siguiente nivel
            self._search(assignment, decision_level + 1)
            
            # Deshacer asignación
            del assignment[var]
            del self.variable_to_decision_level[var]
            
            # Restaurar dominios si usa TMS
            if self.use_tms and self.arc_engine.tms:
                self.arc_engine.tms.backtrack_to_decision(decision_level)
        
        self.backtracks_count += 1
    
    def _should_stop(self) -> bool:
        """Verifica si se debe detener la búsqueda."""
        if self.backtracks_count > self.max_backtracks:
            logger.warning(f"[FibrationSearchSolverEnhanced] Límite de backtracks alcanzado: {self.backtracks_count}")
            return True
        
        if self.stats['nodes_explored'] > self.max_iterations:
            logger.warning(f"[FibrationSearchSolverEnhanced] Límite de iteraciones alcanzado: {self.stats['nodes_explored']}")
            return True
        
        if (time.time() - self.start_time) > self.time_limit_seconds:
            logger.warning(f"[FibrationSearchSolverEnhanced] Límite de tiempo alcanzado: {self.time_limit_seconds}s")
            return True
        
        return False
    
    def _process_solution(self, assignment: Dict[str, Any]) -> None:
        """
        Procesa una solución completa encontrada.
        """
        self.num_solutions_found += 1
        
        # Calcular energía de la solución
        energy_components = self.landscape.compute_energy(assignment)
        current_energy = energy_components.total_energy
        
        # Actualizar mejor solución si es mejor
        if current_energy < self.best_energy:
            self.best_energy = current_energy
            self.best_solution = assignment.copy() # Copiar para guardar el estado
            logger.info(f"[FibrationSearchSolverEnhanced] Nueva mejor solución encontrada con energía: {self.best_energy:.4f}")
        
    def _select_next_variable(self, assignment: Dict[str, Any]) -> Optional[str]:
        """
        Selecciona la próxima variable no asignada usando la heurística MRV mejorada.
        """
        unassigned_vars = [v for v in self.variables if v not in assignment]
        if not unassigned_vars:
            return None
        
        if not self.use_enhanced_heuristics:
            # MRV básico: variable con el dominio más pequeño
            return min(unassigned_vars, key=lambda var: len(self.domains[var]))
        
        # MRV mejorado: Combinación de factores
        best_var = None
        min_score = float('inf')
        
        for var in unassigned_vars:
            domain_size = len(self.domains[var])
            
             # Grado: número de restricciones que afectan a la variable
            degree = len(self.hierarchy.get_constraints_involving(var))
            
            # Centralidad (placeholder, se podría calcular con networkx)
            centrality = 1.0 # Dummy por ahora
            
            # Impacto (estimación de cuántos valores eliminaría una asignación)
            impact = 0.0 # Placeholder, difícil de calcular eficientemente sin ArcEngine
            
            score = (
                self.heuristic_weights['mrv_domain_size'] * domain_size +
                self.heuristic_weights['mrv_degree'] * (-degree) + # Negativo para maximizar
                self.heuristic_weights['mrv_centrality'] * (-centrality) + # Negativo para maximizar
                self.heuristic_weights['mrv_impact'] * impact
            )
            
            if score < min_score:
                min_score = score
                best_var = var
                
        self.stats['heuristic_calls']['mrv'] += 1
        return best_var
    
    def _get_ordered_domain_values(self, var: str, assignment: Dict[str, Any]) -> List[Any]:
        """
        Ordena los valores del dominio de una variable usando la heurística LCV mejorada.
        """
        if not self.use_enhanced_heuristics:
            return list(self.domains[var]) # Orden original
        
        # LCV mejorado: Combinación de factores
        scored_values = []
        for value in self.domains[var]:
            temp_assignment = assignment.copy()
            temp_assignment[var] = value
            
            # Energía: impacto en la energía total
            energy_impact = self.landscape.compute_energy(temp_assignment).total_energy
            
            # Restricciones afectadas: cuántas restricciones se ven afectadas
            affected_constraints = len(self.hierarchy.get_constraints_involving(var))
            
            # Conmutatividad (usando HomotopyRules si está habilitado)
            commutative_score = 0.0
            # if self.homotopy_rules and self.use_homotopy:
            #     # Placeholder: calcular score de conmutatividad
            #     commutative_score = self.homotopy_rules.get_commutativity_score(var, value, assignment)
            
            score = (
                self.heuristic_weights['lcv_energy'] * energy_impact +
                self.heuristic_weights['lcv_affected'] * affected_constraints +
                self.heuristic_weights['lcv_commutative'] * (-commutative_score) # Negativo para maximizar
            )
            scored_values.append((score, value))
            
        self.stats['heuristic_calls']['lcv'] += 1
        return [val for score, val in sorted(scored_values)]
    
    def _analyze_conflict(self, var: str, assignment: Dict[str, Any], decision_level: int) -> int:
        """
        Analiza el conflicto usando el TMS para determinar el nivel de backjump.
        """
        if not self.use_tms or not self.arc_engine.tms:
            return decision_level - 1 # Backtrack simple
        
        # El TMS debe ser capaz de analizar el conflicto y sugerir un nivel de backjump
        # Esto es una simplificación; el TMS real sería más complejo
        conflict_set = self.arc_engine.tms.get_conflict_set(var, assignment)
        
        # Encontrar el nivel de decisión más alto en el conjunto de conflicto que no sea el actual
        max_dl_in_conflict = 0
        for conflict_var in conflict_set:
            if conflict_var in self.variable_to_decision_level and self.variable_to_decision_level[conflict_var] < decision_level:
                max_dl_in_conflict = max(max_dl_in_conflict, self.variable_to_decision_level[conflict_var])
                
        self.stats['conflicts_analyzed'] += 1
        return max_dl_in_conflict



