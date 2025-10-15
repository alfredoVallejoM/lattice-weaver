"""
Fibration Search Solver Adaptativo

Este módulo implementa una versión adaptativa de FibrationSearchSolver que detecta
automáticamente las características del problema y selecciona la estrategia óptima.

Modos de operación:
- LITE: Problemas pequeños con solo restricciones HARD (sin HomotopyRules, sin TMS)
- MEDIUM: Problemas medianos o con algunas restricciones SOFT (sin HomotopyRules, con TMS)
- FULL: Problemas grandes o con jerarquía compleja (todas las optimizaciones)

Autor: Agente Autónomo - Lattice Weaver
Fecha: 15 de Octubre, 2025
"""

import time
import logging
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from functools import lru_cache
from enum import Enum

from lattice_weaver.fibration.constraint_hierarchy import ConstraintHierarchy, Hardness, ConstraintLevel
from lattice_weaver.fibration.energy_landscape_optimized import EnergyLandscapeOptimized
from lattice_weaver.fibration.hacification_engine_optimized import HacificationEngineOptimized
from lattice_weaver.arc_engine.core import ArcEngine
from lattice_weaver.homotopy.rules import HomotopyRules

logger = logging.getLogger(__name__)


class SolverMode(Enum):
    """Modos de operación del solver."""
    LITE = "lite"       # Mínimas optimizaciones
    MEDIUM = "medium"   # Optimizaciones medias
    FULL = "full"       # Todas las optimizaciones


@dataclass
class ProblemCharacteristics:
    """Características del problema para selección de modo."""
    n_variables: int
    n_constraints: int
    avg_domain_size: float
    max_domain_size: int
    has_soft_constraints: bool
    has_hierarchy: bool
    has_global_constraints: bool
    estimated_complexity: float
    
    def suggest_mode(self) -> SolverMode:
        """Sugiere el modo óptimo basándose en características."""
        # Problema pequeño y simple -> LITE
        if (self.n_variables < 20 and 
            self.max_domain_size < 10 and 
            not self.has_soft_constraints and
            not self.has_hierarchy):
            return SolverMode.LITE
        
        # Problema con restricciones SOFT o jerarquía -> FULL
        if self.has_soft_constraints or self.has_hierarchy or self.has_global_constraints:
            return SolverMode.FULL
        
        # Problema grande -> FULL (para aprovechar backjumping)
        if self.n_variables > 50 or self.max_domain_size > 50:
            return SolverMode.FULL
        
        # Caso intermedio -> MEDIUM
        return SolverMode.MEDIUM


class FibrationSearchSolverAdaptive:
    """
    Solver adaptativo que selecciona automáticamente la estrategia óptima.
    
    Este solver analiza las características del problema y decide dinámicamente
    qué optimizaciones activar para maximizar el rendimiento.
    """
    
    def __init__(
        self,
        hierarchy: ConstraintHierarchy,
        landscape: EnergyLandscapeOptimized,
        arc_engine: ArcEngine,
        variables: List[str],
        domains: Dict[str, List[Any]],
        mode: Optional[SolverMode] = None,  # None = auto-detect
        homotopy_threshold: int = 100,  # Backtracks antes de computar HomotopyRules
        max_backtracks: int = 10000,
        max_iterations: int = 10000,
        time_limit_seconds: float = 60.0
    ):
        """
        Inicializa el solver adaptativo.
        
        Args:
            hierarchy: Jerarquía de restricciones
            landscape: Landscape de energía
            arc_engine: Motor de consistencia de arcos
            variables: Lista de IDs de variables
            domains: Dominios iniciales {variable: [valores]}
            mode: Modo de operación (None = auto-detect)
            homotopy_threshold: Backtracks antes de computar HomotopyRules
            max_backtracks: Máximo número de backtracks
            max_iterations: Máximo número de iteraciones
            time_limit_seconds: Límite de tiempo en segundos
        """
        self.hierarchy = hierarchy
        self.landscape = landscape
        self.arc_engine = arc_engine
        self.variables = variables
        self.domains = {v: list(d) for v, d in domains.items()}
        self.homotopy_threshold = homotopy_threshold
        self.max_backtracks = max_backtracks
        self.max_iterations = max_iterations
        self.time_limit_seconds = time_limit_seconds
        
        # Analizar características del problema
        self.characteristics = self._analyze_problem()
        
        # Seleccionar modo
        if mode is None:
            self.mode = self.characteristics.suggest_mode()
            logger.info(f"[Adaptive] Modo auto-detectado: {self.mode.value}")
        else:
            self.mode = mode
            logger.info(f"[Adaptive] Modo manual: {self.mode.value}")
        
        # Configurar según modo
        self.use_homotopy = (self.mode == SolverMode.FULL)
        self.use_tms = (self.mode in [SolverMode.MEDIUM, SolverMode.FULL])
        self.use_enhanced_heuristics = (self.mode == SolverMode.FULL)
        
        # Lazy initialization de HomotopyRules
        self.homotopy_rules: Optional[HomotopyRules] = None
        self.homotopy_computed = False
        
        # Estado de búsqueda
        self.best_solution: Optional[Dict[str, Any]] = None
        self.best_energy: float = float('inf')
        self.backtracks = 0
        self.iterations = 0
        self.start_time: float = 0.0
        self.variable_to_decision_level: Dict[str, int] = {}
        
        # Caché de heurísticas
        self._mrv_cache: Dict[int, str] = {}
        self._lcv_cache: Dict[tuple, List[Any]] = {}
        
        # Estadísticas
        self.stats = {
            'mode': self.mode.value,
            'characteristics': {
                'n_variables': self.characteristics.n_variables,
                'n_constraints': self.characteristics.n_constraints,
                'avg_domain_size': self.characteristics.avg_domain_size,
                'has_soft': self.characteristics.has_soft_constraints,
                'has_hierarchy': self.characteristics.has_hierarchy
            },
            'search': {
                'backtracks': 0,
                'backjumps': 0,
                'nodes_explored': 0,
                'homotopy_computations': 0
            },
            'solution': {
                'found': False,
                'energy': float('inf'),
                'time_seconds': 0.0
            }
        }
        
        logger.info(f"[Adaptive] Inicializado en modo {self.mode.value}")
        logger.info(f"  Variables: {self.characteristics.n_variables}")
        logger.info(f"  Restricciones: {self.characteristics.n_constraints}")
        logger.info(f"  Dominio promedio: {self.characteristics.avg_domain_size:.1f}")
        logger.info(f"  Restricciones SOFT: {self.characteristics.has_soft_constraints}")
        logger.info(f"  Jerarquía: {self.characteristics.has_hierarchy}")
    
    def _analyze_problem(self) -> ProblemCharacteristics:
        """Analiza las características del problema."""
        n_variables = len(self.variables)
        
        # Contar restricciones por nivel
        n_local = len(self.hierarchy.get_constraints_at_level(ConstraintLevel.LOCAL))
        n_pattern = len(self.hierarchy.get_constraints_at_level(ConstraintLevel.PATTERN))
        n_global = len(self.hierarchy.get_constraints_at_level(ConstraintLevel.GLOBAL))
        n_constraints = n_local + n_pattern + n_global
        
        # Analizar dominios
        domain_sizes = [len(d) for d in self.domains.values()]
        avg_domain_size = sum(domain_sizes) / len(domain_sizes) if domain_sizes else 0
        max_domain_size = max(domain_sizes) if domain_sizes else 0
        
        # Detectar restricciones SOFT
        has_soft = False
        for level in [ConstraintLevel.LOCAL, ConstraintLevel.PATTERN, ConstraintLevel.GLOBAL]:
            for constraint in self.hierarchy.get_constraints_at_level(level):
                if constraint.hardness == Hardness.SOFT:
                    has_soft = True
                    break
            if has_soft:
                break
        
        # Detectar jerarquía (múltiples niveles con restricciones)
        levels_with_constraints = sum([
            1 if n_local > 0 else 0,
            1 if n_pattern > 0 else 0,
            1 if n_global > 0 else 0
        ])
        has_hierarchy = levels_with_constraints > 1
        
        # Detectar restricciones globales
        has_global_constraints = n_global > 0
        
        # Estimar complejidad (heurística simple)
        estimated_complexity = (
            n_variables * 
            avg_domain_size * 
            (n_constraints / max(n_variables, 1)) *
            (2.0 if has_soft else 1.0) *
            (1.5 if has_hierarchy else 1.0)
        )
        
        return ProblemCharacteristics(
            n_variables=n_variables,
            n_constraints=n_constraints,
            avg_domain_size=avg_domain_size,
            max_domain_size=max_domain_size,
            has_soft_constraints=has_soft,
            has_hierarchy=has_hierarchy,
            has_global_constraints=has_global_constraints,
            estimated_complexity=estimated_complexity
        )
    
    def solve(self) -> Optional[Dict[str, Any]]:
        """
        Resuelve el CSP con estrategia adaptativa.
        
        Returns:
            Mejor solución encontrada, o None si no hay solución
        """
        self.start_time = time.time()
        self.backtracks = 0
        self.iterations = 0
        self.best_solution = None
        self.best_energy = float('inf')
        
        logger.info(f"[Adaptive] Iniciando búsqueda en modo {self.mode.value}...")
        
        # Inicializar asignación vacía
        assignment: Dict[str, Any] = {}
        
        # Ejecutar búsqueda
        try:
            self._search(assignment, decision_level=0)
        except TimeoutError:
            logger.warning("[Adaptive] Tiempo límite alcanzado")
        
        # Actualizar estadísticas
        elapsed = time.time() - self.start_time
        self.stats['search']['backtracks'] = self.backtracks
        self.stats['search']['nodes_explored'] = self.iterations
        self.stats['solution']['found'] = self.best_solution is not None
        self.stats['solution']['energy'] = self.best_energy
        self.stats['solution']['time_seconds'] = elapsed
        
        logger.info(f"[Adaptive] Búsqueda completada en {elapsed:.2f}s")
        logger.info(f"  Soluciones encontradas: {1 if self.best_solution else 0}")
        logger.info(f"  Mejor energía: {self.best_energy:.4f}")
        logger.info(f"  Backtracks: {self.backtracks}")
        logger.info(f"  Nodos explorados: {self.iterations}")
        if self.use_tms:
            logger.info(f"  Backjumps realizados: {self.stats['search']['backjumps']}")
        if self.homotopy_computed:
            logger.info(f"  HomotopyRules computado: {self.stats['search']['homotopy_computations']} veces")
        
        return self.best_solution
    
    def _search(self, assignment: Dict[str, Any], decision_level: int):
        """Búsqueda recursiva con estrategia adaptativa."""
        # Verificar límites
        self.iterations += 1
        if self.iterations > self.max_iterations:
            raise TimeoutError("Max iterations reached")
        if time.time() - self.start_time > self.time_limit_seconds:
            raise TimeoutError("Time limit reached")
        
        # Caso base: solución completa
        if len(assignment) == len(self.variables):
            energy_components = self.landscape.compute_energy(assignment)
            total_energy = energy_components.total_energy
            if total_energy < self.best_energy:
                self.best_solution = assignment.copy()
                self.best_energy = total_energy
                logger.debug(f"[Adaptive] Nueva mejor solución: energía={total_energy:.4f}")
            return
        
        # Seleccionar siguiente variable
        var = self._select_next_variable(assignment)
        if var is None:
            return
        
        # Ordenar valores del dominio
        ordered_values = self._get_ordered_domain_values(var, assignment)
        
        # Probar cada valor
        for value in ordered_values:
            # Registrar decisión en TMS (si está activado)
            if self.use_tms and self.arc_engine.tms:
                self.arc_engine.tms.record_decision(var, value)
            
            # Hacer asignación
            assignment[var] = value
            self.variable_to_decision_level[var] = decision_level
            
            # Verificar consistencia con HacificationEngine
            hacification_result = self._hacify(assignment)
            
            if hacification_result.is_coherent:
                # Recursión
                self._search(assignment, decision_level + 1)
            else:
                # Backtrack
                self.backtracks += 1
                
                # Backjumping si TMS está activado
                if self.use_tms and self.arc_engine.tms:
                    # Aquí se implementaría backjumping real
                    # Por ahora es backtracking simple
                    self.stats['search']['backjumps'] += 0
            
            # Deshacer asignación
            del assignment[var]
            if var in self.variable_to_decision_level:
                del self.variable_to_decision_level[var]
    
    def _select_next_variable(self, assignment: Dict[str, Any]) -> Optional[str]:
        """Selecciona la siguiente variable a asignar."""
        unassigned = [v for v in self.variables if v not in assignment]
        if not unassigned:
            return None
        
        # Caché hit?
        assignment_hash = hash(frozenset(assignment.keys()))
        if assignment_hash in self._mrv_cache:
            cached_var = self._mrv_cache[assignment_hash]
            if cached_var in unassigned:
                return cached_var
        
        # Modo LITE o MEDIUM: MRV simple
        if self.mode in [SolverMode.LITE, SolverMode.MEDIUM]:
            var = min(unassigned, key=lambda v: len(self.domains[v]))
            self._mrv_cache[assignment_hash] = var
            return var
        
        # Modo FULL: MRV mejorado con HomotopyRules (lazy)
        if self.use_homotopy and self.backtracks > self.homotopy_threshold:
            if not self.homotopy_computed:
                self._compute_homotopy_rules()
        
        # MRV mejorado (4 componentes)
        best_var = None
        best_score = float('inf')
        
        for var in unassigned:
            # Componente 1: Tamaño de dominio (MRV clásico)
            domain_size = len(self.domains[var])
            
            # Componente 2: Grado (número de restricciones)
            degree = self._count_constraints_involving(var)
            
            # Componente 3: Dependencias (de HomotopyRules si está disponible)
            dependency_score = 0.0
            if self.homotopy_computed and self.homotopy_rules:
                # Usar información de dependencias
                dependency_score = len(self.homotopy_rules.get_dependencies(var))
            
            # Componente 4: Energía potencial
            energy_score = 0.0
            if self.characteristics.has_soft_constraints:
                # Estimar impacto en energía
                energy_score = self._estimate_energy_impact(var, assignment)
            
            # Score combinado (pesos ajustables)
            score = (
                1.0 * domain_size +
                0.1 * degree +
                0.05 * dependency_score +
                0.02 * energy_score
            )
            
            if score < best_score:
                best_score = score
                best_var = var
        
        self._mrv_cache[assignment_hash] = best_var
        return best_var
    
    def _get_ordered_domain_values(self, var: str, assignment: Dict[str, Any]) -> List[Any]:
        """Ordena los valores del dominio de una variable."""
        # Caché hit?
        cache_key = (var, hash(frozenset(assignment.items())))
        if cache_key in self._lcv_cache:
            return self._lcv_cache[cache_key]
        
        values = list(self.domains[var])
        
        # Modo LITE: sin ordenamiento
        if self.mode == SolverMode.LITE:
            self._lcv_cache[cache_key] = values
            return values
        
        # Modo MEDIUM o FULL: LCV (Least Constraining Value)
        if self.mode in [SolverMode.MEDIUM, SolverMode.FULL]:
            # Ordenar por número de valores eliminados en vecinos
            value_scores = []
            for value in values:
                # Contar cuántos valores eliminaría en vecinos
                eliminated = self._count_eliminated_values(var, value, assignment)
                value_scores.append((value, eliminated))
            
            # Ordenar por menor número de eliminaciones (LCV)
            value_scores.sort(key=lambda x: x[1])
            ordered_values = [v for v, _ in value_scores]
            
            self._lcv_cache[cache_key] = ordered_values
            return ordered_values
        
        return values
    
    def _compute_homotopy_rules(self):
        """Computa HomotopyRules (lazy)."""
        if self.homotopy_computed:
            return
        
        logger.info("[Adaptive] Computando HomotopyRules...")
        start = time.time()
        
        self.homotopy_rules = HomotopyRules(self.hierarchy)
        self.homotopy_rules.precompute_dependencies()
        
        elapsed = time.time() - start
        self.homotopy_computed = True
        self.stats['search']['homotopy_computations'] += 1
        
        logger.info(f"[Adaptive] HomotopyRules computado en {elapsed:.3f}s")
    
    def _hacify(self, assignment: Dict[str, Any]) -> Dict[str, Any]:
        """Ejecuta hacificación sobre la asignación."""
        engine = HacificationEngineOptimized(
            hierarchy=self.hierarchy,
            landscape=self.landscape,
            arc_engine=self.arc_engine
        )
        return engine.hacify(assignment, strict=True)
    
    def _count_constraints_involving(self, var: str) -> int:
        """Cuenta restricciones que involucran una variable."""
        count = 0
        for level in [ConstraintLevel.LOCAL, ConstraintLevel.PATTERN, ConstraintLevel.GLOBAL]:
            for constraint in self.hierarchy.get_constraints_at_level(level):
                if var in constraint.variables:
                    count += 1
        return count
    
    def _estimate_energy_impact(self, var: str, assignment: Dict[str, Any]) -> float:
        """Estima el impacto en energía de asignar una variable."""
        # Heurística simple: contar restricciones SOFT que involucran var
        impact = 0.0
        for level in [ConstraintLevel.LOCAL, ConstraintLevel.PATTERN, ConstraintLevel.GLOBAL]:
            for constraint in self.hierarchy.get_constraints_at_level(level):
                if var in constraint.variables and constraint.hardness == Hardness.SOFT:
                    impact += constraint.weight
        return impact
    
    def _count_eliminated_values(self, var: str, value: Any, assignment: Dict[str, Any]) -> int:
        """Cuenta valores que serían eliminados en vecinos si var=value."""
        # Heurística simple: contar restricciones binarias
        eliminated = 0
        temp_assignment = assignment.copy()
        temp_assignment[var] = value
        
        for level in [ConstraintLevel.LOCAL]:
            for constraint in self.hierarchy.get_constraints_at_level(level):
                if var in constraint.variables and len(constraint.variables) == 2:
                    other_var = [v for v in constraint.variables if v != var][0]
                    if other_var not in assignment:
                        # Contar valores inconsistentes
                        for other_value in self.domains[other_var]:
                            temp_assignment[other_var] = other_value
                            if not constraint.predicate(temp_assignment):
                                eliminated += 1
                            del temp_assignment[other_var]
        
        return eliminated
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estadísticas de la búsqueda."""
        return self.stats


def create_adaptive_solver(
    hierarchy: ConstraintHierarchy,
    landscape: EnergyLandscapeOptimized,
    arc_engine: ArcEngine,
    variables: List[str],
    domains: Dict[str, List[Any]],
    **kwargs
) -> FibrationSearchSolverAdaptive:
    """
    Factory function para crear un solver adaptativo.
    
    Args:
        hierarchy: Jerarquía de restricciones
        landscape: Landscape de energía
        arc_engine: Motor de consistencia de arcos
        variables: Lista de variables
        domains: Dominios iniciales
        **kwargs: Argumentos adicionales para el solver
    
    Returns:
        Solver adaptativo configurado
    """
    return FibrationSearchSolverAdaptive(
        hierarchy=hierarchy,
        landscape=landscape,
        arc_engine=arc_engine,
        variables=variables,
        domains=domains,
        **kwargs
    )

