"""
Motor de Consistencia Adaptativa (ACE) - Resolución Estructurada.

Este módulo implementa el motor principal que orquesta la resolución
de CSPs usando clustering dinámico y propagación de consistencia.

Autor: LatticeWeaver Team
Fecha: 12 de Octubre de 2025
Versión: 4.2.0
"""

from typing import Dict, Set, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from collections import deque
import time

from .graph_structures import ConstraintGraph, DynamicClusterGraph, Cluster
from .clustering import ClusterDetector, BoundaryManager, ClusteringMetrics


@dataclass
class SolutionStats:
    """
    Estadísticas de resolución.
    
    Attributes:
        solutions: Lista de soluciones encontradas
        nodes_explored: Nodos explorados en búsqueda
        backtracks: Número de backtracks
        arc_consistency_calls: Llamadas a AC-3
        cluster_operations: Operaciones de renormalización
        time_elapsed: Tiempo total de ejecución (segundos)
        clustering_metrics: Métricas del clustering inicial
    """
    solutions: List[Dict[str, any]] = field(default_factory=list)
    nodes_explored: int = 0
    backtracks: int = 0
    arc_consistency_calls: int = 0
    cluster_operations: Dict[str, int] = field(default_factory=lambda: {
        "merge": 0, "split": 0, "prune": 0
    })
    time_elapsed: float = 0.0
    clustering_metrics: Optional[ClusteringMetrics] = None
    
    def add_solution(self, solution: Dict[str, any]):
        """Añade una solución encontrada."""
        self.solutions.append(dict(solution))
    
    def __repr__(self) -> str:
        return (f"SolutionStats(solutions={len(self.solutions)}, "
                f"nodes={self.nodes_explored}, "
                f"backtracks={self.backtracks})")


class AC3Solver:
    """
    Implementación mejorada de AC-3 con last_support.
    
    AC-3.1 mantiene el último soporte encontrado para cada valor,
    acelerando la verificación de consistencia de arco.
    
    Examples:
        >>> solver = AC3Solver()
        >>> changed = solver.enforce_arc_consistency(cg, cluster_vars)
        >>> if not changed:
        ...     print("Clúster es arco-consistente")
    """
    
    def __init__(self):
        """Inicializa el solver AC-3."""
        self.last_support: Dict[Tuple[str, any, str], any] = {}
        self.calls = 0
    
    def enforce_arc_consistency(
        self,
        cg: ConstraintGraph,
        variables: Optional[Set[str]] = None
    ) -> bool:
        """
        Aplica AC-3 al grafo de restricciones.
        
        Args:
            cg: Grafo de restricciones
            variables: Conjunto de variables a procesar (None = todas)
        
        Returns:
            True si se encontró consistencia, False si hay inconsistencia
        """
        self.calls += 1
        
        if variables is None:
            variables = set(cg.get_all_variables())
        
        # Inicializar cola con todos los arcos
        queue = deque()
        for var in variables:
            for neighbor in cg.get_neighbors(var):
                if neighbor in variables:
                    queue.append((var, neighbor))
        
        # Procesar cola
        while queue:
            xi, xj = queue.popleft()
            
            revised = self._revise(cg, xi, xj)
            
            # Verificar si el dominio quedó vacío después de revise
            if len(cg.get_domain(xi)) == 0:
                return False  # Inconsistencia detectada
            
            if revised:
                # Dominio de xi cambió
                
                # Añadir vecinos de xi a la cola (excepto xj)
                for xk in cg.get_neighbors(xi):
                    if xk != xj and xk in variables:
                        queue.append((xk, xi))
        
        return True
    
    def _revise(self, cg: ConstraintGraph, xi: str, xj: str) -> bool:
        """
        Revisa el arco (xi, xj) usando AC-3.1 con last_support.
        
        Args:
            cg: Grafo de restricciones
            xi: Primera variable
            xj: Segunda variable
        
        Returns:
            True si el dominio de xi cambió
        """
        revised = False
        domain_xi = cg.get_domain(xi)
        domain_xj = cg.get_domain(xj)
        constraint = cg.get_constraint(xi, xj)
        
        if constraint is None:
            return False
        
        values_to_remove = []
        
        for vi in domain_xi:
            # Intentar usar last_support
            support_key = (xi, vi, xj)
            last_vj = self.last_support.get(support_key)
            
            # Verificar si last_support sigue siendo válido
            if last_vj is not None and last_vj in domain_xj:
                if constraint.evaluate(vi, last_vj):
                    continue  # Soporte válido encontrado
            
            # Buscar nuevo soporte
            support_found = False
            for vj in domain_xj:
                if constraint.evaluate(vi, vj):
                    # Soporte encontrado
                    self.last_support[support_key] = vj
                    support_found = True
                    break
            
            if not support_found:
                # Sin soporte: eliminar vi
                values_to_remove.append(vi)
                revised = True
        
        # Actualizar dominio
        if values_to_remove:
            new_domain = domain_xi - set(values_to_remove)
            if len(new_domain) > 0:
                cg.update_domain(xi, new_domain)
            # Si el dominio queda vacío, no actualizamos (dejamos que el caller lo detecte)
        
        return revised
    
    def reset_support_cache(self):
        """Limpia el caché de last_support."""
        self.last_support.clear()


class AdaptiveConsistencyEngine:
    """
    Motor de Consistencia Adaptativa (ACE).
    
    Orquesta la resolución de CSPs usando:
    1. Clustering dinámico de variables
    2. Resolución por clústeres con AC-3
    3. Propagación de fronteras
    4. Backtracking estructurado
    
    Examples:
        >>> engine = AdaptiveConsistencyEngine()
        >>> stats = engine.solve(problem, max_solutions=10)
        >>> print(f"Encontradas {len(stats.solutions)} soluciones")
    """
    
    def __init__(
        self,
        min_cluster_size: int = 2,
        max_cluster_size: int = 20,
        clustering_resolution: float = 1.0
    ):
        """
        Inicializa el motor ACE.
        
        Args:
            min_cluster_size: Tamaño mínimo de clúster
            max_cluster_size: Tamaño máximo de clúster
            clustering_resolution: Resolución del clustering
        """
        self.min_cluster_size = min_cluster_size
        self.max_cluster_size = max_cluster_size
        self.clustering_resolution = clustering_resolution
        
        self.cluster_detector = ClusterDetector(
            min_cluster_size=min_cluster_size,
            max_cluster_size=max_cluster_size,
            resolution=clustering_resolution
        )
        self.boundary_manager = BoundaryManager()
        self.ac3_solver = AC3Solver()
    
    def solve(
        self,
        cg: ConstraintGraph,
        max_solutions: int = 1,
        timeout: Optional[float] = None
    ) -> SolutionStats:
        """
        Resuelve el CSP usando clustering adaptativo.
        
        Args:
            cg: Grafo de restricciones
            max_solutions: Número máximo de soluciones a encontrar
            timeout: Timeout en segundos (None = sin límite)
        
        Returns:
            SolutionStats con soluciones y estadísticas
        """
        start_time = time.time()
        stats = SolutionStats()
        
        # Paso 1: Clustering inicial
        gcd, clustering_metrics = self.cluster_detector.detect_clusters(cg)
        stats.clustering_metrics = clustering_metrics
        
        # Paso 2: Aplicar AC-3 a cada clúster
        for cluster in gcd.get_active_clusters():
            consistent = self.ac3_solver.enforce_arc_consistency(
                cg, cluster.variables
            )
            stats.arc_consistency_calls += 1
            
            if not consistent:
                # Clúster inconsistente: marcar y podar
                cluster.mark_inconsistent()
                gcd.prune_cluster(cluster.id)
                stats.cluster_operations["prune"] += 1
        
        # Paso 3: Resolver con backtracking estructurado
        assignment: Dict[str, any] = {}
        self._backtrack_structured(
            cg, gcd, assignment, stats, max_solutions, start_time, timeout
        )
        
        stats.time_elapsed = time.time() - start_time
        return stats
    
    def _backtrack_structured(
        self,
        cg: ConstraintGraph,
        gcd: DynamicClusterGraph,
        assignment: Dict[str, any],
        stats: SolutionStats,
        max_solutions: int,
        start_time: float,
        timeout: Optional[float]
    ) -> bool:
        """
        Backtracking estructurado por clústeres.
        
        Args:
            cg: Grafo de restricciones
            gcd: Grafo de clústeres dinámico
            assignment: Asignación parcial actual
            stats: Estadísticas de resolución
            max_solutions: Número máximo de soluciones
            start_time: Tiempo de inicio
            timeout: Timeout en segundos
        
        Returns:
            True si se encontró solución, False si no hay más
        """
        # Verificar timeout
        if timeout is not None and (time.time() - start_time) > timeout:
            return False
        
        # Verificar si se alcanzó el límite de soluciones
        if len(stats.solutions) >= max_solutions:
            return False
        
        # Verificar si la asignación está completa
        if len(assignment) == len(cg.get_all_variables()):
            stats.add_solution(assignment)
            return len(stats.solutions) < max_solutions
        
        stats.nodes_explored += 1
        
        # Seleccionar siguiente variable (MRV - Minimum Remaining Values)
        var = self._select_variable(cg, assignment)
        
        if var is None:
            return False
        
        # Intentar valores del dominio
        for value in cg.get_domain(var):
            # Verificar consistencia con asignación actual
            if self._is_consistent(cg, var, value, assignment):
                # Hacer asignación
                assignment[var] = value
                
                # Propagar restricciones (AC-3 en vecinos)
                saved_domains = self._save_domains(cg)
                consistent = self._propagate_assignment(cg, var, value)
                
                if consistent:
                    # Continuar búsqueda
                    if self._backtrack_structured(
                        cg, gcd, assignment, stats, max_solutions, start_time, timeout
                    ):
                        return True
                
                # Deshacer asignación
                del assignment[var]
                self._restore_domains(cg, saved_domains)
                stats.backtracks += 1
        
        return False
    
    def _select_variable(
        self,
        cg: ConstraintGraph,
        assignment: Dict[str, any]
    ) -> Optional[str]:
        """
        Selecciona la siguiente variable a asignar usando MRV.
        
        Args:
            cg: Grafo de restricciones
            assignment: Asignación parcial actual
        
        Returns:
            Variable seleccionada o None si no hay
        """
        unassigned = [
            var for var in cg.get_all_variables()
            if var not in assignment
        ]
        
        if not unassigned:
            return None
        
        # MRV: seleccionar variable con menor dominio
        return min(unassigned, key=lambda v: len(cg.get_domain(v)))
    
    def _is_consistent(
        self,
        cg: ConstraintGraph,
        var: str,
        value: any,
        assignment: Dict[str, any]
    ) -> bool:
        """
        Verifica si asignar value a var es consistente.
        
        Args:
            cg: Grafo de restricciones
            var: Variable a asignar
            value: Valor a asignar
            assignment: Asignación parcial actual
        
        Returns:
            True si es consistente
        """
        for neighbor in cg.get_neighbors(var):
            if neighbor in assignment:
                constraint = cg.get_constraint(var, neighbor)
                if constraint and not constraint.evaluate(value, assignment[neighbor]):
                    return False
        return True
    
    def _propagate_assignment(
        self,
        cg: ConstraintGraph,
        var: str,
        value: any
    ) -> bool:
        """
        Propaga la asignación usando AC-3 en vecinos.
        
        Args:
            cg: Grafo de restricciones
            var: Variable asignada
            value: Valor asignado
        
        Returns:
            True si la propagación es consistente
        """
        # Reducir dominio de var a singleton
        cg.update_domain(var, {value})
        
        # Aplicar AC-3 en vecinos
        neighbors = set(cg.get_neighbors(var))
        if not neighbors:
            return True
        
        return self.ac3_solver.enforce_arc_consistency(cg, neighbors | {var})
    
    def _save_domains(self, cg: ConstraintGraph) -> Dict[str, set]:
        """Guarda los dominios actuales."""
        return {var: set(cg.get_domain(var)) for var in cg.get_all_variables()}
    
    def _restore_domains(self, cg: ConstraintGraph, saved: Dict[str, set]):
        """Restaura dominios guardados."""
        for var, domain in saved.items():
            cg.update_domain(var, domain)


class ClusterSolver:
    """
    Solver especializado para resolver clústeres individuales.
    
    Usa AC-3 y backtracking optimizado para resolver un clúster
    de variables de forma aislada.
    
    Examples:
        >>> solver = ClusterSolver()
        >>> solutions = solver.solve_cluster(cg, cluster_vars, max_solutions=10)
        >>> print(f"Encontradas {len(solutions)} soluciones parciales")
    """
    
    def __init__(self):
        """Inicializa el solver de clústeres."""
        self.ac3_solver = AC3Solver()
    
    def solve_cluster(
        self,
        cg: ConstraintGraph,
        cluster_vars: Set[str],
        max_solutions: int = 10,
        timeout: Optional[float] = None
    ) -> List[Dict[str, any]]:
        """
        Resuelve un clúster de variables.
        
        Args:
            cg: Grafo de restricciones
            cluster_vars: Variables del clúster
            max_solutions: Número máximo de soluciones
            timeout: Timeout en segundos
        
        Returns:
            Lista de soluciones parciales (asignaciones del clúster)
        """
        start_time = time.time()
        
        # Aplicar AC-3 al clúster
        consistent = self.ac3_solver.enforce_arc_consistency(cg, cluster_vars)
        if not consistent:
            return []  # Clúster inconsistente
        
        # Resolver con backtracking
        solutions = []
        assignment = {}
        
        self._backtrack_cluster(
            cg, cluster_vars, assignment, solutions,
            max_solutions, start_time, timeout
        )
        
        return solutions
    
    def _backtrack_cluster(
        self,
        cg: ConstraintGraph,
        cluster_vars: Set[str],
        assignment: Dict[str, any],
        solutions: List[Dict[str, any]],
        max_solutions: int,
        start_time: float,
        timeout: Optional[float]
    ) -> bool:
        """Backtracking para resolver clúster."""
        # Verificar timeout
        if timeout is not None and (time.time() - start_time) > timeout:
            return False
        
        # Verificar límite de soluciones
        if len(solutions) >= max_solutions:
            return False
        
        # Verificar si asignación completa
        if len(assignment) == len(cluster_vars):
            solutions.append(dict(assignment))
            return len(solutions) < max_solutions
        
        # Seleccionar variable
        var = self._select_variable(cg, cluster_vars, assignment)
        if var is None:
            return False
        
        # Intentar valores
        for value in cg.get_domain(var):
            if self._is_consistent(cg, var, value, assignment):
                assignment[var] = value
                
                if self._backtrack_cluster(
                    cg, cluster_vars, assignment, solutions,
                    max_solutions, start_time, timeout
                ):
                    return True
                
                del assignment[var]
        
        return False
    
    def _select_variable(
        self,
        cg: ConstraintGraph,
        cluster_vars: Set[str],
        assignment: Dict[str, any]
    ) -> Optional[str]:
        """Selecciona siguiente variable (MRV)."""
        unassigned = [v for v in cluster_vars if v not in assignment]
        if not unassigned:
            return None
        return min(unassigned, key=lambda v: len(cg.get_domain(v)))
    
    def _is_consistent(
        self,
        cg: ConstraintGraph,
        var: str,
        value: any,
        assignment: Dict[str, any]
    ) -> bool:
        """Verifica consistencia de asignación."""
        for neighbor in cg.get_neighbors(var):
            if neighbor in assignment:
                constraint = cg.get_constraint(var, neighbor)
                if constraint and not constraint.evaluate(value, assignment[neighbor]):
                    return False
        return True

