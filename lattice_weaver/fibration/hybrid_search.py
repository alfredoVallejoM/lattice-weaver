"""
Búsqueda Local Híbrida

Combina búsqueda sistemática (backtracking) con búsqueda local (hill climbing,
simulated annealing) para escapar de mínimos locales y explorar eficientemente
espacios grandes.

Estrategias:
1. Systematic + Hill Climbing: Búsqueda sistemática hasta cierto punto, luego local
2. Systematic + Simulated Annealing: SA para escapar mínimos locales
3. Large Neighborhood Search (LNS): Destruir y reconstruir partes de la solución
4. Iterated Local Search (ILS): Perturbaciones + búsqueda local

Autor: Agente Autónomo - Lattice Weaver
Fecha: 15 de Octubre, 2025
"""

import logging
import random
import math
import time
from typing import Dict, List, Set, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum

from lattice_weaver.fibration.constraint_hierarchy import ConstraintHierarchy
from lattice_weaver.fibration.energy_landscape_optimized import EnergyLandscapeOptimized

logger = logging.getLogger(__name__)


class SearchStrategy(Enum):
    """Estrategias de búsqueda híbrida."""
    HILL_CLIMBING = "hill_climbing"
    SIMULATED_ANNEALING = "simulated_annealing"
    LARGE_NEIGHBORHOOD = "large_neighborhood"
    ITERATED_LOCAL = "iterated_local"


@dataclass
class HybridSearchConfig:
    """Configuración para búsqueda híbrida."""
    strategy: SearchStrategy
    systematic_depth: int = 5  # Profundidad de búsqueda sistemática antes de local
    local_iterations: int = 1000  # Iteraciones de búsqueda local
    temperature_initial: float = 100.0  # Temperatura inicial para SA
    temperature_decay: float = 0.95  # Factor de decay para SA
    neighborhood_size: int = 3  # Tamaño de vecindario para LNS
    perturbation_strength: float = 0.2  # Fuerza de perturbación para ILS
    time_limit_seconds: float = 60.0


class HybridSearch:
    """
    Motor de búsqueda híbrida.
    
    Combina búsqueda sistemática con búsqueda local para problemas grandes
    donde la búsqueda sistemática pura es demasiado lenta.
    """
    
    def __init__(
        self,
        hierarchy: ConstraintHierarchy,
        landscape: EnergyLandscapeOptimized,
        variables: List[str],
        domains: Dict[str, List[Any]],
        config: HybridSearchConfig
    ):
        """
        Inicializa búsqueda híbrida.
        
        Args:
            hierarchy: Jerarquía de restricciones
            landscape: Landscape de energía
            variables: Lista de variables
            domains: Dominios de variables
            config: Configuración de búsqueda
        """
        self.hierarchy = hierarchy
        self.landscape = landscape
        self.variables = variables
        self.domains = domains
        self.config = config
        
        # Estado de búsqueda
        self.best_solution: Optional[Dict[str, Any]] = None
        self.best_energy: float = float('inf')
        self.current_solution: Optional[Dict[str, Any]] = None
        self.current_energy: float = float('inf')
        
        # Estadísticas
        self.stats = {
            'strategy': config.strategy.value,
            'systematic_nodes': 0,
            'local_iterations': 0,
            'improvements': 0,
            'restarts': 0,
            'time_seconds': 0.0
        }
        
        logger.info(f"[HybridSearch] Inicializado con estrategia {config.strategy.value}")
    
    def search(self) -> Optional[Dict[str, Any]]:
        """
        Ejecuta búsqueda híbrida.
        
        Returns:
            Mejor solución encontrada, o None
        """
        start_time = time.time()
        
        logger.info(f"[HybridSearch] Iniciando búsqueda {self.config.strategy.value}...")
        
        # Seleccionar estrategia
        if self.config.strategy == SearchStrategy.HILL_CLIMBING:
            self._search_hill_climbing()
        elif self.config.strategy == SearchStrategy.SIMULATED_ANNEALING:
            self._search_simulated_annealing()
        elif self.config.strategy == SearchStrategy.LARGE_NEIGHBORHOOD:
            self._search_large_neighborhood()
        elif self.config.strategy == SearchStrategy.ITERATED_LOCAL:
            self._search_iterated_local()
        
        elapsed = time.time() - start_time
        self.stats['time_seconds'] = elapsed
        
        logger.info(f"[HybridSearch] Completado en {elapsed:.2f}s")
        logger.info(f"  Mejor energía: {self.best_energy:.4f}")
        logger.info(f"  Nodos sistemáticos: {self.stats['systematic_nodes']}")
        logger.info(f"  Iteraciones locales: {self.stats['local_iterations']}")
        logger.info(f"  Mejoras: {self.stats['improvements']}")
        
        return self.best_solution
    
    def _search_hill_climbing(self):
        """Búsqueda con Hill Climbing."""
        # Fase 1: Búsqueda sistemática hasta profundidad limitada
        initial_solution = self._systematic_search_limited()
        
        if initial_solution is None:
            # No se encontró solución inicial
            logger.warning("[HybridSearch] No se encontró solución inicial")
            return
        
        self.current_solution = initial_solution
        self.current_energy = self.landscape.compute_energy(initial_solution).total_energy
        self.best_solution = initial_solution.copy()
        self.best_energy = self.current_energy
        
        logger.info(f"[HybridSearch] Solución inicial: energía={self.current_energy:.4f}")
        
        # Fase 2: Hill Climbing
        for iteration in range(self.config.local_iterations):
            self.stats['local_iterations'] += 1
            
            # Generar vecino
            neighbor = self._get_random_neighbor(self.current_solution)
            
            if neighbor is None:
                continue
            
            # Evaluar vecino
            neighbor_energy = self.landscape.compute_energy(neighbor).total_energy
            
            # Aceptar si es mejor (hill climbing estricto)
            if neighbor_energy < self.current_energy:
                self.current_solution = neighbor
                self.current_energy = neighbor_energy
                self.stats['improvements'] += 1
                
                # Actualizar mejor solución
                if neighbor_energy < self.best_energy:
                    self.best_solution = neighbor.copy()
                    self.best_energy = neighbor_energy
                    logger.debug(f"[HybridSearch] Nueva mejor: {self.best_energy:.4f}")
            
            # Verificar tiempo límite
            if time.time() - self.stats.get('start_time', 0) > self.config.time_limit_seconds:
                break
    
    def _search_simulated_annealing(self):
        """Búsqueda con Simulated Annealing."""
        # Fase 1: Solución inicial
        initial_solution = self._systematic_search_limited()
        
        if initial_solution is None:
            logger.warning("[HybridSearch] No se encontró solución inicial")
            return
        
        self.current_solution = initial_solution
        self.current_energy = self.landscape.compute_energy(initial_solution).total_energy
        self.best_solution = initial_solution.copy()
        self.best_energy = self.current_energy
        
        logger.info(f"[HybridSearch] Solución inicial: energía={self.current_energy:.4f}")
        
        # Fase 2: Simulated Annealing
        temperature = self.config.temperature_initial
        
        for iteration in range(self.config.local_iterations):
            self.stats['local_iterations'] += 1
            
            # Generar vecino
            neighbor = self._get_random_neighbor(self.current_solution)
            
            if neighbor is None:
                continue
            
            # Evaluar vecino
            neighbor_energy = self.landscape.compute_energy(neighbor).total_energy
            delta = neighbor_energy - self.current_energy
            
            # Criterio de aceptación de Metropolis
            if delta < 0 or random.random() < math.exp(-delta / temperature):
                self.current_solution = neighbor
                self.current_energy = neighbor_energy
                
                if delta < 0:
                    self.stats['improvements'] += 1
                
                # Actualizar mejor solución
                if neighbor_energy < self.best_energy:
                    self.best_solution = neighbor.copy()
                    self.best_energy = neighbor_energy
                    logger.debug(f"[HybridSearch] Nueva mejor: {self.best_energy:.4f}")
            
            # Decay de temperatura
            temperature *= self.config.temperature_decay
            
            # Verificar tiempo límite
            if time.time() - self.stats.get('start_time', 0) > self.config.time_limit_seconds:
                break
    
    def _search_large_neighborhood(self):
        """Large Neighborhood Search (LNS)."""
        # Fase 1: Solución inicial
        initial_solution = self._systematic_search_limited()
        
        if initial_solution is None:
            logger.warning("[HybridSearch] No se encontró solución inicial")
            return
        
        self.current_solution = initial_solution
        self.current_energy = self.landscape.compute_energy(initial_solution).total_energy
        self.best_solution = initial_solution.copy()
        self.best_energy = self.current_energy
        
        logger.info(f"[HybridSearch] Solución inicial: energía={self.current_energy:.4f}")
        
        # Fase 2: LNS (destruir y reconstruir)
        for iteration in range(self.config.local_iterations):
            self.stats['local_iterations'] += 1
            
            # Destruir: eliminar k variables aleatorias
            destroyed = self._destroy_solution(
                self.current_solution,
                self.config.neighborhood_size
            )
            
            # Reconstruir: asignar las variables eliminadas
            reconstructed = self._reconstruct_solution(destroyed)
            
            if reconstructed is None:
                continue
            
            # Evaluar
            reconstructed_energy = self.landscape.compute_energy(reconstructed).total_energy
            
            # Aceptar si es mejor
            if reconstructed_energy < self.current_energy:
                self.current_solution = reconstructed
                self.current_energy = reconstructed_energy
                self.stats['improvements'] += 1
                
                if reconstructed_energy < self.best_energy:
                    self.best_solution = reconstructed.copy()
                    self.best_energy = reconstructed_energy
                    logger.debug(f"[HybridSearch] Nueva mejor: {self.best_energy:.4f}")
            
            # Verificar tiempo límite
            if time.time() - self.stats.get('start_time', 0) > self.config.time_limit_seconds:
                break
    
    def _search_iterated_local(self):
        """Iterated Local Search (ILS)."""
        # Fase 1: Solución inicial
        initial_solution = self._systematic_search_limited()
        
        if initial_solution is None:
            logger.warning("[HybridSearch] No se encontró solución inicial")
            return
        
        self.current_solution = initial_solution
        self.current_energy = self.landscape.compute_energy(initial_solution).total_energy
        self.best_solution = initial_solution.copy()
        self.best_energy = self.current_energy
        
        logger.info(f"[HybridSearch] Solución inicial: energía={self.current_energy:.4f}")
        
        # Fase 2: ILS (perturbación + búsqueda local)
        for restart in range(10):  # Múltiples restarts
            self.stats['restarts'] += 1
            
            # Perturbar solución actual
            perturbed = self._perturb_solution(
                self.current_solution,
                self.config.perturbation_strength
            )
            
            # Búsqueda local desde solución perturbada
            local_best = self._local_search(perturbed, iterations=100)
            
            if local_best is None:
                continue
            
            local_energy = self.landscape.compute_energy(local_best).total_energy
            
            # Aceptar si es mejor que la mejor global
            if local_energy < self.best_energy:
                self.best_solution = local_best.copy()
                self.best_energy = local_energy
                self.current_solution = local_best
                self.current_energy = local_energy
                logger.debug(f"[HybridSearch] Nueva mejor: {self.best_energy:.4f}")
            
            # Verificar tiempo límite
            if time.time() - self.stats.get('start_time', 0) > self.config.time_limit_seconds:
                break
    
    def _systematic_search_limited(self) -> Optional[Dict[str, Any]]:
        """
        Búsqueda sistemática hasta profundidad limitada.
        
        Returns:
            Solución parcial o completa, o None
        """
        # Implementación simplificada: asignación aleatoria válida
        assignment = {}
        
        for var in self.variables[:self.config.systematic_depth]:
            self.stats['systematic_nodes'] += 1
            
            # Probar valores en orden aleatorio
            values = list(self.domains[var])
            random.shuffle(values)
            
            assigned = False
            for value in values:
                assignment[var] = value
                
                # Verificar consistencia básica
                if self._is_consistent(assignment):
                    assigned = True
                    break
                
                del assignment[var]
            
            if not assigned:
                # No se pudo asignar esta variable
                return None
        
        # Completar asignación con valores aleatorios
        for var in self.variables[self.config.systematic_depth:]:
            if var not in assignment:
                assignment[var] = random.choice(self.domains[var])
        
        return assignment
    
    def _get_random_neighbor(self, solution: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Genera un vecino aleatorio cambiando una variable."""
        neighbor = solution.copy()
        var = random.choice(self.variables)
        new_value = random.choice(self.domains[var])
        neighbor[var] = new_value
        return neighbor
    
    def _destroy_solution(
        self,
        solution: Dict[str, Any],
        k: int
    ) -> Dict[str, Any]:
        """Destruye solución eliminando k variables aleatorias."""
        destroyed = solution.copy()
        vars_to_remove = random.sample(self.variables, min(k, len(self.variables)))
        for var in vars_to_remove:
            if var in destroyed:
                del destroyed[var]
        return destroyed
    
    def _reconstruct_solution(
        self,
        partial: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Reconstruye solución asignando variables faltantes."""
        reconstructed = partial.copy()
        
        for var in self.variables:
            if var not in reconstructed:
                reconstructed[var] = random.choice(self.domains[var])
        
        return reconstructed
    
    def _perturb_solution(
        self,
        solution: Dict[str, Any],
        strength: float
    ) -> Dict[str, Any]:
        """Perturba solución cambiando una fracción de variables."""
        perturbed = solution.copy()
        n_changes = max(1, int(len(self.variables) * strength))
        
        vars_to_change = random.sample(self.variables, n_changes)
        for var in vars_to_change:
            perturbed[var] = random.choice(self.domains[var])
        
        return perturbed
    
    def _local_search(
        self,
        initial: Dict[str, Any],
        iterations: int
    ) -> Optional[Dict[str, Any]]:
        """Búsqueda local simple (hill climbing)."""
        current = initial.copy()
        current_energy = self.landscape.compute_energy(current).total_energy
        
        for _ in range(iterations):
            self.stats['local_iterations'] += 1
            
            neighbor = self._get_random_neighbor(current)
            if neighbor is None:
                continue
            
            neighbor_energy = self.landscape.compute_energy(neighbor).total_energy
            
            if neighbor_energy < current_energy:
                current = neighbor
                current_energy = neighbor_energy
                self.stats['improvements'] += 1
        
        return current
    
    def _is_consistent(self, assignment: Dict[str, Any]) -> bool:
        """Verificación básica de consistencia."""
        # Simplificado: siempre retorna True
        # En implementación real, verificaría restricciones HARD
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estadísticas de búsqueda."""
        return self.stats


def create_hybrid_search(
    hierarchy: ConstraintHierarchy,
    landscape: EnergyLandscapeOptimized,
    variables: List[str],
    domains: Dict[str, List[Any]],
    strategy: SearchStrategy = SearchStrategy.SIMULATED_ANNEALING,
    **kwargs
) -> HybridSearch:
    """
    Factory para crear búsqueda híbrida.
    
    Args:
        hierarchy: Jerarquía de restricciones
        landscape: Landscape de energía
        variables: Variables del problema
        domains: Dominios de variables
        strategy: Estrategia de búsqueda
        **kwargs: Argumentos adicionales para config
    
    Returns:
        Motor de búsqueda híbrida configurado
    """
    config = HybridSearchConfig(strategy=strategy, **kwargs)
    return HybridSearch(hierarchy, landscape, variables, domains, config)

