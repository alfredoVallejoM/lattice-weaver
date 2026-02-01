import logging
import random
import math
import time
from typing import Dict, List, Set, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum

from ..constraint_hierarchy import ConstraintHierarchy
from ..energy_landscape import EnergyLandscape # Usar el refactorizado

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
        landscape: EnergyLandscape,
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
        self.best_energy: float = float("inf")
        self.current_solution: Optional[Dict[str, Any]] = None
        self.current_energy: float = float("inf")
        
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
                
                # Actualizar mejor solución
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
        for iteration in range(self.config.local_iterations):
            self.stats['local_iterations'] += 1
            
            # Perturbar solución actual
            perturbed_solution = self._perturb_solution(
                self.current_solution,
                self.config.perturbation_strength
            )
            
            # Búsqueda local desde la solución perturbada
            local_optimum = self._local_search(perturbed_solution)
            
            if local_optimum is None:
                continue
            
            local_optimum_energy = self.landscape.compute_energy(local_optimum).total_energy
            
            # Aceptar si es mejor
            if local_optimum_energy < self.current_energy:
                self.current_solution = local_optimum
                self.current_energy = local_optimum_energy
                self.stats['improvements'] += 1
                
                # Actualizar mejor solución
                if local_optimum_energy < self.best_energy:
                    self.best_solution = local_optimum.copy()
                    self.best_energy = local_optimum_energy
                    logger.debug(f"[HybridSearch] Nueva mejor: {self.best_energy:.4f}")
            
            # Verificar tiempo límite
            if time.time() - self.stats.get('start_time', 0) > self.config.time_limit_seconds:
                break
    
    def _systematic_search_limited(self) -> Optional[Dict[str, Any]]:
        """Búsqueda sistemática limitada para encontrar una solución inicial."""
        # Placeholder: se podría usar un FibrationSearchSolver básico aquí
        # Por ahora, una asignación aleatoria simple
        assignment = {}
        for var in self.variables:
            if self.domains[var]:
                assignment[var] = random.choice(self.domains[var])
            else:
                return None # No se puede asignar
        self.stats['systematic_nodes'] += 1
        return assignment
    
    def _get_random_neighbor(self, solution: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Genera un vecino aleatorio cambiando el valor de una variable."""
        if not solution:
            return None
        
        neighbor = solution.copy()
        var_to_change = random.choice(list(self.variables))
        
        if self.domains[var_to_change]:
            new_value = random.choice(self.domains[var_to_change])
            neighbor[var_to_change] = new_value
        
        return neighbor
    
    def _destroy_solution(self, solution: Dict[str, Any], k: int) -> Dict[str, Any]:
        """Destruye k variables aleatorias de la solución."""
        destroyed_solution = solution.copy()
        vars_to_destroy = random.sample(list(self.variables), min(k, len(self.variables)))
        for var in vars_to_destroy:
            del destroyed_solution[var]
        return destroyed_solution
    
    def _reconstruct_solution(self, partial_solution: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Reconstruye una solución parcial usando un solver simple."""
        # Placeholder: se podría usar un FibrationSearchSolver básico aquí
        # Por ahora, asignación aleatoria de las variables no asignadas
        reconstructed_solution = partial_solution.copy()
        unassigned_vars = [v for v in self.variables if v not in reconstructed_solution]
        for var in unassigned_vars:
            if self.domains[var]:
                reconstructed_solution[var] = random.choice(self.domains[var])
            else:
                return None
        return reconstructed_solution
    
    def _perturb_solution(self, solution: Dict[str, Any], strength: float) -> Dict[str, Any]:
        """Perturba una solución cambiando un porcentaje de variables."""
        perturbed_solution = solution.copy()
        num_changes = int(len(self.variables) * strength)
        vars_to_change = random.sample(list(self.variables), num_changes)
        for var in vars_to_change:
            if self.domains[var]:
                perturbed_solution[var] = random.choice(self.domains[var])
        return perturbed_solution
    
    def _local_search(self, solution: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Realiza una búsqueda local (e.g., hill climbing) desde una solución."""
        # Placeholder: se podría usar una instancia de Hill Climbing aquí
        # Por ahora, simplemente devuelve la solución de entrada
        return solution


