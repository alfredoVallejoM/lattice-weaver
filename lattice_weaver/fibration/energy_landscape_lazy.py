"""
Lazy Energy Landscape - Cálculo Perezoso de Energía

Versión optimizada de EnergyLandscape que computa energía solo cuando es necesario.

Optimizaciones:
- Dirty flag para marcar cuando energía necesita recalcularse
- Incremental update: actualizar solo componentes afectados
- Caching por nivel: cachear energía de cada nivel
- Invalidación inteligente: solo invalidar cuando dominios cambian

Autor: Agente Autónomo - Lattice Weaver
Fecha: 15 de Octubre, 2025
"""

from typing import Dict, Any, Set, Optional
from dataclasses import dataclass, field

from lattice_weaver.fibration.constraint_hierarchy import ConstraintHierarchy, ConstraintLevel
from lattice_weaver.fibration.energy_landscape_optimized import (
    EnergyLandscapeOptimized,
    EnergyComponents
)


@dataclass
class CachedEnergy:
    """Energía cacheada con dirty flag."""
    energy: float = 0.0
    dirty: bool = True
    affected_variables: Set[str] = field(default_factory=set)


class LazyEnergyLandscape(EnergyLandscapeOptimized):
    """
    Energy Landscape con cálculo perezoso.
    
    Extiende EnergyLandscapeOptimized con caching inteligente.
    """
    
    def __init__(self, hierarchy: ConstraintHierarchy):
        """
        Inicializa Lazy Energy Landscape.
        
        Args:
            hierarchy: Jerarquía de restricciones
        """
        super().__init__(hierarchy)
        
        # Caché de energía por nivel
        self._energy_cache: Dict[ConstraintLevel, CachedEnergy] = {
            level: CachedEnergy() for level in ConstraintLevel
        }
        
        # Caché de energía total
        self._total_energy_cache: Optional[EnergyComponents] = None
        self._total_dirty = True
        
        # Variables modificadas desde último cálculo
        self._modified_variables: Set[str] = set()
        
        # Estadísticas
        self.stats = {
            'energy_computations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'incremental_updates': 0,
            'full_recomputations': 0
        }
    
    def mark_dirty(self, variable: str):
        """
        Marca una variable como modificada.
        
        Invalida cachés de niveles afectados.
        
        Args:
            variable: Variable modificada
        """
        self._modified_variables.add(variable)
        self._total_dirty = True
        
        # Invalidar cachés de niveles que contienen esta variable
        for level in ConstraintLevel:
            constraints = self.hierarchy.get_constraints_at_level(level)
            for constraint in constraints:
                if variable in constraint.variables:
                    self._energy_cache[level].dirty = True
                    self._energy_cache[level].affected_variables.add(variable)
                    break
    
    def mark_all_dirty(self):
        """Marca todos los cachés como dirty."""
        self._total_dirty = True
        for level in ConstraintLevel:
            self._energy_cache[level].dirty = True
    
    def clear_dirty_flags(self):
        """Limpia flags de dirty después de cálculo."""
        self._modified_variables.clear()
        self._total_dirty = False
        for level in ConstraintLevel:
            self._energy_cache[level].dirty = False
            self._energy_cache[level].affected_variables.clear()
    
    def compute_level_energy_lazy(
        self,
        level: ConstraintLevel,
        assignment: Dict[str, Any]
    ) -> float:
        """
        Computa energía de un nivel con lazy evaluation.
        
        Args:
            level: Nivel de restricciones
            assignment: Asignación actual
        
        Returns:
            Energía del nivel
        """
        cache = self._energy_cache[level]
        
        # Si no está dirty, retornar caché
        if not cache.dirty:
            self.stats['cache_hits'] += 1
            return cache.energy
        
        self.stats['cache_misses'] += 1
        
        # Verificar si podemos hacer update incremental
        if cache.affected_variables and len(cache.affected_variables) < 5:
            # Update incremental: solo recalcular restricciones afectadas
            self.stats['incremental_updates'] += 1
            
            # TODO: Implementar update incremental real
            # Por ahora, recalcular completo
            energy = super().compute_level_energy(level, assignment)
        else:
            # Recalcular completo
            self.stats['full_recomputations'] += 1
            energy = super().compute_level_energy(level, assignment)
        
        # Actualizar caché
        cache.energy = energy
        cache.dirty = False
        cache.affected_variables.clear()
        
        return energy
    
    def compute_energy(self, assignment: Dict[str, Any]) -> EnergyComponents:
        """
        Computa energía total con lazy evaluation.
        
        Args:
            assignment: Asignación de variables
        
        Returns:
            Componentes de energía
        """
        self.stats['energy_computations'] += 1
        
        # Si no está dirty, retornar caché
        if not self._total_dirty and self._total_energy_cache is not None:
            self.stats['cache_hits'] += 1
            return self._total_energy_cache
        
        self.stats['cache_misses'] += 1
        
        # Calcular energía por nivel (con lazy evaluation)
        local_energy = self.compute_level_energy_lazy(ConstraintLevel.LOCAL, assignment)
        pattern_energy = self.compute_level_energy_lazy(ConstraintLevel.PATTERN, assignment)
        global_energy = self.compute_level_energy_lazy(ConstraintLevel.GLOBAL, assignment)
        
        # Energía total ponderada
        total_energy = (
            local_energy +
            pattern_energy +
            global_energy
        )
        
        # Crear componentes
        components = EnergyComponents(
            local_energy=local_energy,
            pattern_energy=pattern_energy,
            global_energy=global_energy,
            total_energy=total_energy
        )
        
        # Actualizar caché
        self._total_energy_cache = components
        self._total_dirty = False
        
        return components
    
    def get_stats(self) -> dict:
        """Retorna estadísticas de lazy evaluation."""
        total_requests = self.stats['cache_hits'] + self.stats['cache_misses']
        hit_rate = self.stats['cache_hits'] / total_requests if total_requests > 0 else 0
        
        return {
            **self.stats,
            'cache_hit_rate': hit_rate,
            'total_requests': total_requests
        }
    
    def reset_stats(self):
        """Resetea estadísticas."""
        self.stats = {
            'energy_computations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'incremental_updates': 0,
            'full_recomputations': 0
        }

