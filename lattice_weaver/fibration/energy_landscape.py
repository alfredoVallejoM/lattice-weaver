"""
Energy Landscape Module

Este módulo implementa el paisaje de energía del espacio de búsqueda.
El paisaje se define mediante un funcional de energía que combina
violaciones de restricciones a todos los niveles de la jerarquía.

Parte de la implementación del Flujo de Fibración (Propuesta 2).
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from .constraint_hierarchy import ConstraintHierarchy, Constraint, ConstraintLevel


@dataclass
class EnergyComponents:
    """
    Desglose de la energía por componentes.
    
    Attributes:
        local_energy: Energía de restricciones locales
        pattern_energy: Energía de restricciones de patrón
        global_energy: Energía de restricciones globales
        total_energy: Energía total (suma ponderada)
    """
    local_energy: float
    pattern_energy: float
    global_energy: float
    total_energy: float
    
    def __repr__(self):
        return (f"E_total={self.total_energy:.3f} "
                f"(local={self.local_energy:.3f}, "
                f"pattern={self.pattern_energy:.3f}, "
                f"global={self.global_energy:.3f})")


class EnergyLandscape:
    """
    Define el paisaje de energía del espacio de búsqueda.
    
    El paisaje se construye a partir de la jerarquía de restricciones,
    asignando una energía a cada asignación parcial basada en el grado
    de violación de las restricciones.
    
    Los atractores del paisaje son mínimos locales (asignaciones con
    energía localmente mínima).
    """
    
    def __init__(self, hierarchy: ConstraintHierarchy):
        """
        Inicializa el paisaje de energía.
        
        Args:
            hierarchy: Jerarquía de restricciones
        """
        self.hierarchy = hierarchy
        
        # Pesos por nivel (modulables dinámicamente)
        self.level_weights = {
            ConstraintLevel.LOCAL: 1.0,
            ConstraintLevel.PATTERN: 1.0,
            ConstraintLevel.GLOBAL: 1.0
        }
        
        # Cache de energías calculadas
        # Clave: hash de la asignación, Valor: EnergyComponents
        self._energy_cache: Dict[str, EnergyComponents] = {}
        
        # Estadísticas
        self.cache_hits = 0
        self.cache_misses = 0
        
    def compute_energy(self, 
                      assignment: Dict[str, Any], 
                      use_cache: bool = True) -> EnergyComponents:
        """
        Calcula la energía de una asignación parcial.
        
        La energía se calcula como:
        E(assignment) = Σ_level w_level * Σ_constraint w_constraint * violation
        
        Args:
            assignment: Diccionario {variable: valor}
            use_cache: Si usar caché de energías
            
        Returns:
            EnergyComponents con desglose de energía
        """
        # Generar clave de cache
        cache_key = self._assignment_to_key(assignment)
        
        if use_cache and cache_key in self._energy_cache:
            self.cache_hits += 1
            return self._energy_cache[cache_key]
        
        self.cache_misses += 1
        
        # Calcular energía por nivel
        local_energy = self._compute_level_energy(assignment, ConstraintLevel.LOCAL)
        pattern_energy = self._compute_level_energy(assignment, ConstraintLevel.PATTERN)
        global_energy = self._compute_level_energy(assignment, ConstraintLevel.GLOBAL)
        
        total_energy = local_energy + pattern_energy + global_energy
        
        components = EnergyComponents(
            local_energy=local_energy,
            pattern_energy=pattern_energy,
            global_energy=global_energy,
            total_energy=total_energy
        )
        
        # Guardar en cache
        if use_cache:
            self._energy_cache[cache_key] = components
            
        return components
    
    def _compute_level_energy(self, 
                             assignment: Dict[str, Any], 
                             level: ConstraintLevel) -> float:
        """
        Calcula la energía de un nivel específico.
        
        Args:
            assignment: Asignación parcial
            level: Nivel de la jerarquía
            
        Returns:
            Energía del nivel
        """
        constraints = self.hierarchy.get_constraints_at_level(level)
        level_weight = self.level_weights[level]
        
        energy = 0.0
        for constraint in constraints:
            satisfied, violation = constraint.evaluate(assignment)
            # Energía = peso_nivel * peso_restricción * violación
            energy += level_weight * constraint.weight * violation
            
        return energy
    
    def compute_energy_gradient(self, 
                               assignment: Dict[str, Any], 
                               variable: str,
                               domain: List[Any]) -> Dict[Any, float]:
        """
        Calcula el gradiente de energía para una variable.
        
        Devuelve un diccionario {valor: energía} para cada valor posible
        de la variable, manteniendo el resto de la asignación fija.
        
        Args:
            assignment: Asignación parcial actual
            variable: Variable para la cual calcular el gradiente
            domain: Dominio de valores posibles para la variable
            
        Returns:
            Diccionario {valor: energía}
        """
        gradient = {}
        
        for value in domain:
            # Crear asignación temporal
            temp_assignment = assignment.copy()
            temp_assignment[variable] = value
            
            # Calcular energía
            energy = self.compute_energy(temp_assignment, use_cache=False).total_energy
            gradient[value] = energy
            
        return gradient
    
    def find_energy_minimum(self,
                           assignment: Dict[str, Any],
                           variable: str,
                           domain: List[Any]) -> Tuple[Any, float]:
        """
        Encuentra el valor de mínima energía para una variable.
        
        Args:
            assignment: Asignación parcial actual
            variable: Variable a optimizar
            domain: Dominio de la variable
            
        Returns:
            Tupla (valor_óptimo, energía_mínima)
        """
        gradient = self.compute_energy_gradient(assignment, variable, domain)
        
        if not gradient:
            return (None, float('inf'))
        
        min_value = min(gradient, key=gradient.get)
        min_energy = gradient[min_value]
        
        return (min_value, min_energy)
    
    def find_local_minima(self, 
                         assignment: Dict[str, Any], 
                         unassigned_vars: List[str],
                         domains: Dict[str, List[Any]],
                         max_minima: int = 10) -> List[Tuple[Dict[str, Any], float]]:
        """
        Identifica atractores (mínimos locales) en la vecindad de una asignación.
        
        Explora asignaciones vecinas (difieren en una variable) y devuelve
        aquellas que son mínimos locales.
        
        Args:
            assignment: Asignación parcial actual
            unassigned_vars: Variables no asignadas
            domains: Dominios de las variables
            max_minima: Número máximo de mínimos a devolver
            
        Returns:
            Lista de tuplas (asignación, energía) ordenadas por energía
        """
        current_energy = self.compute_energy(assignment).total_energy
        local_minima = []
        
        for var in unassigned_vars:
            if var not in domains:
                continue
                
            # Encontrar el valor de mínima energía para esta variable
            min_value, min_energy = self.find_energy_minimum(
                assignment, var, domains[var]
            )
            
            # Si es menor que la energía actual, es un atractor
            if min_energy < current_energy:
                new_assignment = assignment.copy()
                new_assignment[var] = min_value
                local_minima.append((new_assignment, min_energy))
        
        # Ordenar por energía y devolver los mejores
        local_minima.sort(key=lambda x: x[1])
        return local_minima[:max_minima]
    
    def compute_energy_delta(self,
                            assignment: Dict[str, Any],
                            variable: str,
                            old_value: Any,
                            new_value: Any) -> float:
        """
        Calcula el cambio de energía al modificar una asignación.
        
        Útil para cálculo incremental de energía.
        
        Args:
            assignment: Asignación actual
            variable: Variable a modificar
            old_value: Valor anterior
            new_value: Valor nuevo
            
        Returns:
            Delta de energía (positivo = aumenta, negativo = disminuye)
        """
        # Energía con valor anterior
        old_assignment = assignment.copy()
        old_assignment[variable] = old_value
        old_energy = self.compute_energy(old_assignment, use_cache=False).total_energy
        
        # Energía con valor nuevo
        new_assignment = assignment.copy()
        new_assignment[variable] = new_value
        new_energy = self.compute_energy(new_assignment, use_cache=False).total_energy
        
        return new_energy - old_energy
    
    def _assignment_to_key(self, assignment: Dict[str, Any]) -> str:
        """
        Convierte una asignación a una clave de cache.
        
        Args:
            assignment: Asignación a convertir
            
        Returns:
            Clave de cache (string)
        """
        # Ordenar items para garantizar consistencia
        items = sorted(assignment.items())
        return str(items)
    
    def clear_cache(self):
        """Limpia el cache de energías."""
        self._energy_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del cache.
        
        Returns:
            Diccionario con estadísticas
        """
        total_accesses = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_accesses if total_accesses > 0 else 0.0
        
        return {
            'cache_size': len(self._energy_cache),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate
        }
    
    def __repr__(self):
        stats = self.get_cache_statistics()
        return (f"EnergyLandscape(cache_size={stats['cache_size']}, "
                f"hit_rate={stats['hit_rate']:.2%})")

