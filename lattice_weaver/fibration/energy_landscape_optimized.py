"""
Energy Landscape Module - Optimized Version

Versión optimizada del paisaje de energía con:
1. Cálculo incremental de energía
2. Evaluación solo de restricciones relevantes
3. Cache habilitado por defecto
4. Gradiente optimizado

Parte de la implementación del Flujo de Fibración (Propuesta 2) - Fase 1 Optimizada.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import OrderedDict
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


class EnergyLandscapeOptimized:
    """
    Versión optimizada del paisaje de energía del espacio de búsqueda.
    
    Optimizaciones principales:
    1. Cálculo incremental de energía (solo evalúa restricciones afectadas)
    2. Cache habilitado por defecto
    3. Filtrado de restricciones irrelevantes
    4. Gradiente optimizado con evaluación lazy
    """
    
    def __init__(self, hierarchy: ConstraintHierarchy):
        """
        Inicializa el paisaje de energía optimizado.
        
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
        self._energy_cache: OrderedDict[str, EnergyComponents] = OrderedDict()
        self.cache_max_size = 100000  # Tamaño máximo del caché LRU
        
        # Índice: variable -> restricciones que la involucran
        self._var_to_constraints: Dict[str, List[Constraint]] = {}
        self._build_constraint_index()
        
        # Estadísticas
        self.cache_hits = 0
        self.cache_misses = 0
        self.incremental_calculations = 0
        self.full_calculations = 0
        
    def _build_constraint_index(self):
        """Construye índice de variables a restricciones para acceso rápido."""
        self._var_to_constraints.clear()
        
        for level in ConstraintLevel:
            for constraint in self.hierarchy.get_constraints_at_level(level):
                for var in constraint.variables:
                    if var not in self._var_to_constraints:
                        self._var_to_constraints[var] = []
                    self._var_to_constraints[var].append(constraint)
    
    def compute_energy(self, 
                      assignment: Dict[str, Any], 
                      use_cache: bool = True) -> EnergyComponents:
        """
        Calcula la energía de una asignación parcial.
        
        OPTIMIZACIÓN: Cache habilitado por defecto.
        
        Args:
            assignment: Diccionario {variable: valor}
            use_cache: Si usar caché de energías (por defecto True)
            
        Returns:
            EnergyComponents con desglose de energía
        """
        cache_key = self._assignment_to_key(assignment)
        
        # Optimización: Si la asignación es parcial, la clave puede ser una tupla de (variable, valor)
        # para evitar la serialización completa de asignaciones grandes.
        # Sin embargo, para la coherencia del caché, mantendremos la serialización completa por ahora.
        # La optimización de la clave se puede explorar más a fondo si la serialización se convierte en un cuello de botella.
        
        if use_cache and cache_key in self._energy_cache:
            self.cache_hits += 1
            return self._energy_cache[cache_key]
        
        self.cache_misses += 1
        self.full_calculations += 1
        
        # Calcular energía por nivel (solo restricciones relevantes)
        local_energy = self._compute_level_energy_optimized(assignment, ConstraintLevel.LOCAL)
        pattern_energy = self._compute_level_energy_optimized(assignment, ConstraintLevel.PATTERN)
        global_energy = self._compute_level_energy_optimized(assignment, ConstraintLevel.GLOBAL)
        
        total_energy = local_energy + pattern_energy + global_energy
        
        components = EnergyComponents(
            local_energy=local_energy,
            pattern_energy=pattern_energy,
            global_energy=global_energy,
            total_energy=total_energy
        )
        
        if use_cache:
            self._energy_cache[cache_key] = components
            
        return components
    
    def compute_energy_incremental(self,
                                   base_assignment: Dict[str, Any],
                                   base_energy: EnergyComponents,
                                   new_var: str,
                                   new_value: Any) -> EnergyComponents:
        """
        Calcula energía de forma incremental.
        
        OPTIMIZACIÓN CRÍTICA: Solo evalúa restricciones que involucran new_var.
        
        Args:
            base_assignment: Asignación base
            base_energy: Energía de la asignación base
            new_var: Variable a añadir
            new_value: Valor de la nueva variable
            
        Returns:
            Nueva energía
        """
        self.incremental_calculations += 1
        
        # Crear nueva asignación
        new_assignment = base_assignment.copy()
        new_assignment[new_var] = new_value
        
        # Obtener restricciones que involucran new_var
        if new_var not in self._var_to_constraints:
            # No hay restricciones que involucren esta variable
            return base_energy
        
        affected_constraints = self._var_to_constraints[new_var]
        
        # Calcular delta de energía por nivel
        delta_local = 0.0
        delta_pattern = 0.0
        delta_global = 0.0
        
        for constraint in affected_constraints:
            # Evaluar restricción en asignación base (si es posible)
            old_satisfied, old_violation = constraint.evaluate(base_assignment)
            
            # Evaluar restricción en nueva asignación
            new_satisfied, new_violation = constraint.evaluate(new_assignment)
            
            # Calcular delta
            delta = (new_violation - old_violation) * constraint.weight
            
            # Aplicar peso de nivel
            level_weight = self.level_weights[constraint.level]
            delta *= level_weight
            
            # Acumular por nivel
            if constraint.level == ConstraintLevel.LOCAL:
                delta_local += delta
            elif constraint.level == ConstraintLevel.PATTERN:
                delta_pattern += delta
            elif constraint.level == ConstraintLevel.GLOBAL:
                delta_global += delta
        
        # Nueva energía = base + delta
        new_energy = EnergyComponents(
            local_energy=base_energy.local_energy + delta_local,
            pattern_energy=base_energy.pattern_energy + delta_pattern,
            global_energy=base_energy.global_energy + delta_global,
            total_energy=base_energy.total_energy + delta_local + delta_pattern + delta_global
        )
        
        # Guardar en cache
        cache_key = self._assignment_to_key(new_assignment)
        self._energy_cache[cache_key] = new_energy
        
        return new_energy
    
    def _compute_level_energy_optimized(self, 
                                       assignment: Dict[str, Any], 
                                       level: ConstraintLevel) -> float:
        """
        Calcula la energía de un nivel específico (versión optimizada).
        
        OPTIMIZACIÓN: Solo evalúa restricciones que tienen al menos una variable asignada.
        
        Args:
            assignment: Asignación parcial
            level: Nivel de la jerarquía
            
        Returns:
            Energía del nivel
        """
        constraints = self.hierarchy.get_constraints_at_level(level)
        
        # OPTIMIZACIÓN: Early exit si no hay restricciones en este nivel
        if not constraints:
            return 0.0
        
        level_weight = self.level_weights[level]
        energy = 0.0
        
        for constraint in constraints:
            # OPTIMIZACIÓN: Solo evaluar si alguna variable está asignada
            relevant = any(var in assignment for var in constraint.variables)
            if not relevant:
                continue
            
            satisfied, violation = constraint.evaluate(assignment)
            energy += level_weight * constraint.weight * violation
            
        return energy
    
    def compute_energy_gradient_optimized(self, 
                                         assignment: Dict[str, Any],
                                         base_energy: EnergyComponents,
                                         variable: str,
                                         domain: List[Any]) -> Dict[Any, float]:
        """
        Calcula el gradiente de energía de forma optimizada.
        
        OPTIMIZACIÓN CRÍTICA: Usa cálculo incremental en lugar de recalcular todo.
        
        Args:
            assignment: Asignación parcial actual
            base_energy: Energía de la asignación base
            variable: Variable para la cual calcular el gradiente
            domain: Dominio de valores posibles para la variable
            
        Returns:
            Diccionario {valor: energía}
        """
        gradient = {}
        
        for value in domain:
            # OPTIMIZACIÓN: Cálculo incremental
            energy_components = self.compute_energy_incremental(
                assignment, base_energy, variable, value
            )
            gradient[value] = energy_components.total_energy
            
        return gradient
    
    def compute_energy_gradient(self, 
                               assignment: Dict[str, Any], 
                               variable: str,
                               domain: List[Any]) -> Dict[Any, float]:
        """
        Calcula el gradiente de energía (interfaz compatible).
        
        Calcula primero la energía base y luego usa el método optimizado.
        
        Args:
            assignment: Asignación parcial actual
            variable: Variable para la cual calcular el gradiente
            domain: Dominio de valores posibles para la variable
            
        Returns:
            Diccionario {valor: energía}
        """
        # Calcular energía base
        base_energy = self.compute_energy(assignment, use_cache=True)
        
        # Usar método optimizado
        return self.compute_energy_gradient_optimized(
            assignment, base_energy, variable, domain
        )
    
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
                
            min_value, min_energy = self.find_energy_minimum(
                assignment, var, domains[var]
            )
            
            if min_energy < current_energy:
                new_assignment = assignment.copy()
                new_assignment[var] = min_value
                local_minima.append((new_assignment, min_energy))
        
        local_minima.sort(key=lambda x: x[1])
        return local_minima[:max_minima]
    
    def compute_energy_delta(self,
                            assignment: Dict[str, Any],
                            variable: str,
                            old_value: Any,
                            new_value: Any) -> float:
        """
        Calcula el cambio de energía al modificar una asignación.
        
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
        old_energy = self.compute_energy(old_assignment, use_cache=True).total_energy
        
        # Energía con valor nuevo
        new_assignment = assignment.copy()
        new_assignment[variable] = new_value
        new_energy = self.compute_energy(new_assignment, use_cache=True).total_energy
        
        return new_energy - old_energy
    
    def _assignment_to_key(self, assignment: Dict[str, Any]) -> str:
        """Convierte una asignación a una clave de cache."""
        items = sorted(assignment.items())
        return str(items)
    
    def clear_cache(self):
        """Limpia el cache de energías."""
        self._energy_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        self.incremental_calculations = 0
        self.full_calculations = 0
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas del cache y optimizaciones."""
        total_accesses = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_accesses if total_accesses > 0 else 0.0
        
        total_calculations = self.incremental_calculations + self.full_calculations
        incremental_rate = self.incremental_calculations / total_calculations if total_calculations > 0 else 0.0
        
        return {
            'cache_size': len(self._energy_cache),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'incremental_calculations': self.incremental_calculations,
            'full_calculations': self.full_calculations,
            'incremental_rate': incremental_rate
        }
    
    def __repr__(self):
        stats = self.get_cache_statistics()
        return (f"EnergyLandscapeOptimized(cache_size={stats['cache_size']}, "
                f"hit_rate={stats['hit_rate']:.2%}, "
                f"incremental_rate={stats['incremental_rate']:.2%})")

