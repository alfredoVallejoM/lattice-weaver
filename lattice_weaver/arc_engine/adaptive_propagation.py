"""
Adaptive Propagation Level - Ajuste Dinámico de Nivel de Propagación

Sistema que ajusta el nivel de propagación dinámicamente según efectividad.

Niveles de propagación:
- NONE: Sin propagación
- FC (Forward Checking): Solo variables futuras
- AC3 (Arc Consistency 3): Consistencia de arcos completa
- PC (Path Consistency): Más fuerte que AC-3

Autor: Agente Autónomo - Lattice Weaver
Fecha: 15 de Octubre, 2025
"""

from enum import Enum
from typing import Dict, Any, List, Set, Optional
from dataclasses import dataclass

from lattice_weaver.arc_engine.core import ArcEngine


class PropagationLevel(Enum):
    """Niveles de propagación."""
    NONE = 0
    FC = 1      # Forward Checking
    AC3 = 2     # Arc Consistency 3
    PC = 3      # Path Consistency


@dataclass
class PropagationStats:
    """Estadísticas de propagación."""
    level: PropagationLevel
    num_propagations: int = 0
    values_eliminated: int = 0
    time_spent: float = 0.0
    effectiveness: float = 0.0  # valores eliminados / tiempo


class AdaptivePropagationEngine:
    """
    Motor de propagación adaptativo.
    
    Ajusta el nivel de propagación dinámicamente según efectividad.
    """
    
    def __init__(
        self,
        arc_engine: ArcEngine,
        initial_level: PropagationLevel = PropagationLevel.AC3,
        adaptation_threshold: int = 50
    ):
        """
        Inicializa Adaptive Propagation Engine.
        
        Args:
            arc_engine: Motor de consistencia de arcos
            initial_level: Nivel inicial de propagación
            adaptation_threshold: Número de propagaciones antes de adaptar
        """
        self.arc_engine = arc_engine
        self.current_level = initial_level
        self.adaptation_threshold = adaptation_threshold
        
        # Estadísticas por nivel
        self.stats: Dict[PropagationLevel, PropagationStats] = {
            level: PropagationStats(level=level) for level in PropagationLevel
        }
        
        # Contador de propagaciones
        self.propagation_count = 0
        
        # Historial de efectividad
        self.effectiveness_history: List[float] = []
    
    def propagate(
        self,
        variable: str,
        value: Any,
        domains: Dict[str, List[Any]],
        constraints: List[Any]
    ) -> tuple[bool, int]:
        """
        Propaga restricciones según nivel actual.
        
        Args:
            variable: Variable asignada
            value: Valor asignado
            domains: Dominios de variables
            constraints: Restricciones a propagar
        
        Returns:
            (consistent, values_eliminated)
        """
        import time
        
        self.propagation_count += 1
        start_time = time.perf_counter()
        
        if self.current_level == PropagationLevel.NONE:
            # Sin propagación
            consistent = True
            eliminated = 0
        
        elif self.current_level == PropagationLevel.FC:
            # Forward Checking: solo variables no asignadas
            consistent, eliminated = self._forward_checking(
                variable, value, domains, constraints
            )
        
        elif self.current_level == PropagationLevel.AC3:
            # AC-3: consistencia de arcos completa
            consistent, eliminated = self._arc_consistency_3(
                variable, value, domains, constraints
            )
        
        elif self.current_level == PropagationLevel.PC:
            # Path Consistency: más fuerte que AC-3
            consistent, eliminated = self._path_consistency(
                variable, value, domains, constraints
            )
        
        else:
            consistent = True
            eliminated = 0
        
        elapsed = time.perf_counter() - start_time
        
        # Actualizar estadísticas
        stats = self.stats[self.current_level]
        stats.num_propagations += 1
        stats.values_eliminated += eliminated
        stats.time_spent += elapsed
        
        if elapsed > 0:
            stats.effectiveness = stats.values_eliminated / stats.time_spent
        
        # Adaptar nivel si es necesario
        if self.propagation_count % self.adaptation_threshold == 0:
            self._adapt_level()
        
        return consistent, eliminated
    
    def _forward_checking(
        self,
        variable: str,
        value: Any,
        domains: Dict[str, List[Any]],
        constraints: List[Any]
    ) -> tuple[bool, int]:
        """
        Forward Checking: elimina valores inconsistentes en variables no asignadas.
        
        Args:
            variable: Variable asignada
            value: Valor asignado
            domains: Dominios de variables
            constraints: Restricciones
        
        Returns:
            (consistent, values_eliminated)
        """
        eliminated = 0
        
        # Para cada restricción que involucra la variable
        for constraint in constraints:
            if variable not in constraint.variables:
                continue
            
            # Para cada otra variable en la restricción
            for other_var in constraint.variables:
                if other_var == variable:
                    continue
                
                # Eliminar valores inconsistentes
                to_remove = []
                for other_value in domains.get(other_var, []):
                    assignment = {variable: value, other_var: other_value}
                    if not constraint.predicate(assignment):
                        to_remove.append(other_value)
                
                # Aplicar eliminaciones
                for val in to_remove:
                    if val in domains.get(other_var, []):
                        domains[other_var].remove(val)
                        eliminated += 1
                
                # Verificar consistencia
                if not domains.get(other_var):
                    return False, eliminated
        
        return True, eliminated
    
    def _arc_consistency_3(
        self,
        variable: str,
        value: Any,
        domains: Dict[str, List[Any]],
        constraints: List[Any]
    ) -> tuple[bool, int]:
        """
        AC-3: Consistencia de arcos completa.
        
        Args:
            variable: Variable asignada
            value: Valor asignado
            domains: Dominios de variables
            constraints: Restricciones
        
        Returns:
            (consistent, values_eliminated)
        """
        # Usar ArcEngine existente
        initial_sizes = {var: len(dom) for var, dom in domains.items()}
        
        # Aplicar AC-3
        consistent = self.arc_engine.enforce_arc_consistency(domains, constraints)
        
        # Calcular valores eliminados
        eliminated = sum(
            initial_sizes.get(var, 0) - len(dom)
            for var, dom in domains.items()
        )
        
        return consistent, eliminated
    
    def _path_consistency(
        self,
        variable: str,
        value: Any,
        domains: Dict[str, List[Any]],
        constraints: List[Any]
    ) -> tuple[bool, int]:
        """
        Path Consistency: Más fuerte que AC-3.
        
        Por ahora, implementado como AC-3 (PC es muy costoso).
        
        Args:
            variable: Variable asignada
            value: Valor asignado
            domains: Dominios de variables
            constraints: Restricciones
        
        Returns:
            (consistent, values_eliminated)
        """
        # Por ahora, usar AC-3
        # TODO: Implementar PC real si es necesario
        return self._arc_consistency_3(variable, value, domains, constraints)
    
    def _adapt_level(self):
        """Adapta el nivel de propagación según efectividad."""
        current_stats = self.stats[self.current_level]
        
        # Calcular efectividad actual
        if current_stats.time_spent > 0:
            current_effectiveness = current_stats.values_eliminated / current_stats.time_spent
        else:
            current_effectiveness = 0
        
        self.effectiveness_history.append(current_effectiveness)
        
        # Decisión de adaptación
        if current_effectiveness < 1.0 and self.current_level != PropagationLevel.NONE:
            # Propagación no es efectiva, bajar nivel
            if self.current_level == PropagationLevel.PC:
                self.current_level = PropagationLevel.AC3
            elif self.current_level == PropagationLevel.AC3:
                self.current_level = PropagationLevel.FC
            elif self.current_level == PropagationLevel.FC:
                self.current_level = PropagationLevel.NONE
        
        elif current_effectiveness > 10.0 and self.current_level != PropagationLevel.PC:
            # Propagación muy efectiva, subir nivel
            if self.current_level == PropagationLevel.NONE:
                self.current_level = PropagationLevel.FC
            elif self.current_level == PropagationLevel.FC:
                self.current_level = PropagationLevel.AC3
            elif self.current_level == PropagationLevel.AC3:
                self.current_level = PropagationLevel.PC
    
    def get_current_level(self) -> PropagationLevel:
        """Retorna nivel actual de propagación."""
        return self.current_level
    
    def set_level(self, level: PropagationLevel):
        """Establece nivel de propagación manualmente."""
        self.current_level = level
    
    def get_stats(self) -> Dict[PropagationLevel, PropagationStats]:
        """Retorna estadísticas de propagación."""
        return self.stats.copy()
    
    def get_effectiveness_history(self) -> List[float]:
        """Retorna historial de efectividad."""
        return self.effectiveness_history.copy()
    
    def reset_stats(self):
        """Resetea estadísticas."""
        self.stats = {
            level: PropagationStats(level=level) for level in PropagationLevel
        }
        self.propagation_count = 0
        self.effectiveness_history.clear()

