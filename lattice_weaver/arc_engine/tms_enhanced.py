"""
Truth Maintenance System (TMS) Mejorado

Extensión del TMS básico con:
1. Conflict-Directed Backjumping (CBJ)
2. No-Good Learning
3. Conflict Analysis
4. Backjump level calculation

Estas mejoras reducen dramáticamente el número de backtracks al:
- Saltar directamente al punto de conflicto (no backtrack secuencial)
- Aprender de conflictos previos para evitar ramas fallidas
- Analizar causas de conflictos para mejor poda

Autor: Agente Autónomo - Lattice Weaver
Fecha: 15 de Octubre, 2025
"""

import logging
from typing import Dict, List, Set, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class Conflict:
    """Representa un conflicto detectado durante la búsqueda."""
    variable: str
    value: Any
    decision_level: int
    conflicting_variables: Set[str] = field(default_factory=set)
    conflicting_constraints: List[str] = field(default_factory=list)
    
    def get_backjump_level(self) -> int:
        """
        Calcula el nivel al que se debe saltar.
        
        Returns:
            Nivel de decisión al que backjumpear (segundo más alto en conflicto)
        """
        if not self.conflicting_variables:
            return max(0, self.decision_level - 1)
        
        # Encontrar el segundo nivel más alto en las variables conflictivas
        # (el más alto es el nivel actual)
        levels = sorted([self.decision_level] + 
                       [lv for lv in self.conflicting_variables if isinstance(lv, int)],
                       reverse=True)
        
        if len(levels) > 1:
            return levels[1]
        return 0


@dataclass
class NoGood:
    """
    Representa un no-good (conjunto de asignaciones que llevan a conflicto).
    
    Un no-good es una cláusula aprendida que dice "estas asignaciones juntas
    son inconsistentes, no las intentes de nuevo".
    """
    assignments: Dict[str, Any]
    reason: str  # Descripción del conflicto
    learned_at_level: int
    
    def matches(self, current_assignment: Dict[str, Any]) -> bool:
        """
        Verifica si el no-good aplica a la asignación actual.
        
        Returns:
            True si todas las asignaciones del no-good están presentes
        """
        for var, value in self.assignments.items():
            if var not in current_assignment or current_assignment[var] != value:
                return False
        return True
    
    def conflicts_with(self, var: str, value: Any, assignment: Dict[str, Any]) -> bool:
        """
        Verifica si asignar var=value con la asignación actual violaría este no-good.
        
        Returns:
            True si la asignación resultante matchearía el no-good
        """
        temp = assignment.copy()
        temp[var] = value
        return self.matches(temp)


class TMSEnhanced:
    """
    Truth Maintenance System mejorado con CBJ y no-good learning.
    
    Características:
    - Conflict-Directed Backjumping (CBJ)
    - No-Good Learning con límite de tamaño
    - Conflict analysis para identificar causas
    - Gestión eficiente de memoria (LRU para no-goods)
    """
    
    def __init__(
        self,
        max_nogoods: int = 10000,
        nogood_max_size: int = 5,
        enable_learning: bool = True,
        enable_backjumping: bool = True
    ):
        """
        Inicializa el TMS mejorado.
        
        Args:
            max_nogoods: Máximo número de no-goods a mantener
            nogood_max_size: Tamaño máximo de un no-good (variables involucradas)
            enable_learning: Activar aprendizaje de no-goods
            enable_backjumping: Activar backjumping
        """
        self.max_nogoods = max_nogoods
        self.nogood_max_size = nogood_max_size
        self.enable_learning = enable_learning
        self.enable_backjumping = enable_backjumping
        
        # Estado de decisiones
        self.decisions: List[Tuple[str, Any, int]] = []  # (var, value, level)
        self.variable_to_level: Dict[str, int] = {}
        self.level_to_variables: Dict[int, List[str]] = defaultdict(list)
        
        # No-goods aprendidos
        self.nogoods: List[NoGood] = []
        self.nogood_hits = 0  # Cuántas veces un no-good previno exploración
        
        # Conflictos detectados
        self.conflicts: List[Conflict] = []
        self.total_conflicts = 0
        self.backjumps = 0
        
        # Dependencias entre variables (para conflict analysis)
        self.dependencies: Dict[str, Set[str]] = defaultdict(set)
        
        logger.info(f"[TMSEnhanced] Inicializado")
        logger.info(f"  Learning: {enable_learning}")
        logger.info(f"  Backjumping: {enable_backjumping}")
        logger.info(f"  Max no-goods: {max_nogoods}")
    
    def record_decision(self, variable: str, value: Any, level: int = None):
        """
        Registra una decisión de asignación.
        
        Args:
            variable: Variable asignada
            value: Valor asignado
            level: Nivel de decisión (auto-incrementa si None)
        """
        if level is None:
            level = len(self.decisions)
        
        self.decisions.append((variable, value, level))
        self.variable_to_level[variable] = level
        self.level_to_variables[level].append(variable)
        
        logger.debug(f"[TMSEnhanced] Decisión: {variable}={value} @ level {level}")
    
    def record_dependency(self, var1: str, var2: str):
        """
        Registra una dependencia entre variables.
        
        Args:
            var1: Variable dependiente
            var2: Variable de la que depende
        """
        self.dependencies[var1].add(var2)
    
    def check_nogood(self, var: str, value: Any, assignment: Dict[str, Any]) -> bool:
        """
        Verifica si asignar var=value violaría algún no-good aprendido.
        
        Args:
            var: Variable a asignar
            value: Valor a asignar
            assignment: Asignación actual
        
        Returns:
            True si la asignación es segura, False si viola un no-good
        """
        if not self.enable_learning:
            return True
        
        for nogood in self.nogoods:
            if nogood.conflicts_with(var, value, assignment):
                self.nogood_hits += 1
                logger.debug(f"[TMSEnhanced] No-good hit: {var}={value} (razón: {nogood.reason})")
                return False
        
        return True
    
    def record_conflict(
        self,
        variable: str,
        value: Any,
        conflicting_vars: Set[str],
        constraint_ids: List[str],
        assignment: Dict[str, Any]
    ) -> Optional[int]:
        """
        Registra un conflicto y retorna el nivel al que backjumpear.
        
        Args:
            variable: Variable que causó el conflicto
            value: Valor que causó el conflicto
            conflicting_vars: Variables involucradas en el conflicto
            constraint_ids: IDs de restricciones violadas
            assignment: Asignación actual
        
        Returns:
            Nivel al que backjumpear, o None para backtrack normal
        """
        current_level = self.variable_to_level.get(variable, len(self.decisions))
        
        # Crear objeto de conflicto
        conflict = Conflict(
            variable=variable,
            value=value,
            decision_level=current_level,
            conflicting_variables=conflicting_vars,
            conflicting_constraints=constraint_ids
        )
        
        self.conflicts.append(conflict)
        self.total_conflicts += 1
        
        # Aprender no-good
        if self.enable_learning:
            self._learn_nogood(conflict, assignment)
        
        # Calcular backjump level
        if self.enable_backjumping:
            backjump_level = self._analyze_conflict(conflict, assignment)
            if backjump_level < current_level - 1:
                self.backjumps += 1
                logger.debug(f"[TMSEnhanced] Backjump: level {current_level} -> {backjump_level}")
                return backjump_level
        
        return None  # Backtrack normal
    
    def _learn_nogood(self, conflict: Conflict, assignment: Dict[str, Any]):
        """
        Aprende un no-good del conflicto.
        
        Args:
            conflict: Conflicto detectado
            assignment: Asignación que causó el conflicto
        """
        # Extraer asignaciones relevantes (solo variables conflictivas)
        relevant_vars = conflict.conflicting_variables | {conflict.variable}
        nogood_assignments = {
            var: assignment[var] 
            for var in relevant_vars 
            if var in assignment
        }
        
        # Limitar tamaño del no-good
        if len(nogood_assignments) > self.nogood_max_size:
            # Mantener solo las variables más recientes
            sorted_vars = sorted(
                nogood_assignments.keys(),
                key=lambda v: self.variable_to_level.get(v, 0),
                reverse=True
            )
            nogood_assignments = {
                var: nogood_assignments[var]
                for var in sorted_vars[:self.nogood_max_size]
            }
        
        # Crear no-good
        nogood = NoGood(
            assignments=nogood_assignments,
            reason=f"Conflict at {conflict.variable}={conflict.value}",
            learned_at_level=conflict.decision_level
        )
        
        # Añadir a lista (con límite LRU)
        self.nogoods.append(nogood)
        if len(self.nogoods) > self.max_nogoods:
            # Eliminar el más antiguo
            self.nogoods.pop(0)
        
        logger.debug(f"[TMSEnhanced] No-good aprendido: {len(nogood_assignments)} vars")
    
    def _analyze_conflict(self, conflict: Conflict, assignment: Dict[str, Any]) -> int:
        """
        Analiza el conflicto para determinar el nivel de backjump.
        
        Args:
            conflict: Conflicto a analizar
            assignment: Asignación actual
        
        Returns:
            Nivel al que backjumpear
        """
        # Encontrar niveles de decisión de variables conflictivas
        conflict_levels = []
        for var in conflict.conflicting_variables:
            if var in self.variable_to_level:
                conflict_levels.append(self.variable_to_level[var])
        
        if not conflict_levels:
            return max(0, conflict.decision_level - 1)
        
        # Backjumpear al segundo nivel más alto
        # (el más alto es el nivel actual)
        conflict_levels.sort(reverse=True)
        if len(conflict_levels) > 1:
            return conflict_levels[1]
        
        return conflict_levels[0] if conflict_levels[0] < conflict.decision_level else 0
    
    def backtrack_to_level(self, target_level: int):
        """
        Deshace decisiones hasta el nivel objetivo.
        
        Args:
            target_level: Nivel al que retroceder
        """
        # Eliminar decisiones posteriores al nivel objetivo
        self.decisions = [
            (var, val, lvl) 
            for var, val, lvl in self.decisions 
            if lvl <= target_level
        ]
        
        # Actualizar mapeos
        self.variable_to_level = {
            var: lvl 
            for var, val, lvl in self.decisions
        }
        
        self.level_to_variables = defaultdict(list)
        for var, val, lvl in self.decisions:
            self.level_to_variables[lvl].append(var)
        
        logger.debug(f"[TMSEnhanced] Backtracked to level {target_level}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estadísticas del TMS."""
        return {
            'total_conflicts': self.total_conflicts,
            'backjumps': self.backjumps,
            'nogoods_learned': len(self.nogoods),
            'nogood_hits': self.nogood_hits,
            'backjump_ratio': self.backjumps / max(self.total_conflicts, 1)
        }
    
    def clear(self):
        """Limpia el estado del TMS."""
        self.decisions.clear()
        self.variable_to_level.clear()
        self.level_to_variables.clear()
        self.conflicts.clear()
        self.dependencies.clear()
        # No limpiar no-goods (se mantienen entre búsquedas)


def create_enhanced_tms(
    max_nogoods: int = 10000,
    nogood_max_size: int = 5,
    enable_learning: bool = True,
    enable_backjumping: bool = True
) -> TMSEnhanced:
    """
    Factory function para crear un TMS mejorado.
    
    Args:
        max_nogoods: Máximo número de no-goods
        nogood_max_size: Tamaño máximo de no-good
        enable_learning: Activar aprendizaje
        enable_backjumping: Activar backjumping
    
    Returns:
        TMS mejorado configurado
    """
    return TMSEnhanced(
        max_nogoods=max_nogoods,
        nogood_max_size=nogood_max_size,
        enable_learning=enable_learning,
        enable_backjumping=enable_backjumping
    )

