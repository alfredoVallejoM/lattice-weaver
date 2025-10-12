"""
Truth Maintenance System (TMS)

Rastrea dependencias entre decisiones y permite retroceso eficiente.

Autor: LatticeWeaver Team
Fecha: 11 de Octubre de 2025
"""

from typing import Set, Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class Justification:
    """
    Justificación de una decisión.
    
    Registra por qué se eliminó un valor de un dominio.
    
    Attributes:
        variable: Variable afectada
        removed_value: Valor eliminado
        reason_constraint: Restricción que causó la eliminación
        supporting_values: Valores en otras variables que justifican esto
    """
    variable: str
    removed_value: Any
    reason_constraint: str
    supporting_values: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        """Hash para usar en sets."""
        return hash((self.variable, self.removed_value, self.reason_constraint))
    
    def __eq__(self, other):
        """Igualdad para comparación."""
        if not isinstance(other, Justification):
            return False
        return (self.variable == other.variable and 
                self.removed_value == other.removed_value and
                self.reason_constraint == other.reason_constraint)


@dataclass
class Decision:
    """
    Decisión tomada durante la resolución.
    
    Attributes:
        variable: Variable asignada
        value: Valor asignado
        justifications: Justificaciones que dependen de esta decisión
    """
    variable: str
    value: Any
    justifications: List[Justification] = field(default_factory=list)


class TruthMaintenanceSystem:
    """
    Sistema de Mantenimiento de Verdad.
    
    Rastrea dependencias y permite retroceso eficiente.
    """
    
    def __init__(self):
        """Inicializa el TMS."""
        self.justifications: List[Justification] = []
        self.decisions: List[Decision] = []
        self.dependency_graph: Dict[str, Set[Justification]] = {}
        self.constraint_removals: Dict[str, List[Justification]] = {}
        
        logger.debug("TMS inicializado")
    
    def record_removal(self, variable: str, value: Any, 
                      constraint_id: str, 
                      supporting_values: Dict[str, List[Any]]):
        """
        Registra la eliminación de un valor.
        
        Args:
            variable: Variable afectada
            value: Valor eliminado
            constraint_id: Restricción que causó la eliminación
            supporting_values: Valores que justifican la eliminación
        """
        justification = Justification(
            variable=variable,
            removed_value=value,
            reason_constraint=constraint_id,
            supporting_values=supporting_values.copy()
        )
        
        self.justifications.append(justification)
        
        # Actualizar grafo de dependencias
        if variable not in self.dependency_graph:
            self.dependency_graph[variable] = set()
        self.dependency_graph[variable].add(justification)
        
        # Indexar por restricción
        if constraint_id not in self.constraint_removals:
            self.constraint_removals[constraint_id] = []
        self.constraint_removals[constraint_id].append(justification)
        
        logger.debug(f"Registrado: {variable}={value} eliminado por {constraint_id}")
    
    def record_decision(self, variable: str, value: Any):
        """
        Registra una decisión (asignación).
        
        Args:
            variable: Variable asignada
            value: Valor asignado
        """
        decision = Decision(variable=variable, value=value)
        self.decisions.append(decision)
        
        logger.debug(f"Decisión registrada: {variable}={value}")
    
    def explain_inconsistency(self, variable: str) -> List[Justification]:
        """
        Explica por qué una variable quedó sin valores.
        
        Args:
            variable: Variable inconsistente
        
        Returns:
            Lista de justificaciones que causaron la inconsistencia
        """
        if variable not in self.dependency_graph:
            return []
        
        explanations = list(self.dependency_graph[variable])
        
        logger.info(f"Inconsistencia en {variable}:")
        for just in explanations:
            logger.info(f"  - Valor {just.removed_value} eliminado por {just.reason_constraint}")
        
        return explanations
    
    def suggest_constraint_to_relax(self, variable: str) -> Optional[str]:
        """
        Sugiere qué restricción relajar para resolver inconsistencia.
        
        Estrategia: Restricción que causó más eliminaciones.
        
        Args:
            variable: Variable inconsistente
        
        Returns:
            ID de restricción sugerida o None
        """
        if variable not in self.dependency_graph:
            return None
        
        # Contar eliminaciones por restricción
        constraint_counts: Dict[str, int] = {}
        
        for just in self.dependency_graph[variable]:
            cid = just.reason_constraint
            constraint_counts[cid] = constraint_counts.get(cid, 0) + 1
        
        if not constraint_counts:
            return None
        
        # Restricción con más eliminaciones
        suggested = max(constraint_counts.items(), key=lambda x: x[1])
        
        logger.info(f"Sugerencia: relajar restricción {suggested[0]} ({suggested[1]} eliminaciones)")
        
        return suggested[0]
    
    def get_restorable_values(self, constraint_id: str) -> Dict[str, List[Any]]:
        """
        Identifica valores que pueden restaurarse al eliminar una restricción.
        
        Args:
            constraint_id: ID de la restricción a eliminar
        
        Returns:
            Diccionario {variable: [valores_restaurables]}
        """
        if constraint_id not in self.constraint_removals:
            return {}
        
        restorable: Dict[str, List[Any]] = {}
        
        for just in self.constraint_removals[constraint_id]:
            var = just.variable
            val = just.removed_value
            
            if var not in restorable:
                restorable[var] = []
            restorable[var].append(val)
        
        logger.debug(f"Valores restaurables al eliminar {constraint_id}: {restorable}")
        
        return restorable
    
    def remove_constraint_justifications(self, constraint_id: str):
        """
        Elimina todas las justificaciones asociadas a una restricción.
        
        Args:
            constraint_id: ID de la restricción eliminada
        """
        if constraint_id not in self.constraint_removals:
            return
        
        # Obtener justificaciones a eliminar
        to_remove = self.constraint_removals[constraint_id]
        
        # Eliminar de la lista principal
        self.justifications = [j for j in self.justifications if j not in to_remove]
        
        # Eliminar del grafo de dependencias
        for var in self.dependency_graph:
            self.dependency_graph[var] = {
                j for j in self.dependency_graph[var]
                if j.reason_constraint != constraint_id
            }
        
        # Eliminar del índice
        del self.constraint_removals[constraint_id]
        
        logger.debug(f"Justificaciones de {constraint_id} eliminadas")
    
    def backtrack_to_decision(self, decision_level: int):
        """
        Retrocede a un nivel de decisión anterior.
        
        Args:
            decision_level: Nivel al que retroceder (0-indexed)
        """
        if decision_level >= len(self.decisions):
            return
        
        # Eliminar decisiones posteriores
        removed_decisions = self.decisions[decision_level:]
        self.decisions = self.decisions[:decision_level]
        
        # Eliminar justificaciones que dependen de decisiones eliminadas
        removed_vars = {d.variable for d in removed_decisions}
        
        self.justifications = [
            j for j in self.justifications
            if not any(var in removed_vars for var in j.supporting_values)
        ]
        
        # Reconstruir grafo de dependencias
        self.dependency_graph.clear()
        for just in self.justifications:
            if just.variable not in self.dependency_graph:
                self.dependency_graph[just.variable] = set()
            self.dependency_graph[just.variable].add(just)
        
        logger.info(f"Retroceso a nivel de decisión {decision_level}")
    
    def get_conflict_graph(self, variable: str) -> Dict[str, List[str]]:
        """
        Construye un grafo de conflictos para una variable.
        
        Args:
            variable: Variable inconsistente
        
        Returns:
            Grafo de conflictos {restricción: [variables_involucradas]}
        """
        if variable not in self.dependency_graph:
            return {}
        
        conflict_graph: Dict[str, List[str]] = {}
        
        for just in self.dependency_graph[variable]:
            cid = just.reason_constraint
            involved_vars = list(just.supporting_values.keys())
            
            if cid not in conflict_graph:
                conflict_graph[cid] = []
            conflict_graph[cid].extend(involved_vars)
        
        # Eliminar duplicados
        for cid in conflict_graph:
            conflict_graph[cid] = list(set(conflict_graph[cid]))
        
        return conflict_graph
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del TMS.
        
        Returns:
            Diccionario con estadísticas
        """
        return {
            'total_justifications': len(self.justifications),
            'total_decisions': len(self.decisions),
            'variables_with_removals': len(self.dependency_graph),
            'constraints_involved': len(self.constraint_removals),
            'avg_removals_per_variable': (
                len(self.justifications) / len(self.dependency_graph)
                if self.dependency_graph else 0
            )
        }
    
    def clear(self):
        """Limpia todos los datos del TMS."""
        self.justifications.clear()
        self.decisions.clear()
        self.dependency_graph.clear()
        self.constraint_removals.clear()
        
        logger.debug("TMS limpiado")
    
    def __repr__(self) -> str:
        """Representación del TMS."""
        stats = self.get_statistics()
        return (f"TMS(justifications={stats['total_justifications']}, "
                f"decisions={stats['total_decisions']}, "
                f"constraints={stats['constraints_involved']})")


def create_tms() -> TruthMaintenanceSystem:
    """
    Crea una instancia del Truth Maintenance System.
    
    Returns:
        Instancia de TruthMaintenanceSystem
    """
    return TruthMaintenanceSystem()

