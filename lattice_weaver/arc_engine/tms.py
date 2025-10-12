from typing import Set, Dict, List, Optional, Any, Tuple, Callable
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
        decision_level: Nivel de decisión en el que se realizó la eliminación
    """
    variable: str
    removed_value: Any
    reason_constraint: str
    supporting_values: Dict[str, Any] = field(default_factory=dict)
    decision_level: int = -1 # Nivel de decisión en el que se realizó la eliminación
    
    def __hash__(self):
        """
        Hash para usar en sets.
        Incluye decision_level para diferenciar justificaciones en diferentes ramas.
        """
        return hash((self.variable, self.removed_value, self.reason_constraint, self.decision_level))
    
    def __eq__(self, other):
        """
        Igualdad para comparación.
        Incluye decision_level para diferenciar justificaciones en diferentes ramas.
        """
        if not isinstance(other, Justification):
            return False
        return (self.variable == other.variable and 
                self.removed_value == other.removed_value and
                self.reason_constraint == other.reason_constraint and
                self.decision_level == other.decision_level)


@dataclass
class Decision:
    """
    Decisión tomada durante la resolución.
    
    Attributes:
        variable: Variable asignada
        value: Valor asignado
        justifications_start_index: Índice en la lista global de justificaciones donde empiezan las de esta decisión.
    """
    variable: str
    value: Any
    justifications_start_index: int


class TruthMaintenanceSystem:
    """
    Sistema de Mantenimiento de Verdad.
    
    Rastrea dependencias y permite retroceso eficiente.
    """
    
    def __init__(self):
        """
        Inicializa el TMS.
        
        `domain_restore_callback`: Función que el TMS llamará para restaurar el dominio de una variable.
                                   Debe aceptar (variable_name: str, value_to_restore: Any).
        """
        self.justifications: List[Justification] = []
        self.decisions: List[Decision] = []
        # self.dependency_graph: Dict[str, Set[Justification]] = {}
        # self.constraint_removals: Dict[str, List[Justification]] = {}
        
        # Nuevo: Historial de eliminaciones por nivel de decisión
        self.removals_by_decision_level: List[List[Justification]] = []
        
        # Callback para restaurar dominios en el ArcEngine
        self.domain_restore_callback: Optional[Callable[[str, Any], None]] = None
        
        logger.debug("TMS inicializado")
    
    def set_domain_restore_callback(self, callback: Callable[[str, Any], None]):
        """
        Establece la función de callback para restaurar dominios.
        """
        self.domain_restore_callback = callback

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
        current_decision_level = len(self.decisions) - 1
        justification = Justification(
            variable=variable,
            removed_value=value,
            reason_constraint=constraint_id,
            supporting_values=supporting_values.copy(),
            decision_level=current_decision_level
        )
        
        self.justifications.append(justification)
        
        if current_decision_level >= 0:
            while len(self.removals_by_decision_level) <= current_decision_level:
                self.removals_by_decision_level.append([])
            self.removals_by_decision_level[current_decision_level].append(justification)
        
        logger.debug(f"Registrado: {variable}={value} eliminado por {constraint_id} en nivel {current_decision_level}")
    
    def record_decision(self, variable: str, value: Any):
        """
        Registra una decisión (asignación).
        
        Args:
            variable: Variable asignada
            value: Valor asignado
        """
        # El índice de inicio de justificaciones para esta decisión es el tamaño actual de la lista global
        justifications_start_index = len(self.justifications)
        decision = Decision(variable=variable, value=value, justifications_start_index=justifications_start_index)
        self.decisions.append(decision)
        
        # Asegurarse de que haya una lista para este nivel de decisión
        self.removals_by_decision_level.append([])
        
        logger.debug(f"Decisión registrada: {variable}={value} en nivel {len(self.decisions) - 1}")
    
    def backtrack_to_decision(self, decision_level: int):
        """
        Retrocede a un nivel de decisión anterior.
        
        Args:
            decision_level: Nivel al que retroceder (0-indexed). Si es -1, limpia todo.
        """
        if decision_level < -1 or decision_level >= len(self.decisions):
            logger.warning(f"Nivel de decisión inválido para retroceso: {decision_level}. Nivel actual: {len(self.decisions) - 1}")
            return
        
        if decision_level == -1:
            self.clear()
            return

        # Restaurar dominios de las justificaciones realizadas DESPUÉS del nivel de decisión
        for i in range(len(self.removals_by_decision_level) - 1, decision_level, -1):
            for justification in self.removals_by_decision_level[i]:
                if self.domain_restore_callback:
                    self.domain_restore_callback(justification.variable, justification.removed_value)
                logger.debug(f"Restaurado {justification.removed_value} a {justification.variable} (nivel {justification.decision_level})")
        
        # Eliminar decisiones posteriores
        self.decisions = self.decisions[:decision_level + 1]
        
        # Eliminar justificaciones posteriores
        # Todas las justificaciones con decision_level > decision_level deben ser eliminadas
        self.justifications = [j for j in self.justifications if j.decision_level <= decision_level]
        
        # Recortar la lista de eliminaciones por nivel de decisión
        self.removals_by_decision_level = self.removals_by_decision_level[:decision_level + 1]

        logger.info(f"Retroceso a nivel de decisión {decision_level}. Decisiones restantes: {len(self.decisions)}")
    
    def get_current_decision_level(self) -> int:
        """
        Retorna el nivel de decisión actual (0-indexed).
        """
        return len(self.decisions) - 1

    def explain_inconsistency(self, variable: str) -> List[Justification]:
        """
        Explica por qué una variable quedó sin valores.
        
        Args:
            variable: Variable inconsistente
        
        Returns:
            Lista de justificaciones que causaron la inconsistencia
        """
        # Filtra las justificaciones activas (no retrocedidas)
        active_justifications = [j for j in self.justifications if j.variable == variable and j.decision_level <= self.get_current_decision_level()]
        
        logger.info(f"Inconsistencia en {variable}:")
        for just in active_justifications:
            logger.info(f"  - Valor {just.removed_value} eliminado por {just.reason_constraint} en nivel {just.decision_level}")
        
        return active_justifications
    
    def suggest_constraint_to_relax(self, variable: str) -> Optional[str]:
        """
        Sugiere qué restricción relajar para resolver inconsistencia.
        
        Estrategia: Restricción que causó más eliminaciones.
        
        Args:
            variable: Variable inconsistente
        
        Returns:
            ID de restricción sugerida o None
        """
        active_justifications = [j for j in self.justifications if j.variable == variable and j.decision_level <= self.get_current_decision_level()]
        
        # Contar eliminaciones por restricción
        constraint_counts: Dict[str, int] = {}
        
        for just in active_justifications:
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
        # Filtrar justificaciones activas asociadas a esta restricción
        active_removals = [j for j in self.justifications if j.reason_constraint == constraint_id and j.decision_level <= self.get_current_decision_level()]
        
        restorable: Dict[str, List[Any]] = {}
        
        for just in active_removals:
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
        # Eliminar de la lista principal de justificaciones
        self.justifications = [j for j in self.justifications if j.reason_constraint != constraint_id]
        
        # Eliminar de las listas por nivel de decisión
        for level_removals in self.removals_by_decision_level:
            level_removals[:] = [j for j in level_removals if j.reason_constraint != constraint_id]
        
        logger.debug(f"Justificaciones de {constraint_id} eliminadas")
    
    def get_conflict_graph(self, variable: str) -> Dict[str, List[str]]:
        """
        Construye un grafo de conflictos para una variable.
        
        Args:
            variable: Variable inconsistente
        
        Returns:
            Grafo de conflictos {restricción: [variables_involucradas]}
        """
        active_justifications = [j for j in self.justifications if j.variable == variable and j.decision_level <= self.get_current_decision_level()]
        
        conflict_graph: Dict[str, List[str]] = {}
        
        for just in active_justifications:
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
            'current_decision_level': self.get_current_decision_level(),
            'removals_by_level': [len(r) for r in self.removals_by_decision_level]
        }
    
    def clear(self):
        """
        Limpia todos los datos del TMS.
        """
        self.justifications.clear()
        self.decisions.clear()
        self.removals_by_decision_level.clear()
        
        logger.debug("TMS limpiado")
    
    def __repr__(self) -> str:
        """
        Representación del TMS.
        """
        stats = self.get_statistics()
        return (f"TMS(justifications={stats['total_justifications']}, "
                f"decisions={stats['total_decisions']}, "
                f"current_level={stats['current_decision_level']})")


def create_tms() -> TruthMaintenanceSystem:
    """
    Crea una instancia del Truth Maintenance System.
    
    Returns:
        Instancia de TruthMaintenanceSystem
    """
    return TruthMaintenanceSystem()

