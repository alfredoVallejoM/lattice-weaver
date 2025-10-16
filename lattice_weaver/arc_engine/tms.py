from typing import Callable, Any, Dict, List, Tuple, Optional
from collections import defaultdict
import time
import logging

logger = logging.getLogger(__name__)

class TruthMaintenanceSystem:
    def __init__(self):
        self.removed_values: Dict[str, List[Tuple[Any, str, Dict[str, Any]]]] = defaultdict(list)
        self.decision_stack: List[Tuple[str, Any]] = []
        self.domain_restore_callback: Optional[Callable[[str, Any], None]] = None

    def set_domain_restore_callback(self, callback: Callable[[str, Any], None]):
        self.domain_restore_callback = callback

    def record_removal(self, variable: str, value: Any, constraint_id: str, supporting_values: Dict[str, Any]):
        self.removed_values[variable].append((value, constraint_id, supporting_values))
        logger.debug(f"[TMS] Registrada eliminación: {variable}={value} por {constraint_id}")

    def record_decision(self, variable: str, value: Any):
        self.decision_stack.append((variable, value))
        logger.debug(f"[TMS] Registrada decisión: {variable}={value}")

    def backtrack_to_decision(self, decision_level: int):
        # Simplified backtracking for now
        # In a full TMS, this would involve restoring values based on justifications
        logger.debug(f"[TMS] Retrocediendo a nivel de decisión {decision_level}")
        while len(self.decision_stack) > decision_level:
            self.decision_stack.pop()

    def get_current_decision_level(self) -> int:
        return len(self.decision_stack)

    def explain_inconsistency(self, variable: str) -> List[str]:
        # Simplified explanation
        return [f"Variable {variable} se quedó sin valores debido a eliminaciones: {self.removed_values.get(variable, [])}"]

    def suggest_constraint_to_relax(self, variable: str) -> Optional[str]:
        # Simplified suggestion
        if self.removed_values.get(variable):
            return self.removed_values[variable][-1][1] # Last constraint that caused removal
        return None

    def get_restorable_values(self, constraint_id: str) -> Dict[str, List[Any]]:
        restorable: Dict[str, List[Any]] = defaultdict(list)
        for var, removals in self.removed_values.items():
            for value, cid, _ in removals:
                if cid == constraint_id:
                    restorable[var].append(value)
        return restorable

    def remove_constraint_justifications(self, constraint_id: str):
        # Simplified: remove all justifications related to this constraint
        for var in list(self.removed_values.keys()):
            self.removed_values[var] = [(v, c, s) for v, c, s in self.removed_values[var] if c != constraint_id]
            if not self.removed_values[var]:
                del self.removed_values[var]

    def clear(self):
        self.removed_values.clear()
        self.decision_stack.clear()

def create_tms() -> TruthMaintenanceSystem:
    return TruthMaintenanceSystem()

