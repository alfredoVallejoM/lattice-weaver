# lattice_weaver/core/csp_engine/tms.py

"""
Adaptador de compatibilidad para el Truth Maintenance System (TMS)

Este módulo proporciona una implementación stub del TMS si no existe
un equivalente directo en arc_engine.
"""

from typing import Dict, Set, Any, List, Optional
from collections import defaultdict

class TruthMaintenanceSystem:
    """
    Sistema de Mantenimiento de la Verdad para rastrear dependencias
    entre decisiones en el proceso de resolución de CSP.
    """
    
    def __init__(self):
        """Inicializa el TMS."""
        self.justifications: Dict[str, List[Set[str]]] = defaultdict(list)
        self.beliefs: Set[str] = set()
        self.contradictions: List[Set[str]] = []
    
    def add_justification(self, conclusion: str, premises: Set[str]):
        """
        Añade una justificación: premises -> conclusion
        
        Args:
            conclusion: La conclusión derivada
            premises: El conjunto de premisas que soportan la conclusión
        """
        self.justifications[conclusion].append(premises)
        
        # Si todas las premisas son creídas, la conclusión también lo es
        if premises.issubset(self.beliefs):
            self.beliefs.add(conclusion)
    
    def add_belief(self, belief: str):
        """Añade una creencia al sistema."""
        self.beliefs.add(belief)
        self._propagate()
    
    def remove_belief(self, belief: str):
        """Elimina una creencia y propaga los cambios."""
        if belief in self.beliefs:
            self.beliefs.remove(belief)
            self._retract_dependencies(belief)
    
    def _propagate(self):
        """Propaga las creencias a través de las justificaciones."""
        changed = True
        while changed:
            changed = False
            for conclusion, premise_sets in self.justifications.items():
                if conclusion not in self.beliefs:
                    for premises in premise_sets:
                        if premises.issubset(self.beliefs):
                            self.beliefs.add(conclusion)
                            changed = True
                            break
    
    def _retract_dependencies(self, belief: str):
        """Retrae todas las conclusiones que dependen de una creencia."""
        to_retract = set()
        for conclusion, premise_sets in self.justifications.items():
            if conclusion in self.beliefs:
                # Si todas las justificaciones requieren la creencia eliminada
                all_require_belief = all(belief in premises for premises in premise_sets)
                if all_require_belief:
                    to_retract.add(conclusion)
        
        for conclusion in to_retract:
            self.beliefs.discard(conclusion)
            self._retract_dependencies(conclusion)
    
    def is_believed(self, proposition: str) -> bool:
        """Verifica si una proposición es creída."""
        return proposition in self.beliefs
    
    def get_support(self, conclusion: str) -> List[Set[str]]:
        """Obtiene todas las justificaciones para una conclusión."""
        return self.justifications.get(conclusion, [])
    
    def record_contradiction(self, conflict_set: Set[str]):
        """Registra un conjunto de creencias contradictorias."""
        self.contradictions.append(conflict_set)
    
    def get_contradictions(self) -> List[Set[str]]:
        """Obtiene todos los conjuntos de contradicciones registrados."""
        return self.contradictions
    
    def reset(self):
        """Reinicia el TMS."""
        self.justifications.clear()
        self.beliefs.clear()
        self.contradictions.clear()

def create_tms() -> TruthMaintenanceSystem:
    """
    Función de factoría para crear una instancia de TruthMaintenanceSystem.
    
    Returns:
        Una nueva instancia de TruthMaintenanceSystem
    """
    return TruthMaintenanceSystem()

__all__ = ['TruthMaintenanceSystem', 'create_tms']

