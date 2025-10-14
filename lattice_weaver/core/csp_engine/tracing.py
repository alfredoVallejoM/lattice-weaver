# lattice_weaver/core/csp_engine/tracing.py

"""
Adaptador de compatibilidad para el módulo tracing

Este módulo proporciona funcionalidades de trazado para el proceso de resolución de CSP.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class TraceEvent:
    """Representa un evento en el proceso de resolución."""
    timestamp: datetime = field(default_factory=datetime.now)
    event_type: str = ""
    variable: Optional[str] = None
    value: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self):
        return f"[{self.timestamp}] {self.event_type}: {self.variable}={self.value}"

class ExecutionTracer:
    """
    Rastrea la ejecución del algoritmo de resolución de CSP.
    """
    
    def __init__(self, enabled: bool = True):
        """
        Inicializa el trazador.
        
        Args:
            enabled: Si el trazado está habilitado
        """
        self.enabled = enabled
        self.events: List[TraceEvent] = []
        self.statistics: Dict[str, int] = {
            'assignments': 0,
            'backtracks': 0,
            'constraint_checks': 0,
            'propagations': 0
        }
    
    def record_assignment(self, variable: str, value: Any, **metadata):
        """Registra una asignación de variable."""
        if self.enabled:
            event = TraceEvent(
                event_type='assignment',
                variable=variable,
                value=value,
                metadata=metadata
            )
            self.events.append(event)
            self.statistics['assignments'] += 1
    
    def record_backtrack(self, variable: str, **metadata):
        """Registra un retroceso (backtrack)."""
        if self.enabled:
            event = TraceEvent(
                event_type='backtrack',
                variable=variable,
                metadata=metadata
            )
            self.events.append(event)
            self.statistics['backtracks'] += 1
    
    def record_constraint_check(self, constraint: Any, result: bool, **metadata):
        """Registra una verificación de restricción."""
        if self.enabled:
            event = TraceEvent(
                event_type='constraint_check',
                metadata={'constraint': str(constraint), 'result': result, **metadata}
            )
            self.events.append(event)
            self.statistics['constraint_checks'] += 1
    
    def record_propagation(self, variable: str, domain_before: Any, domain_after: Any, **metadata):
        """Registra una propagación de restricciones."""
        if self.enabled:
            event = TraceEvent(
                event_type='propagation',
                variable=variable,
                metadata={
                    'domain_before': str(domain_before),
                    'domain_after': str(domain_after),
                    **metadata
                }
            )
            self.events.append(event)
            self.statistics['propagations'] += 1
    
    def get_events(self, event_type: Optional[str] = None) -> List[TraceEvent]:
        """
        Obtiene los eventos registrados, opcionalmente filtrados por tipo.
        
        Args:
            event_type: Tipo de evento a filtrar (None para todos)
            
        Returns:
            Lista de eventos
        """
        if event_type is None:
            return self.events
        return [e for e in self.events if e.event_type == event_type]
    
    def get_statistics(self) -> Dict[str, int]:
        """Obtiene las estadísticas de ejecución."""
        return self.statistics.copy()
    
    def clear(self):
        """Limpia todos los eventos y estadísticas."""
        self.events.clear()
        self.statistics = {
            'assignments': 0,
            'backtracks': 0,
            'constraint_checks': 0,
            'propagations': 0
        }
    
    def enable(self):
        """Habilita el trazado."""
        self.enabled = True
    
    def disable(self):
        """Deshabilita el trazado."""
        self.enabled = False

class SearchSpaceTracer(ExecutionTracer):
    """
    Trazador especializado para el espacio de búsqueda.
    Alias de ExecutionTracer para compatibilidad.
    """
    pass

__all__ = ['TraceEvent', 'ExecutionTracer', 'SearchSpaceTracer']

