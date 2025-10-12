"""
Módulo de utilidades para LatticeWeaver v4.0.

Este módulo contiene clases auxiliares para gestión de estados,
persistencia y recolección de métricas.
"""

from .state_manager import StateManager, CanonicalState, BOTTOM_STATE_ID, TOP_STATE_ID
from .persistence import PersistenceManager, CheckpointManager
from .metrics import MetricsCollector, BenchmarkComparison, PhaseMetrics

__all__ = [
    'StateManager',
    'CanonicalState',
    'BOTTOM_STATE_ID',
    'TOP_STATE_ID',
    'PersistenceManager',
    'CheckpointManager',
    'MetricsCollector',
    'BenchmarkComparison',
    'PhaseMetrics'
]

