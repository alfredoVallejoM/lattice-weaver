"""
Módulo de compatibilidad para arc_engine (DEPRECATED).
Este módulo se mantendrá hasta la versión 6.0 para compatibilidad.
"""
import warnings

# Importar los componentes del nuevo motor CSP
from ..core.csp_engine.graph import ConstraintGraph
from ..core.csp_engine.solver import AdaptiveConsistencyEngine
from ..core.csp_engine.tracing import SearchSpaceTracer, SearchEvent
from ..core.csp_engine.clustering import ClusterDetector
from ..core.csp_engine.topology_utils import TopologyUtils

warnings.warn(
    "El módulo 'lattice_weaver.arc_engine' está DEPRECATED. "
    "Use 'lattice_weaver.core.csp_engine' en su lugar.",
    DeprecationWarning, stacklevel=2
)

# Alias para mantener la compatibilidad
ArcEngine = AdaptiveConsistencyEngine

# Exportar otros componentes que podrían haber sido usados directamente
__all__ = [
    "ArcEngine",
    "ConstraintGraph",
    "SearchSpaceTracer",
    "ClusterDetector",
    "TopologyUtils",
]

