import warnings

# Importar los componentes del motor CSP desde su ubicación actual en esta rama
from .core import ArcEngine
from .csp_solver import CSPProblem, CSPSolution, CSPSolver
from .tms import create_tms, TruthMaintenanceSystem

# Módulos de optimización
from .adaptive_propagation import AdaptivePropagationEngine, PropagationLevel
from .advanced_optimizations import SmartMemoizer, ConstraintCompiler, SpatialIndex, ObjectPool
from .multiprocess_ac3 import MultiprocessAC3
from .parallel_ac3 import ParallelAC3
from .serializable_constraints import SerializableConstraint
from .tms_enhanced import TMSEnhanced
from .topological_parallel import TopologicalParallelAC3

warnings.warn(
    "El módulo 'lattice_weaver.arc_engine' está DEPRECATED. "
    "Use 'lattice_weaver.arc_engine' directamente o sus submódulos.",
    DeprecationWarning, stacklevel=2
)

# Exportar los componentes para facilitar su importación
__all__ = [
    "ArcEngine",
    "CSPProblem",
    "CSPSolution",
    "CSPSolver",
    "create_tms",
    "TruthMaintenanceSystem",
    "AdaptivePropagationEngine",
    "PropagationLevel",
    "SmartMemoizer",
    "ConstraintCompiler",
    "SpatialIndex",
    "ObjectPool",
    "MultiprocessAC3",
    "ParallelAC3",
    "SerializableConstraint",
    "TMSEnhanced",
    "TopologicalParallelAC3",
]

