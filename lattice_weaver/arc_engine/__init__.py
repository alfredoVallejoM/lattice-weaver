import warnings

# Importar los componentes del motor CSP desde su ubicación actual en esta rama
from .core import ArcEngine
from .csp_solver import CSPProblem, CSPSolution, CSPSolver
from .tms import create_tms, TruthMaintenanceSystem

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
]

