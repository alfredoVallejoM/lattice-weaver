"""
Módulo de compatibilidad para arc_engine (DEPRECATED).
Este módulo se mantendrá hasta la versión 6.0 para compatibilidad.
"""
import warnings

warnings.warn(
    "El módulo 'lattice_weaver.arc_engine' está DEPRECATED. "
    "Use 'lattice_weaver.core.csp_engine' en su lugar.",
    DeprecationWarning, stacklevel=2
)

# Importar los componentes directamente desde arc_engine para mantener la funcionalidad
from .core import ArcEngine
from .domains import SetDomain, create_optimal_domain
from .constraints import Constraint, get_relation, register_relation
from .ac31 import revise_with_last_support

# Exportar los componentes que podrían haber sido usados directamente
__all__ = [
    "ArcEngine",
    "SetDomain",
    "create_optimal_domain",
    "Constraint",
    "get_relation",
    "register_relation",
    "revise_with_last_support",
]

