"""
Fibration Flow Module

Este módulo implementa el Flujo de Fibración como mecanismo de coherencia multinivel
para LatticeWeaver. Basado en el TFM "Una Arquitectura Cognitiva inspirada en la
Teoría de Haces y la Lógica del Devenir".

Componentes principales:
- ConstraintHierarchy: Organización multinivel de restricciones
- EnergyLandscape: Paisaje de energía del espacio de búsqueda
- HacificationEngine: Motor de hacificación (binding multinivel)
- LandscapeModulator: Modulación dinámica del paisaje
- FibrationSearchSolver: Solver integrado

Autor: Manus AI (basado en TFM de Alfredo Vallejo Martín)
Fecha: 14 de Octubre de 2025
Versión: 2.0 (Consolidado)
"""

from .constraint_hierarchy import (
    ConstraintHierarchy,
    Constraint,
    ConstraintLevel,
    Hardness
)
from .constraint_hierarchy_api import ConstraintHierarchyAPI

from .energy_landscape_optimized import (
    EnergyLandscapeOptimized
)
from .energy_landscape_api import EnergyLandscapeAPI

from .coherence_solver_optimized import (
    CoherenceSolverOptimized
)

from .optimization_solver import (
    OptimizationSolver
)

from .hacification_engine import (
    HacificationEngine,
    HacificationResult
)

from .landscape_modulator import (
    LandscapeModulator,
    ModulationStrategy,
    FocusOnLocalStrategy,
    FocusOnGlobalStrategy,
    AdaptiveStrategy
)

from .fibration_search_solver import (
    FibrationSearchSolver
)
from .fibration_search_solver_api import FibrationSearchSolverAPI

from .hill_climbing_solver import (
    HillClimbingFibrationSolver
)

__all__ = [
    # Constraint Hierarchy
    'ConstraintHierarchy',
    'Constraint',
    'ConstraintLevel',
    'Hardness',
    'ConstraintHierarchyAPI',
    
    # Energy Landscape
    'EnergyLandscapeOptimized',
    'EnergyLandscapeAPI',
    
    # Optimized Components
    'CoherenceSolverOptimized',
    
    # Optimization
    'OptimizationSolver',
    'FibrationSearchSolver',
    'FibrationSearchSolverAPI',
    'HillClimbingFibrationSolver',

    # Hacification
    'HacificationEngine',
    'HacificationResult',

    # Landscape Modulation
    'LandscapeModulator',
    'ModulationStrategy',
    'FocusOnLocalStrategy',
    'FocusOnGlobalStrategy',
    'AdaptiveStrategy',
]

__version__ = '2.0.0'

