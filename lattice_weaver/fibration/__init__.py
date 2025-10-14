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
- CoherenceSolver: Solver integrado

Autor: Manus AI (basado en TFM de Alfredo Vallejo Martín)
Fecha: Octubre 2025
Versión: 1.0 (Fase 1 - Fundamentos)
"""

from .constraint_hierarchy import (
    ConstraintHierarchy,
    Constraint,
    ConstraintLevel,
    Hardness
)

from .energy_landscape import (
    EnergyLandscape,
    EnergyComponents
)

from .energy_landscape_optimized import (
    EnergyLandscapeOptimized
)

from .coherence_solver_optimized import (
    CoherenceSolverOptimized
)

from .optimization_solver import (
    OptimizationSolver
)

from .simple_optimization_solver import (
    SimpleOptimizationSolver
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

__all__ = [
    # Constraint Hierarchy
    'ConstraintHierarchy',
    'Constraint',
    'ConstraintLevel',
    'Hardness',
    
    # Energy Landscape
    'EnergyLandscape',
    'EnergyComponents',
    
    # Optimized Components
    'EnergyLandscapeOptimized',
    'CoherenceSolverOptimized',
    
    # Optimization
    'OptimizationSolver',
    'SimpleOptimizationSolver',

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

__version__ = '1.0.3-phase2-hacification-modulation'
