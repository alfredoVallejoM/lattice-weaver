# lattice_weaver/arc_engine/__init__.py

from .core import ArcEngine
from .core_extended import ArcEngineExtended
from .parallel_ac3 import ParallelAC3
from .topological_parallel import TopologicalParallelAC3
from .tms import TruthMaintenanceSystem, Justification, Decision, create_tms
from .multiprocess_ac3 import MultiprocessAC3, GroupParallelAC3, create_multiprocess_ac3, create_group_parallel_ac3
from .serializable_constraints import (
    SerializableConstraint,
    LessThanConstraint, LessEqualConstraint,
    GreaterThanConstraint, GreaterEqualConstraint,
    EqualConstraint, NotEqualConstraint,
    AllDifferentPairConstraint, NoAttackQueensConstraint,
    SudokuConstraint,
    LT, LE, GT, GE, EQ, NE, AllDiff
)
from .optimizations import (
    ArcRevisionCache, ArcOrderingStrategy, RedundantArcDetector,
    PerformanceMonitor, OptimizedAC3, create_optimized_ac3
)

__all__ = [
    "ArcEngine", "ArcEngineExtended", "ParallelAC3", "TopologicalParallelAC3",
    "TruthMaintenanceSystem", "Justification", "Decision", "create_tms",
    "MultiprocessAC3", "GroupParallelAC3", "create_multiprocess_ac3", "create_group_parallel_ac3",
    "SerializableConstraint",
    "LessThanConstraint", "LessEqualConstraint",
    "GreaterThanConstraint", "GreaterEqualConstraint",
    "EqualConstraint", "NotEqualConstraint",
    "AllDifferentPairConstraint", "NoAttackQueensConstraint",
    "SudokuConstraint",
    "LT", "LE", "GT", "GE", "EQ", "NE", "AllDiff",
    "ArcRevisionCache", "ArcOrderingStrategy", "RedundantArcDetector",
    "PerformanceMonitor", "OptimizedAC3", "create_optimized_ac3"
]

from .advanced_optimizations import *
