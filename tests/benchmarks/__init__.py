"""
Framework de benchmarking para LatticeWeaver.

Este paquete proporciona herramientas para benchmarking comparativo
con algoritmos del estado del arte.
"""
from .problems import (
    BenchmarkProblem,
    create_nqueens,
    create_sudoku_4x4,
    create_graph_coloring,
    create_map_coloring,
    create_scheduling_problem,
    STANDARD_SUITE,
    get_problem_by_name,
    get_problems_by_difficulty,
    get_problems_by_category,
)

from .algorithms import (
    CSPSolver,
    BacktrackingSolver,
    ForwardCheckingSolver,
    AC3Solver,
    SolutionStats,
    ALGORITHMS,
    get_solver,
)

from .runner import (
    BenchmarkResult,
    AggregatedResult,
    BenchmarkRunner,
)

from .factory import (
    ProblemFactory,
    ProblemGenerator,
    ProblemConfig,
    ProblemType,
    quick_problem,
    batch_problems,
    scalability_suite,
    get_quick_suite,
    get_stress_suite,
    get_nqueens_suite,
    get_comparison_suite,
)

__all__ = [
    # Problems
    'BenchmarkProblem',
    'create_nqueens',
    'create_sudoku_4x4',
    'create_graph_coloring',
    'create_map_coloring',
    'create_scheduling_problem',
    'STANDARD_SUITE',
    'get_problem_by_name',
    'get_problems_by_difficulty',
    'get_problems_by_category',
    
    # Algorithms
    'CSPSolver',
    'BacktrackingSolver',
    'ForwardCheckingSolver',
    'AC3Solver',
    'SolutionStats',
    'ALGORITHMS',
    'get_solver',
    
    # Runner
    'BenchmarkResult',
    'AggregatedResult',
    'BenchmarkRunner',
    
    # Factory
    'ProblemFactory',
    'ProblemGenerator',
    'ProblemConfig',
    'ProblemType',
    'quick_problem',
    'batch_problems',
    'scalability_suite',
    'get_quick_suite',
    'get_stress_suite',
    'get_nqueens_suite',
    'get_comparison_suite',
]

