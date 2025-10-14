"""
Módulo de Benchmarking y Experimentación de LatticeWeaver.

Este módulo proporciona herramientas para ejecutar experimentos masivos,
analizar resultados y comparar diferentes configuraciones del solver.

Autor: LatticeWeaver Team
Fecha: 12 de Octubre de 2025
Versión: 1.0
"""

# Comentado temporalmente para evitar dependencias problemáticas
# from .runner import ExperimentRunner, ExperimentConfig
# from .analysis import (
#     analyze_results,
#     generate_comparison_report,
#     compute_statistics_with_confidence,
#     detect_outliers,
#     generate_detailed_report,
#     export_results_to_csv,
#     export_summary_to_markdown
# )

from .orchestrator import (
    Orchestrator,
    BenchmarkMetrics,
    CompilationStrategy,
    NoCompilationStrategy,
    FixedLevelStrategy
)

from .generators import (
    generate_nqueens,
    generate_sudoku,
    generate_graph_coloring,
    generate_job_shop_scheduling,
    generate_simple_csp
)

__all__ = [
    # 'ExperimentRunner',
    # 'ExperimentConfig',
    # 'analyze_results',
    # 'generate_comparison_report',
    # 'compute_statistics_with_confidence',
    # 'detect_outliers',
    # 'generate_detailed_report',
    # 'export_results_to_csv',
    # 'export_summary_to_markdown',
    'Orchestrator',
    'BenchmarkMetrics',
    'CompilationStrategy',
    'NoCompilationStrategy',
    'FixedLevelStrategy',
    'generate_nqueens',
    'generate_sudoku',
    'generate_graph_coloring',
    'generate_job_shop_scheduling',
    'generate_simple_csp'
]

