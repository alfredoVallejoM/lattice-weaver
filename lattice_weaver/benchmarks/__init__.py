"""
Módulo de Benchmarking y Experimentación de LatticeWeaver.

Este módulo proporciona herramientas para ejecutar experimentos masivos,
analizar resultados y comparar diferentes configuraciones del solver.

Autor: LatticeWeaver Team
Fecha: 12 de Octubre de 2025
Versión: 1.0
"""

from .runner import ExperimentRunner, ExperimentConfig
from .analysis import (
    analyze_results,
    generate_comparison_report,
    compute_statistics_with_confidence,
    detect_outliers,
    generate_detailed_report,
    export_results_to_csv,
    export_summary_to_markdown
)

__all__ = [
    'ExperimentRunner',
    'ExperimentConfig',
    'analyze_results',
    'generate_comparison_report',
    'compute_statistics_with_confidence',
    'detect_outliers',
    'generate_detailed_report',
    'export_results_to_csv',
    'export_summary_to_markdown'
]

