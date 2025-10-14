"""
Módulo de Visualización de LatticeWeaver.

Este módulo proporciona herramientas para visualizar el espacio de búsqueda
y generar reportes interactivos del proceso de resolución de CSPs.

Autor: LatticeWeaver Team
Fecha: 12 de Octubre de 2025
Versión: 1.0
"""

from .search_viz import (
    load_trace,
    plot_search_tree,
    plot_domain_evolution,
    plot_backtrack_heatmap,
    plot_timeline,
    plot_variable_statistics,
    compare_traces,
    export_visualizations,
    generate_report,
    generate_advanced_report
)
from . import api

__all__ = [
    'api',
    'load_trace',
    'plot_search_tree',
    'plot_domain_evolution',
    'plot_backtrack_heatmap',
    'plot_timeline',
    'plot_variable_statistics',
    'compare_traces',
    'export_visualizations',
    'generate_report',
    'generate_advanced_report'
]

