# lattice_weaver/visualization/__init__.py

"""
@i18n:key visualization_package
@i18n:category visualization
@i18n:desc_es Paquete de visualización para LatticeWeaver. Proporciona herramientas para visualizar CSPs, lattices, espacios de búsqueda y estructuras topológicas.
@i18n:desc_en Visualization package for LatticeWeaver. Provides tools to visualize CSPs, lattices, search spaces, and topological structures.
@i18n:desc_fr Package de visualisation pour LatticeWeaver. Fournit des outils pour visualiser les CSP, les lattices, les espaces de recherche et les structures topologiques.
"""

from .core import VisualizationEngine, Theme, create_visualization_engine

__all__ = [
    "VisualizationEngine",
    "Theme",
    "create_visualization_engine"
]

__version__ = "0.1.0"

