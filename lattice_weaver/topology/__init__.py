"""
Módulo de análisis topológico para LatticeWeaver v4.0.

Este módulo proporciona análisis topológico completo de problemas CSP,
incluyendo construcción de grafos de consistencia, complejos simpliciales
y cálculo de números de Betti.
"""

from .analyzer import TopologyAnalyzer

__all__ = ['TopologyAnalyzer']

from .tda_engine import *
