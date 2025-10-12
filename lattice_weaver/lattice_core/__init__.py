"""
Módulo de núcleo lógico (FCA) para LatticeWeaver v4.0.

Este módulo implementa Formal Concept Analysis para extraer la teoría
lógica de problemas CSP.
"""

from .context import FormalContext
from .builder import LatticeBuilder
from .parallel_fca import ParallelFCABuilder

__all__ = ['FormalContext', 'LatticeBuilder', 'ParallelFCABuilder']

