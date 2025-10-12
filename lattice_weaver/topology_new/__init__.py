"""
Módulo de Locales y Frames (Track B)

Este módulo implementa la teoría de Locales y Frames, proporcionando
una base formal para topología sin puntos y lógica modal S4.

Componentes principales:
- Estructuras básicas: PartialOrder, CompleteLattice, Frame, Locale
- Morfismos: FrameMorphism, LocaleMorphism
- Operaciones: ModalOperators, TopologicalOperators
- Análisis: ConnectivityAnalyzer, LocaleAnalyzer
- Construcciones: FrameConstructions, Builders

Autor: LatticeWeaver Team (Track B)
Fecha: 12 de Octubre de 2025
Versión: 1.0
"""

# Estructuras básicas
from .locale import (
    PartialOrder,
    CompleteLattice,
    Frame,
    Locale,
    FrozenDict,
    LatticeBuilder,
    FrameBuilder,
    LocaleBuilder
)

# Morfismos
from .morphisms import (
    FrameMorphism,
    LocaleMorphism,
    FrameConstructions,
    MorphismBuilder
)

# Operaciones
from .operations import (
    ModalOperators,
    TopologicalOperators,
    ConnectivityAnalyzer,
    LocaleAnalyzer
)

__all__ = [
    # Estructuras
    'PartialOrder',
    'CompleteLattice',
    'Frame',
    'Locale',
    'FrozenDict',
    
    # Builders
    'LatticeBuilder',
    'FrameBuilder',
    'LocaleBuilder',
    
    # Morfismos
    'FrameMorphism',
    'LocaleMorphism',
    'FrameConstructions',
    'MorphismBuilder',
    
    # Operaciones
    'ModalOperators',
    'TopologicalOperators',
    'ConnectivityAnalyzer',
    'LocaleAnalyzer',
]

__version__ = '1.0.0'
__author__ = 'LatticeWeaver Team'
