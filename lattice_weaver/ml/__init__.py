"""
LatticeWeaver ML Module

Este módulo implementa aceleración mediante Machine Learning para LatticeWeaver.

Componentes:
- adapters: Capas de adaptación (feature extraction, logging, decoding, augmentation)
- mini_nets: Suite de 120 mini-modelos especializados
- training: Infraestructura de entrenamiento
- optimization: Optimizaciones (ONNX, cuantización, pruning)
- datasets: Generación y gestión de datasets
- benchmarks: Benchmarks y validación

Autor: LatticeWeaver Development Team
Fecha: 14 de Octubre de 2025
Versión: 1.0 (Consolidado)
"""

__version__ = "1.0.0"

# Importar componentes principales (lazy import para evitar dependencias circulares)
def get_feature_extractors():
    from .adapters import feature_extractors
    return feature_extractors

def get_data_augmentation():
    from .adapters import data_augmentation
    return data_augmentation

def get_trainer():
    from .training import trainer
    return trainer

__all__ = [
    "get_feature_extractors",
    "get_data_augmentation",
    "get_trainer",
]

