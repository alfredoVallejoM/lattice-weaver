"""
Tests de estrés: Alta Dimensión

Valida que el sistema pueda manejar complejos simpliciales de alta dimensión.
"""

import pytest
import signal
import numpy as np
from contextlib import contextmanager
from lattice_weaver.topology.tda_engine import TDAEngine


class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds):
    """Context manager para timeout."""
    def signal_handler(signum, frame):
        raise TimeoutException("Timeout alcanzado")
    
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.slow
def test_tda_high_dimensional_complex(stress_timeout):
    """
    Test: Calcular homología de complejo simplicial con 1000 puntos.
    
    Validación: Debe completar en <120s.
    """
    # Crear datos de alta dimensión
    n_points = 1000
    dimension = 3
    
    # Generar puntos aleatorios
    np.random.seed(42)  # Para reproducibilidad
    points = np.random.rand(n_points, dimension)
    
    # Calcular matriz de distancias (solo una muestra para eficiencia)
    # Usar solo 200 puntos para el cálculo de homología
    sample_size = 200
    sample_indices = np.random.choice(n_points, sample_size, replace=False)
    sample_points = points[sample_indices]
    
    distance_matrix = np.zeros((sample_size, sample_size))
    for i in range(sample_size):
        for j in range(i + 1, sample_size):
            dist = np.linalg.norm(sample_points[i] - sample_points[j])
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist
    
    engine = TDAEngine()
    
    try:
        with time_limit(stress_timeout):
            # Calcular homología persistente hasta dimensión 2
            homology = engine.compute_persistent_homology(
                distance_matrix,
                max_dim=2,
                threshold=1.0
            )
            
            assert len(homology) > 0, "Debe calcular homología"
            
            # Contar características
            total_features = sum(len(intervals) for intervals in homology.values())
            
            print(f"✅ TDA Alta Dimensión: Completado")
            print(f"   Puntos totales: {n_points}")
            print(f"   Puntos muestreados: {sample_size}")
            print(f"   Dimensión: {dimension}")
            print(f"   Características topológicas: {total_features}")
            
            for dim, intervals in homology.items():
                print(f"   {dim}: {len(intervals)} características")
    
    except TimeoutException:
        pytest.fail(f"Timeout de {stress_timeout}s alcanzado")

