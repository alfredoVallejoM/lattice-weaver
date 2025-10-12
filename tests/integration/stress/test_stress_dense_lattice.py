"""
Tests de estrés: Retículos Densos

Valida que el sistema pueda manejar retículos conceptuales densos.
"""

import pytest
import signal
from contextlib import contextmanager
from lattice_weaver.lattice_core.parallel_fca import ParallelFCABuilder


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
def test_fca_dense_context(stress_timeout):
    """
    Test: Construir retículo conceptual desde contexto denso 100x100.
    
    Validación: Debe completar en <120s.
    """
    # Crear contexto formal denso
    n_objects = 100
    n_attributes = 100
    
    objects = [f'o{i}' for i in range(n_objects)]
    attributes = [f'a{i}' for i in range(n_attributes)]
    
    # Incidencia: 30% de densidad
    incidence = set()
    for i, obj in enumerate(objects):
        for j, attr in enumerate(attributes):
            if (i + j) % 10 < 3:  # 30% de densidad
                incidence.add((obj, attr))
    
    builder = ParallelFCABuilder()
    
    try:
        with time_limit(stress_timeout):
            concepts = builder.build_concepts(objects, attributes, incidence)
            
            assert len(concepts) > 0, "Debe generar conceptos"
            
            print(f"✅ FCA Contexto Denso (100x100): Completado")
            print(f"   Objetos: {n_objects}")
            print(f"   Atributos: {n_attributes}")
            print(f"   Incidencia: {len(incidence)} ({len(incidence)/(n_objects*n_attributes)*100:.1f}%)")
            print(f"   Conceptos generados: {len(concepts)}")
    
    except TimeoutException:
        pytest.fail(f"Timeout de {stress_timeout}s alcanzado")

