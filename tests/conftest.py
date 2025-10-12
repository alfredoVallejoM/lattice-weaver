"""
Configuración global de pytest y fixtures compartidas.

Este módulo proporciona fixtures compartidas para todos los tests
y configura el entorno de testing.
"""
import pytest
import sys
from pathlib import Path

# Añadir el directorio raíz al path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

@pytest.fixture(scope="session")
def project_root():
    """Retorna el directorio raíz del proyecto."""
    return ROOT_DIR

@pytest.fixture(scope="session")
def test_data_dir(project_root):
    """Retorna el directorio de datos de test."""
    data_dir = project_root / "tests" / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir

@pytest.fixture
def sample_csp_problem():
    """
    Fixture: Problema CSP simple para tests.
    
    Retorna un problema CSP básico con 2 variables y una restricción de desigualdad.
    """
    from lattice_weaver.arc_engine.core import CSPProblem, Variable, Constraint
    
    problem = CSPProblem()
    problem.add_variable(Variable("A", {1, 2, 3}))
    problem.add_variable(Variable("B", {1, 2, 3}))
    problem.add_constraint(Constraint(["A", "B"], lambda a, b: a != b))
    
    return problem

@pytest.fixture
def sample_fca_context():
    """
    Fixture: Contexto formal simple para tests.
    
    Retorna un contexto formal con 3 objetos y 3 atributos.
    Representa una relación binaria típica para análisis FCA.
    """
    objects = {"o1", "o2", "o3"}
    attributes = {"a1", "a2", "a3"}
    relation = {
        ("o1", "a1"), ("o1", "a2"),
        ("o2", "a2"), ("o2", "a3"),
        ("o3", "a1"), ("o3", "a3")
    }
    return objects, attributes, relation

@pytest.fixture
def sample_point_cloud():
    """
    Fixture: Nube de puntos simple para TDA.
    
    Retorna 20 puntos distribuidos en un círculo en 2D.
    Útil para tests de análisis topológico de datos.
    """
    import numpy as np
    # Círculo en 2D
    theta = np.linspace(0, 2*np.pi, 20, endpoint=False)
    points = np.column_stack([np.cos(theta), np.sin(theta)])
    return points

# Markers personalizados
def pytest_configure(config):
    """
    Configurar markers personalizados para pytest.
    
    Define markers que se pueden usar para categorizar y filtrar tests.
    """
    config.addinivalue_line(
        "markers", "integration: marca tests de integración entre módulos"
    )
    config.addinivalue_line(
        "markers", "slow: marca tests lentos (>1s de ejecución)"
    )
    config.addinivalue_line(
        "markers", "stress: marca tests de estrés con problemas grandes"
    )
    config.addinivalue_line(
        "markers", "regression: marca tests de regresión con salidas conocidas"
    )
    config.addinivalue_line(
        "markers", "property: marca property-based tests"
    )

