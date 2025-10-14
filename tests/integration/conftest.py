"""
Fixtures específicas para tests de integración - Versión adaptada.

Este módulo proporciona fixtures pre-configuradas adaptadas a la estructura
real del proyecto LatticeWeaver.
"""
import pytest
from lattice_weaver.core.csp_problem import CSP
from lattice_weaver.formal.csp_integration import CSPProblem
from lattice_weaver.lattice_core import ParallelFCABuilder
from lattice_weaver.topology import create_tda_engine
from lattice_weaver.formal.cubical_engine import CubicalEngine
from lattice_weaver.formal.type_checker import TypeChecker

@pytest.fixture(scope="module")
def arc_engine():
    """
    Fixture: ArcEngine configurado.
    
    Retorna una instancia del motor de resolución CSP con configuración por defecto.
    Scope 'module' para reutilizar entre tests del mismo módulo.
    """
    return CSP(variables=set(), domains={}, constraints=[], name="TestCSP")

@pytest.fixture(scope="module")
def fca_builder():
    """
    Fixture: FCA Builder configurado.
    
    Retorna una instancia del constructor de retículos FCA
    con paralelización habilitada.
    """
    return ParallelFCABuilder()

@pytest.fixture(scope="module")
def tda_engine():
    """
    Fixture: TDA Engine configurado.
    
    Retorna una instancia del motor de Análisis Topológico de Datos.
    """
    return create_tda_engine()

@pytest.fixture(scope="module")
def cubical_engine():
    """
    Fixture: Cubical Engine configurado.
    
    Retorna una instancia del motor cúbico de HoTT.
    """
    return CubicalEngine()

@pytest.fixture
def simple_csp_problem():
    """
    Fixture: Problema CSP simple para tests básicos.
    
    Retorna un problema CSP con 3 variables y restricciones de desigualdad.
    """
    variables = ["A", "B", "C"]
    domains = {
        "A": {1, 2, 3},
        "B": {1, 2, 3},
        "C": {1, 2, 3}
    }
    constraints = [
        ("A", "B", lambda a, b: a != b),
        ("B", "C", lambda b, c: b != c),
        ("A", "C", lambda a, c: a != c),
    ]
    
    return CSPProblem(
        variables=variables,
        domains=domains,
        constraints=constraints
    )

@pytest.fixture
def sample_fca_context():
    """
    Fixture: Contexto formal simple para FCA.
    
    Retorna un contexto formal con 3 objetos y 3 atributos.
    """
    objects = {"o1", "o2", "o3"}
    attributes = {"a1", "a2", "a3"}
    relation = {
        ("o1", "a1"), ("o1", "a2"),
        ("o2", "a2"), ("o2", "a3"),
        ("o3", "a1"), ("o3", "a3")
    }
    return objects, attributes, relation

