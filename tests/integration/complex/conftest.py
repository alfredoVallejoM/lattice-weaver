"""
Fixtures para tests de integración complejos.
"""

import pytest
from lattice_weaver.arc_engine.core import ArcEngine
from lattice_weaver.formal.cubical_engine import CubicalEngine
from lattice_weaver.formal.type_checker import TypeChecker
from lattice_weaver.lattice_core.parallel_fca import ParallelFCABuilder
from lattice_weaver.topology.tda_engine import TDAEngine
from lattice_weaver.formal.csp_integration import CSPProblem


@pytest.fixture
def csp_solver():
    """Solver CSP configurado."""
    return ArcEngine()


@pytest.fixture
def formal_verifier():
    """Verificador formal con type checker."""
    type_checker = TypeChecker()
    return CubicalEngine(type_checker)


@pytest.fixture
def fca_builder():
    """Constructor FCA paralelo."""
    return ParallelFCABuilder()


@pytest.fixture
def tda_engine():
    """Motor TDA para análisis topológico."""
    return TDAEngine()


@pytest.fixture
def nqueens_4_problem():
    """Problema N-Reinas n=4 para tests."""
    variables = ['Q0', 'Q1', 'Q2', 'Q3']
    domains = {var: list(range(4)) for var in variables}
    
    constraints = []
    for i in range(4):
        for j in range(i + 1, 4):
            # No en la misma fila
            constraints.append((
                f'Q{i}',
                f'Q{j}',
                lambda ri, rj: ri != rj
            ))
            # No en la misma diagonal
            constraints.append((
                f'Q{i}',
                f'Q{j}',
                lambda ri, rj, i=i, j=j: abs(ri - rj) != abs(i - j)
            ))
    
    return CSPProblem(
        variables=variables,
        domains=domains,
        constraints=constraints
    )


@pytest.fixture
def simple_formal_context():
    """Contexto formal simple para tests."""
    objects = ['o1', 'o2', 'o3']
    attributes = ['a1', 'a2', 'a3']
    incidence = {
        ('o1', 'a1'), ('o1', 'a2'),
        ('o2', 'a2'), ('o2', 'a3'),
        ('o3', 'a1'), ('o3', 'a3')
    }
    return objects, attributes, incidence

