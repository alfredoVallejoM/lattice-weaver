import pytest
from lattice_weaver.core.csp_problem import CSP, Constraint
from lattice_weaver.core.csp_engine.solver import CSPSolver
from lattice_weaver.formal.cubical_engine import CubicalEngine
from lattice_weaver.formal.type_checker import TypeChecker
from lattice_weaver.lattice_core.parallel_fca import ParallelFCABuilder
from lattice_weaver.topology.tda_engine import TDAEngine


@pytest.fixture
def csp_solver():
    """Solver CSP configurado."""
    # CSPSolver ahora toma una instancia de CSP en su constructor
    # Para este fixture, retornamos una instancia de CSPSolver que puede ser configurada con un CSP específico en cada test
    # O, si el test necesita un solver pre-configurado con un CSP, se puede crear un fixture para ese CSP y pasarlo aquí.
    # Por simplicidad, aquí retornamos la clase CSPSolver para que los tests la instancien con el CSP adecuado.
    return CSPSolver


@pytest.fixture
def formal_verifier():
    """Verificador formal con type checker."""
    type_checker = TypeChecker()
    return CubicalEngine()


@pytest.fixture
def fca_builder():
    """
    Constructor FCA paralelo.
    """
    return ParallelFCABuilder()


@pytest.fixture
def tda_engine():
    """
    Motor TDA para análisis topológico.
    """
    return TDAEngine()


@pytest.fixture
def nqueens_4_problem():
    """
    Problema N-Reinas n=4 para tests.
    """
    variables = frozenset({f'Q{i}' for i in range(4)})
    domains = {var: frozenset(range(4)) for var in variables}
    
    constraints = []
    for i in range(4):
        for j in range(i + 1, 4):
            # No en la misma fila y no en la misma diagonal
            constraints.append(Constraint(
                scope=frozenset({f'Q{i}', f'Q{j}'}),
                relation=lambda ri, rj, current_i=i, current_j=j: ri != rj and abs(ri - rj) != abs(current_i - current_j),
                name=f'neq_diag_Q{i}Q{j}'
            ))
    
    return CSP(
        variables=variables,
        domains=domains,
        constraints=frozenset(constraints),
        name="NQueens_4"
    )


@pytest.fixture
def simple_formal_context():
    """
    Contexto formal simple para tests.
    """
    objects = ['o1', 'o2', 'o3']
    attributes = ['a1', 'a2', 'a3']
    incidence = {
        ('o1', 'a1'), ('o1', 'a2'),
        ('o2', 'a2'), ('o2', 'a3'),
        ('o3', 'a1'), ('o3', 'a3')
    }
    return objects, attributes, incidence

