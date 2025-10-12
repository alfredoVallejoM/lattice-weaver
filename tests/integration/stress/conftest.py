"""
Fixtures para tests de estrés.
"""

import pytest


@pytest.fixture
def stress_timeout():
    """Timeout para tests de estrés (en segundos)."""
    return 120


@pytest.fixture
def large_nqueens_problem():
    """Problema N-Reinas grande (n=16)."""
    from lattice_weaver.formal.csp_integration import CSPProblem
    
    n = 16
    variables = [f'Q{i}' for i in range(n)]
    domains = {var: list(range(n)) for var in variables}
    
    constraints = []
    for i in range(n):
        for j in range(i + 1, n):
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

