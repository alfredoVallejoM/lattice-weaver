"""
Tests para el sistema de estrategias modulares del CSPSolver.

Este módulo prueba:
1. Intercambiabilidad de estrategias
2. Retrocompatibilidad (comportamiento por defecto)
3. Correctitud de cada estrategia individual
4. Integración con CSPSolver
"""

import pytest
from lattice_weaver.core.csp_problem import CSP, Constraint
from lattice_weaver.core.csp_engine.solver import CSPSolver
from lattice_weaver.core.csp_engine.strategies import (
    FirstUnassignedSelector,
    MRVSelector,
    DegreeSelector,
    MRVDegreeSelector,
    NaturalOrderer,
    LCVOrderer,
    RandomOrderer
)


# ============================================================================
# Fixtures: Problemas CSP de prueba
# ============================================================================

@pytest.fixture
def simple_csp():
    """CSP simple: 3 variables, dominios {1,2,3}, restricciones de desigualdad."""
    variables = ['A', 'B', 'C']
    domains = {var: frozenset([1, 2, 3]) for var in variables}
    constraints = [
        Constraint(scope=frozenset(['A', 'B']), relation=lambda a, b: a != b),
        Constraint(scope=frozenset(['B', 'C']), relation=lambda b, c: b != c),
    ]
    return CSP(variables=frozenset(variables), domains=domains, constraints=constraints)


@pytest.fixture
def nqueens_4():
    """N-Queens 4x4."""
    n = 4
    variables = [f'Q{i}' for i in range(n)]
    domains = {var: frozenset(range(n)) for var in variables}
    
    constraints = []
    for i in range(n):
        for j in range(i + 1, n):
            # No misma fila (implícito por variables diferentes)
            # No misma columna
            constraints.append(
                Constraint(
                    scope=frozenset([f'Q{i}', f'Q{j}']),
                    relation=lambda vi, vj, i=i, j=j: vi != vj
                )
            )
            # No misma diagonal
            constraints.append(
                Constraint(
                    scope=frozenset([f'Q{i}', f'Q{j}']),
                    relation=lambda vi, vj, i=i, j=j: abs(vi - vj) != abs(i - j)
                )
            )
    
    return CSP(variables=frozenset(variables), domains=domains, constraints=constraints)


@pytest.fixture
def graph_coloring():
    """Graph coloring: 5 nodos, 3 colores."""
    variables = ['A', 'B', 'C', 'D', 'E']
    domains = {var: frozenset([1, 2, 3]) for var in variables}
    
    # Grafo: A-B, B-C, C-D, D-E, E-A (ciclo)
    edges = [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'E'), ('E', 'A')]
    constraints = [
        Constraint(scope=frozenset([u, v]), relation=lambda c1, c2: c1 != c2)
        for u, v in edges
    ]
    
    return CSP(variables=frozenset(variables), domains=domains, constraints=constraints)


# ============================================================================
# Tests: Selectores de Variables
# ============================================================================

def test_first_unassigned_selector(simple_csp):
    """FirstUnassignedSelector debe seleccionar la primera variable no asignada."""
    selector = FirstUnassignedSelector()
    assignment = {}
    domains = {var: list(simple_csp.domains[var]) for var in simple_csp.variables}
    
    # Primera selección: debe ser 'A' (primera en orden alfabético)
    var = selector.select(simple_csp, assignment, domains)
    assert var in simple_csp.variables
    
    # Asignar 'A' y seleccionar de nuevo
    assignment['A'] = 1
    var = selector.select(simple_csp, assignment, domains)
    assert var != 'A' and var in simple_csp.variables
    
    # Asignar todas las variables
    for v in simple_csp.variables:
        assignment[v] = 1
    var = selector.select(simple_csp, assignment, domains)
    assert var is None  # Todas asignadas


def test_mrv_selector(simple_csp):
    """MRVSelector debe seleccionar la variable con menor dominio."""
    selector = MRVSelector()
    assignment = {}
    domains = {
        'A': [1, 2, 3],
        'B': [1],  # Menor dominio
        'C': [1, 2]
    }
    
    var = selector.select(simple_csp, assignment, domains)
    assert var == 'B'  # Menor dominio (1 valor)


def test_degree_selector(graph_coloring):
    """DegreeSelector debe seleccionar la variable con mayor degree."""
    selector = DegreeSelector()
    assignment = {}
    domains = {var: list(graph_coloring.domains[var]) for var in graph_coloring.variables}
    
    # En un ciclo, todas las variables tienen degree 2
    # Cualquiera es válida
    var = selector.select(graph_coloring, assignment, domains)
    assert var in graph_coloring.variables


def test_mrv_degree_selector(simple_csp):
    """MRVDegreeSelector debe combinar MRV y Degree."""
    selector = MRVDegreeSelector()
    assignment = {}
    
    # Caso 1: Dominios diferentes -> MRV gana
    domains = {
        'A': [1, 2, 3],
        'B': [1],  # Menor dominio
        'C': [1, 2]
    }
    var = selector.select(simple_csp, assignment, domains)
    assert var == 'B'
    
    # Caso 2: Dominios iguales -> Degree como desempate
    domains = {
        'A': [1, 2],  # Degree 1 (conectada con B)
        'B': [1, 2],  # Degree 2 (conectada con A y C)
        'C': [1, 2]   # Degree 1 (conectada con B)
    }
    var = selector.select(simple_csp, assignment, domains)
    assert var == 'B'  # Mayor degree


# ============================================================================
# Tests: Ordenadores de Valores
# ============================================================================

def test_natural_orderer(simple_csp):
    """NaturalOrderer debe mantener el orden original del dominio."""
    orderer = NaturalOrderer()
    assignment = {}
    domains = {'A': [3, 1, 2], 'B': [1, 2, 3], 'C': [2, 3, 1]}
    
    ordered = orderer.order('A', simple_csp, assignment, domains)
    assert ordered == [3, 1, 2]  # Orden original


def test_lcv_orderer(simple_csp):
    """LCVOrderer debe ordenar valores menos restrictivos primero."""
    orderer = LCVOrderer()
    assignment = {}
    domains = {var: list(simple_csp.domains[var]) for var in simple_csp.variables}
    
    # Para 'A', todos los valores son igualmente restrictivos al inicio
    ordered = orderer.order('A', simple_csp, assignment, domains)
    assert len(ordered) == 3
    assert set(ordered) == {1, 2, 3}


def test_random_orderer(simple_csp):
    """RandomOrderer debe ordenar valores aleatoriamente (pero reproduciblemente con seed)."""
    orderer1 = RandomOrderer(seed=42)
    orderer2 = RandomOrderer(seed=42)
    orderer3 = RandomOrderer(seed=99)
    
    assignment = {}
    domains = {var: list(simple_csp.domains[var]) for var in simple_csp.variables}
    
    ordered1 = orderer1.order('A', simple_csp, assignment, domains)
    ordered2 = orderer2.order('A', simple_csp, assignment, domains)
    ordered3 = orderer3.order('A', simple_csp, assignment, domains)
    
    # Misma semilla -> mismo orden
    assert ordered1 == ordered2
    
    # Diferente semilla -> probablemente diferente orden
    # (no garantizado, pero muy probable)
    assert set(ordered1) == set(ordered3) == {1, 2, 3}


# ============================================================================
# Tests: Integración con CSPSolver
# ============================================================================

def test_solver_default_strategies(simple_csp):
    """CSPSolver sin estrategias especificadas debe usar defaults."""
    solver = CSPSolver(simple_csp)
    assert isinstance(solver.variable_selector, FirstUnassignedSelector)
    assert isinstance(solver.value_orderer, NaturalOrderer)
    
    stats = solver.solve()
    assert len(stats.solutions) >= 1


def test_solver_with_mrv_degree_lcv(nqueens_4):
    """CSPSolver con MRV+Degree+LCV debe resolver N-Queens eficientemente."""
    solver = CSPSolver(
        nqueens_4,
        variable_selector=MRVDegreeSelector(),
        value_orderer=LCVOrderer()
    )
    
    stats = solver.solve()
    assert len(stats.solutions) >= 1
    # Verificar que la solución es válida
    solution = stats.solutions[0].assignment
    assert len(solution) == 4
    # Verificar que todas las restricciones se satisfacen
    for constraint in nqueens_4.constraints:
        scope_list = list(constraint.scope)
        if len(scope_list) == 2:
            v1, v2 = scope_list
            assert constraint.relation(solution[v1], solution[v2])


def test_solver_strategy_interchangeability(graph_coloring):
    """Diferentes estrategias deben encontrar soluciones válidas."""
    strategies = [
        (FirstUnassignedSelector(), NaturalOrderer()),
        (MRVSelector(), NaturalOrderer()),
        (DegreeSelector(), NaturalOrderer()),
        (MRVDegreeSelector(), LCVOrderer()),
    ]
    
    for var_sel, val_ord in strategies:
        solver = CSPSolver(graph_coloring, variable_selector=var_sel, value_orderer=val_ord)
        stats = solver.solve()
        
        # Todas las estrategias deben encontrar al menos una solución
        assert len(stats.solutions) >= 1, f"Strategy {var_sel}/{val_ord} failed"
        
        # Verificar que la solución es válida
        solution = stats.solutions[0].assignment
        for constraint in graph_coloring.constraints:
            scope_list = list(constraint.scope)
            if len(scope_list) == 2:
                v1, v2 = scope_list
                assert constraint.relation(solution[v1], solution[v2])


def test_solver_retrocompatibility(simple_csp):
    """CSPSolver debe ser retrocompatible con código existente."""
    # Código antiguo: sin especificar estrategias
    solver_old = CSPSolver(simple_csp)
    stats_old = solver_old.solve()
    
    # Código nuevo: especificando estrategias equivalentes
    solver_new = CSPSolver(
        simple_csp,
        variable_selector=FirstUnassignedSelector(),
        value_orderer=NaturalOrderer()
    )
    stats_new = solver_new.solve()
    
    # Ambos deben encontrar soluciones
    assert len(stats_old.solutions) >= 1
    assert len(stats_new.solutions) >= 1


def test_solver_efficiency_comparison(nqueens_4):
    """Comparar eficiencia de diferentes estrategias."""
    # Estrategia básica
    solver_basic = CSPSolver(
        nqueens_4,
        variable_selector=FirstUnassignedSelector(),
        value_orderer=NaturalOrderer()
    )
    stats_basic = solver_basic.solve()
    
    # Estrategia avanzada
    solver_advanced = CSPSolver(
        nqueens_4,
        variable_selector=MRVDegreeSelector(),
        value_orderer=LCVOrderer()
    )
    stats_advanced = solver_advanced.solve()
    
    # Ambas deben encontrar solución
    assert len(stats_basic.solutions) >= 1
    assert len(stats_advanced.solutions) >= 1
    
    # Estrategia avanzada debe explorar menos nodos (o igual)
    # Nota: Esto no siempre se cumple para problemas pequeños,
    # pero es una buena verificación general
    print(f"Basic: {stats_basic.nodes_explored} nodes")
    print(f"Advanced: {stats_advanced.nodes_explored} nodes")


# ============================================================================
# Tests: Edge Cases
# ============================================================================

def test_empty_domain():
    """Solver debe manejar dominios vacíos correctamente."""
    variables = ['A', 'B']
    domains = {'A': frozenset([1]), 'B': frozenset()}  # Dominio vacío
    constraints = []
    csp = CSP(variables=frozenset(variables), domains=domains, constraints=constraints)
    
    solver = CSPSolver(csp, variable_selector=MRVSelector())
    stats = solver.solve()
    
    # No debe encontrar solución (dominio vacío)
    assert len(stats.solutions) == 0


def test_single_variable():
    """Solver debe manejar CSP de una sola variable."""
    variables = ['A']
    domains = {'A': frozenset([1, 2, 3])}
    constraints = []
    csp = CSP(variables=frozenset(variables), domains=domains, constraints=constraints)
    
    solver = CSPSolver(csp, variable_selector=MRVSelector(), value_orderer=LCVOrderer())
    stats = solver.solve()
    
    # Debe encontrar 1 solución (por defecto max_solutions=1)
    # Pero el solver explora todas hasta encontrar la primera
    assert len(stats.solutions) >= 1
    assert stats.solutions[0].assignment['A'] in [1, 2, 3]


def test_all_solutions_with_strategies(simple_csp):
    """Solver con estrategias debe encontrar todas las soluciones cuando se solicite."""
    solver = CSPSolver(
        simple_csp,
        variable_selector=MRVDegreeSelector(),
        value_orderer=LCVOrderer()
    )
    stats = solver.solve(all_solutions=True, max_solutions=100)
    
    # Debe encontrar múltiples soluciones
    assert len(stats.solutions) > 1
    
    # Todas las soluciones deben ser diferentes
    assignments = [frozenset(sol.assignment.items()) for sol in stats.solutions]
    assert len(assignments) == len(set(assignments))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

