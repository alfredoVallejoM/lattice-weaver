"""
Tests de integración entre CSPSolver y ArcEngine.
Verifican que los resultados son idénticos con y sin ArcEngine.
"""
import pytest
from lattice_weaver.core.csp_problem import CSP, Constraint
from lattice_weaver.core.csp_engine.solver import CSPSolver

def create_nqueens_csp(n=4):
    """Crea un CSP de N-Queens para testing"""
    variables = [f"Q{i}" for i in range(n)]
    domains = {var: list(range(n)) for var in variables}
    
    constraints = []
    for i in range(n):
        for j in range(i + 1, n):
            def not_same_row(val_i, val_j, i=i, j=j):
                return val_i != val_j
            
            def not_same_diagonal(val_i, val_j, i=i, j=j):
                return abs(val_i - val_j) != abs(i - j)
            
            constraints.append(
                Constraint({f"Q{i}", f"Q{j}"}, not_same_row, name=f"not_same_row_{i}_{j}")
            )
            constraints.append(
                Constraint({f"Q{i}", f"Q{j}"}, not_same_diagonal, name=f"not_same_diagonal_{i}_{j}")
            )
    
    return CSP(variables, domains, constraints)

def test_arc_engine_produces_same_solutions():
    """Verifica que ArcEngine produce las mismas soluciones que AC-3 básico"""
    csp = create_nqueens_csp(n=4)
    
    # Solver sin ArcEngine
    solver_basic = CSPSolver(csp, use_arc_engine=False)
    stats_basic = solver_basic.solve(all_solutions=True)
    
    # Solver con ArcEngine
    csp2 = create_nqueens_csp(n=4)  # CSP fresco
    solver_arc = CSPSolver(csp2, use_arc_engine=True)
    stats_arc = solver_arc.solve(all_solutions=True)
    
    # Verificar mismo número de soluciones
    assert len(stats_basic.solutions) == len(stats_arc.solutions), \
        f"Diferente número de soluciones: {len(stats_basic.solutions)} vs {len(stats_arc.solutions)}"
    
    # Verificar que las soluciones son las mismas (orden puede variar)
    solutions_basic = {frozenset(sol.assignment.items()) for sol in stats_basic.solutions}
    solutions_arc = {frozenset(sol.assignment.items()) for sol in stats_arc.solutions}
    
    assert solutions_basic == solutions_arc, \
        "Las soluciones encontradas son diferentes"

def test_arc_engine_is_faster_or_equal():
    """Verifica que ArcEngine no es significativamente más lento"""
    csp = create_nqueens_csp(n=8)
    
    # Solver sin ArcEngine
    solver_basic = CSPSolver(csp, use_arc_engine=False)
    stats_basic = solver_basic.solve()
    
    # Solver con ArcEngine
    csp2 = create_nqueens_csp(n=8)
    solver_arc = CSPSolver(csp2, use_arc_engine=True)
    stats_arc = solver_arc.solve()
    
    # ArcEngine puede ser más lento en problemas pequeños, pero no >3x
    assert stats_arc.time_elapsed < stats_basic.time_elapsed * 3, \
        f"ArcEngine demasiado lento: {stats_arc.time_elapsed}s vs {stats_basic.time_elapsed}s"

def test_backward_compatibility():
    """Verifica que el comportamiento por defecto no cambia"""
    csp = create_nqueens_csp(n=4)
    
    # Comportamiento por defecto (sin especificar use_arc_engine)
    solver = CSPSolver(csp)
    stats = solver.solve()
    
    # Debe funcionar sin errores
    assert len(stats.solutions) > 0
    assert stats.nodes_explored > 0

@pytest.mark.parametrize("n", [4, 6, 8])
def test_consistency_across_problem_sizes(n):
    """Verifica consistencia en diferentes tamaños de problema"""
    csp_basic = create_nqueens_csp(n=n)
    csp_arc = create_nqueens_csp(n=n)
    
    solver_basic = CSPSolver(csp_basic, use_arc_engine=False)
    solver_arc = CSPSolver(csp_arc, use_arc_engine=True)
    
    stats_basic = solver_basic.solve()
    stats_arc = solver_arc.solve()
    
    assert len(stats_basic.solutions) == len(stats_arc.solutions)

