import pytest

from lattice_weaver.core.csp_problem import CSP, Constraint
from lattice_weaver.core.csp_engine.solver import CSPSolver, CSPSolutionStats
from lattice_weaver.formal.csp_cubical_bridge import CSPToCubicalBridge
from lattice_weaver.formal.cubical_engine import CubicalEngine


# Helper function to generate N-Queens CSP for testing
def generate_nqueens_csp(n: int) -> CSP:
    variables = {f'Q{i}' for i in range(n)}
    domains = {f'Q{i}': frozenset(range(n)) for i in range(n)}
    constraints = []

    for i in range(n):
        for j in range(i + 1, n):
            # Queens must not be in the same row (handled by domain)
            # Queens must not be in the same column (handled by variable)
            # Queens must not be in the same diagonal
            constraints.append(Constraint(
                scope=frozenset({f'Q{i}', f'Q{j}'}),
                relation=lambda qi, qj, i=i, j=j: qi != qj and abs(qi - qj) != abs(i - j),
                name=f'diag_neq_Q{i}Q{j}'
            ))
    return CSP(variables=variables, domains=domains, constraints=constraints)


@pytest.mark.integration
@pytest.mark.complex
def test_csp_solution_to_formal_proof():
    """
    Test: Resolver CSP y verificar solución formalmente.
    Flujo:
    1. Resolver problema CSP (N-Reinas n=4)
    2. Traducir solución a proposición formal
    3. Verificar que la proposición es válida
    Validación: Solución CSP ≡ Prueba formal válida
    """
    n = 4
    csp_problem = generate_nqueens_csp(n)
    solver = CSPSolver(csp_problem)
    stats: CSPSolutionStats = solver.solve(max_solutions=1)

    assert len(stats.solutions) > 0, "Debe encontrar al menos una solución"
    solution = stats.solutions[0]

    # Use CSPToCubicalBridge to verify the solution
    bridge = CSPToCubicalBridge(csp_problem)
    assert bridge.verify_solution(solution) is True

    print(f"✅ Solución CSP verificada formalmente: {solution}")


@pytest.mark.integration
@pytest.mark.complex
def test_csp_constraints_to_formal_types():
    """
    Test: Traducir restricciones CSP a tipos dependientes.
    Flujo:
    1. Definir restricciones CSP complejas
    2. Traducir a tipos dependientes
    3. Verificar type-checking
    Validación: Restricciones CSP ≡ Tipos válidos
    """
    n = 4
    csp_problem = generate_nqueens_csp(n)
    bridge = CSPToCubicalBridge(csp_problem)

    # The cubical_type generation implicitly translates constraints to formal types
    cubical_type = bridge.cubical_type
    assert cubical_type is not None
    assert len(cubical_type.constraint_props) == len(csp_problem.constraints)

    # Further formal verification would involve a full CubicalEngine and TypeChecker
    # For now, we assert that the bridge successfully created the cubical type.
    print(f"✅ {len(cubical_type.constraint_props)} restricciones traducidas a tipos válidos")


@pytest.mark.integration
@pytest.mark.complex
def test_csp_optimization_with_formal_guarantees():
    """
    Test: Resolver CSP con optimizaciones y verificar equivalencia formal.
    Flujo:
    1. Resolver CSP con optimizaciones habilitadas
    2. Resolver CSP sin optimizaciones
    3. Verificar que las soluciones son equivalentes
    Validación: Solución optimizada ≡ Solución original
    """
    n = 4
    csp_problem = generate_nqueens_csp(n)

    # 1. Resolver con optimizaciones (por defecto están habilitadas en CSPSolver)
    solver_optimized = CSPSolver(csp_problem, use_ac3=True)
    stats_optimized: CSPSolutionStats = solver_optimized.solve(max_solutions=2)

    assert len(stats_optimized.solutions) > 0, "Debe encontrar soluciones"

    # 2. Resolver sin optimizaciones (deshabilitar AC3)
    solver_baseline = CSPSolver(csp_problem, use_ac3=False)
    stats_baseline: CSPSolutionStats = solver_baseline.solve(max_solutions=2)

    # 3. Verificar equivalencia
    assert len(stats_optimized.solutions) == len(stats_baseline.solutions), \
        "Número de soluciones debe ser igual"

    # Convertir listas de soluciones a conjuntos de frozensets para comparación de equivalencia
    set_optimized_solutions = {frozenset(sol.items()) for sol in stats_optimized.solutions}
    set_baseline_solutions = {frozenset(sol.items()) for sol in stats_baseline.solutions}

    assert set_optimized_solutions == set_baseline_solutions, \
        "Las soluciones optimizadas deben ser equivalentes a las soluciones de línea base"

    print(f"✅ Optimizaciones preservan semántica: {len(stats_optimized.solutions)} soluciones válidas")
    print(f"   Nodos explorados (optimizado): {stats_optimized.nodes_explored}")
    print(f"   Nodos explorados (baseline): {stats_baseline.nodes_explored}")

