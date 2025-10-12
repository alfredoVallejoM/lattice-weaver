"""
Tests de regresión: N-Reinas

Valida que las soluciones de N-Reinas coincidan con golden outputs conocidos.
"""

import pytest
from tests.integration.helpers import (
    solve_csp_problem,
    create_nqueens_problem
)


@pytest.mark.integration
@pytest.mark.regression
def test_nqueens_4_golden_output(load_golden_output):
    """
    Test: Resolver N-Reinas n=4 y comparar con golden output.
    
    Validación: Soluciones deben coincidir exactamente con las conocidas.
    """
    # Cargar golden output
    golden = load_golden_output("nqueens_4_solutions.json")
    expected_solutions = golden["solutions"]
    
    # Resolver problema
    problem = create_nqueens_problem(4)
    stats = solve_csp_problem(problem, max_solutions=10)
    
    # Verificar número de soluciones
    assert len(stats.solutions) == len(expected_solutions), \
        f"Debe encontrar {len(expected_solutions)} soluciones, encontró {len(stats.solutions)}"
    
    # Verificar que las soluciones coinciden
    solutions_set = {tuple(sorted(sol.items())) for sol in stats.solutions}
    expected_set = {tuple(sorted(sol.items())) for sol in expected_solutions}
    
    assert solutions_set == expected_set, \
        f"Soluciones no coinciden con golden output.\nEncontradas: {solutions_set}\nEsperadas: {expected_set}"
    
    print(f"✅ N-Reinas n=4: {len(stats.solutions)} soluciones coinciden con golden output")


@pytest.mark.integration
@pytest.mark.regression
@pytest.mark.slow
def test_nqueens_8_solution_count(load_golden_output):
    """
    Test: Resolver N-Reinas n=8 y verificar el número total de soluciones.
    
    Validación: Debe encontrar exactamente 92 soluciones (hecho matemático conocido).
    """
    # Cargar golden output
    golden = load_golden_output("nqueens_8_count.json")
    expected_count = golden["total_solutions"]
    
    # Resolver problema
    problem = create_nqueens_problem(8)
    stats = solve_csp_problem(problem, max_solutions=100)  # Buscar todas
    
    # Verificar número de soluciones
    assert len(stats.solutions) == expected_count, \
        f"Debe encontrar {expected_count} soluciones, encontró {len(stats.solutions)}"
    
    print(f"✅ N-Reinas n=8: {len(stats.solutions)} soluciones (correcto)")
    print(f"   Nodos explorados: {stats.nodes_explored}")
    print(f"   Backtracks: {stats.backtracks}")

