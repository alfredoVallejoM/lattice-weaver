"""
Tests de regresión: Graph Coloring

Valida que las soluciones de coloreo de grafos sean correctas.
"""

import pytest
from tests.integration.helpers import (
    solve_csp_problem,
    create_map_coloring_problem
)


@pytest.mark.integration
@pytest.mark.regression
def test_map_coloring_australia(load_golden_output):
    """
    Test: Colorear mapa de Australia con 3 colores.
    
    Validación: Debe encontrar al menos una solución válida.
    """
    # Cargar golden output
    golden = load_golden_output("australia_map_coloring.json")
    regions = golden["regions"]
    borders = [tuple(border) for border in golden["borders"]]
    n_colors = golden["colors"]
    
    # Resolver problema
    problem = create_map_coloring_problem(regions, borders, n_colors)
    stats = solve_csp_problem(problem, max_solutions=5)
    
    assert len(stats.solutions) > 0, "Debe encontrar al menos una solución"
    
    # Verificar que la solución es válida
    solution = stats.solutions[0]
    
    for r1, r2 in borders:
        assert solution[r1] != solution[r2], \
            f"Regiones adyacentes {r1} y {r2} tienen el mismo color"
    
    print(f"✅ Map Coloring Australia: Solución válida encontrada")
    print(f"   Soluciones encontradas: {len(stats.solutions)}")
    print(f"   Nodos explorados: {stats.nodes_explored}")
    print(f"   Solución: {solution}")


@pytest.mark.integration
@pytest.mark.regression
def test_graph_coloring_chromatic_number():
    """
    Test: Verificar número cromático de un grafo conocido.
    
    Validación: Grafo completo K4 requiere 4 colores.
    """
    # Grafo completo K4 (todos conectados)
    regions = ['A', 'B', 'C', 'D']
    borders = []
    for i, r1 in enumerate(regions):
        for r2 in regions[i+1:]:
            borders.append((r1, r2))
    
    # Intentar con 3 colores (debe fallar)
    problem_3 = create_map_coloring_problem(regions, borders, 3)
    stats_3 = solve_csp_problem(problem_3, max_solutions=1)
    
    assert len(stats_3.solutions) == 0, \
        "K4 no debe ser coloreable con 3 colores"
    
    # Intentar con 4 colores (debe funcionar)
    problem_4 = create_map_coloring_problem(regions, borders, 4)
    stats_4 = solve_csp_problem(problem_4, max_solutions=1)
    
    assert len(stats_4.solutions) > 0, \
        "K4 debe ser coloreable con 4 colores"
    
    print(f"✅ Número cromático verificado: χ(K4) = 4")
    print(f"   Con 3 colores: {len(stats_3.solutions)} soluciones")
    print(f"   Con 4 colores: {len(stats_4.solutions)} soluciones")

