import pytest
from tests.integration.helpers import (
    solve_csp_problem,
    create_sudoku_problem
)


@pytest.mark.integration
@pytest.mark.regression
def test_sudoku_4x4_known_solution(load_golden_output):
    """
    Test: Resolver Sudoku 4x4 y comparar con solución conocida.
    
    Validación: Solución debe coincidir exactamente.
    """
    # Cargar golden output
    golden = load_golden_output("sudoku_4x4_solution.json")
    puzzle = golden["puzzle"]
    expected_solution = golden["solution"]
    
    # Resolver problema
    problem = create_sudoku_problem(puzzle)
    stats = solve_csp_problem(problem, max_solutions=1)
    
    assert len(stats.solutions) > 0, "Debe encontrar al menos una solución"
    
    solution = stats.solutions[0]
    
    # Convertir solución a formato de grid
    solution_grid = [[0]*4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            var = f'C{i}{j}'
            solution_grid[i][j] = solution[var]
    
    # Comparar con expected
    assert solution_grid == expected_solution, \
        f"Solución no coincide.\nEncontrada:\n{solution_grid}\nEsperada:\n{expected_solution}"
    
    print(f"✅ Sudoku 4x4: Solución coincide con golden output")
    print(f"   Nodos explorados: {stats.nodes_explored}")


@pytest.mark.integration
@pytest.mark.regression
@pytest.mark.slow
def test_sudoku_4x4_uniqueness():
    """
    Test: Verificar que un Sudoku 4x4 bien formado tiene solución única.
    
    Validación: Debe encontrar exactamente 1 solución.
    """
    # Puzzle con solución única
    puzzle = [
        [1, 0, 0, 4],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [4, 0, 0, 1]
    ]
    
    problem = create_sudoku_problem(puzzle)
    stats = solve_csp_problem(problem, max_solutions=10)  # Buscar hasta 10
    
    # Debe encontrar exactamente 1 solución
    assert len(stats.solutions) == 1, \
        f"Sudoku bien formado debe tener solución única, encontró {len(stats.solutions)}"
    
    print(f"✅ Sudoku 4x4: Solución única verificada")
    print(f"   Nodos explorados: {stats.nodes_explored}")

