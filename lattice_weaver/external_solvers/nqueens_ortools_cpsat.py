from ortools.sat.python import cp_model
from typing import Dict, List, Any, Optional

def solve_nqueens_ortools_cpsat(n: int) -> Optional[Dict[str, int]]:
    """
    Resuelve el problema N-Queens usando Google OR-Tools CP-SAT.
    :param n: El tamaño del tablero (N x N).
    :return: Una solución como diccionario {variable: valor}, o None si no hay solución.
    """
    model = cp_model.CpModel()

    # Variables: q0, q1, ..., q(n-1) representando la fila de la reina en cada columna
    # Dominio: 0, 1, ..., n-1 (filas)
    queens = [model.NewIntVar(0, n - 1, f'q{i}') for i in range(n)]

    # Restricciones: No dos reinas pueden estar en la misma fila, columna o diagonal.
    # La restricción de columna ya está implícita al tener una reina por columna.

    # Restricción de fila: Todas las reinas deben estar en filas diferentes
    model.AddAllDifferent(queens)

    # Restricción de diagonal: No dos reinas pueden estar en la misma diagonal
    for i in range(n):
        for j in range(i + 1, n):
            # Diagonales principales (q_i - i != q_j - j)
            # Diagonales secundarias (q_i + i != q_j + j)
            model.Add(queens[i] - i != queens[j] - j)
            model.Add(queens[i] + i != queens[j] + j)

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        solution = {f'q{i}': solver.Value(queens[i]) for i in range(n)}
        return solution
    return None

if __name__ == "__main__":
    print("Resolviendo 4-Queens con OR-Tools CP-SAT...")
    solution_4 = solve_nqueens_ortools_cpsat(4)
    if solution_4:
        print("Solución encontrada:", solution_4)
    else:
        print("No se encontró solución para 4-Queens.")

    print("\nResolviendo 8-Queens con OR-Tools CP-SAT...")
    solution_8 = solve_nqueens_ortools_cpsat(8)
    if solution_8:
        print("Solución encontrada:", solution_8)
    else:
        print("No se encontró solución para 8-Queens.")

