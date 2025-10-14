from constraint import Problem, AllDifferentConstraint
from typing import Dict, List, Any, Optional

def solve_nqueens_python_constraint(n: int) -> Optional[Dict[str, int]]:
    """
    Resuelve el problema N-Queens usando `python-constraint`.
    :param n: El tamaño del tablero (N x N).
    :return: Una solución como diccionario {variable: valor}, o None si no hay solución.
    """
    problem = Problem()

    # Variables: q0, q1, ..., q(n-1) representando la fila de la reina en cada columna
    # Dominio: 0, 1, ..., n-1 (filas)
    variables = [f"q{i}" for i in range(n)]
    domain = list(range(n))

    for var in variables:
        problem.addVariable(var, domain)

    # Restricciones: No dos reinas pueden estar en la misma fila, columna o diagonal.
    # La restricción de columna ya está implícita al tener una reina por columna.

    # Restricción de fila: Todas las reinas deben estar en filas diferentes
    problem.addConstraint(AllDifferentConstraint(), variables)

    # Restricción de diagonal: No dos reinas pueden estar en la misma diagonal
    for i in range(n):
        for j in range(i + 1, n):
            # Diagonales principales (q_i - i != q_j - j)
            # Diagonales secundarias (q_i + i != q_j + j)
            problem.addConstraint(lambda qi, qj, i=i, j=j: abs(qi - qj) != abs(i - j), (f"q{i}", f"q{j}"))

    solutions = problem.getSolutions()
    if solutions:
        return solutions[0]
    return None

if __name__ == "__main__":
    print("Resolviendo 4-Queens con python-constraint...")
    solution_4 = solve_nqueens_python_constraint(4)
    if solution_4:
        print("Solución encontrada:", solution_4)
    else:
        print("No se encontró solución para 4-Queens.")

    print("\nResolviendo 8-Queens con python-constraint...")
    solution_8 = solve_nqueens_python_constraint(8)
    if solution_8:
        print("Solución encontrada:", solution_8)
    else:
        print("No se encontró solución para 8-Queens.")

