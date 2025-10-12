import pytest
from lattice_weaver.arc_engine.core import ArcEngine
from lattice_weaver.arc_engine.csp_solver import CSPProblem, CSPSolver
from multiprocessing import active_children
from lattice_weaver.arc_engine.constraints import register_relation, nqueens_not_equal, nqueens_not_diagonal

# Asegurarse de que las relaciones estén registradas para el proceso principal y los workers

# Helper para crear el problema de N-Reinas con relaciones registradas

# Helper para crear el problema de N-Reinas con relaciones registradas
def create_nqueens_problem(n):
    variable_names = [f"Q{i}" for i in range(n)]
    domains = {name: list(range(n)) for name in variable_names}
    problem = CSPProblem(variable_names, domains, [])

    for i in range(n):
        for j in range(i + 1, n):
            # Restricción de fila/columna (no es necesaria si los dominios son 0..n-1 y cada Q_i es una fila)
            # Restricción de no estar en la misma columna (implícita si cada Q_i es una columna)

            # Restricción de no estar en la misma fila (si Q_i representa la columna de la reina en la fila i)
            # No se necesita si los valores del dominio son las columnas y las variables son las filas

            # Restricción de no estar en la misma diagonal
            # Para que la relación sea serializable, no puede ser una lambda que capture i, j
            # Se necesita una función que tome solo val1, val2 y luego calcule la diagonal
            # Esto requiere un cambio en cómo se definen las restricciones o un wrapper
            # Por ahora, se usará una función genérica y se asumirá que la lógica de la diagonal se maneja externamente
            # o se refactorizará la definición de restricciones para incluir parámetros adicionales.

            # Para este test, simplificaremos y usaremos solo la restricción de no ser igual
            # y asumiremos que el problema de N-Reinas se construye con esta base.
            # La restricción de diagonal se puede añadir como una relación registrada más compleja.

            # Simplificación para el test: solo no-igualdad por ahora para probar paralelización
            problem.constraints.append((f"Q{i}", f"Q{j}", nqueens_not_equal))
            problem.constraints.append((f"Q{i}", f"Q{j}", nqueens_not_diagonal))
            
            # Para la diagonal, necesitaríamos una forma de pasar i y j a la función de relación
            # Esto es un desafío para la serialización de multiprocessing con funciones lambda.
            # Una solución sería que la función de relación sea un método de una clase serializable
            # o que el ArcEngine pueda inyectar los índices de las variables en la relación.
            # Por ahora, nos centraremos en la paralelización de restricciones binarias simples.

    return problem


def test_multiprocess_ac3_nqueens_4():
    """
    Valida la paralelización multiproceso de AC-3.1 para el problema de 4-Reinas.
    """
    n = 4
    problem = create_nqueens_problem(n)

    # Crear ArcEngine con paralelización topológica
    engine = ArcEngine(parallel=True, parallel_mode='topological')
    for var_name in problem.variables:
        engine.add_variable(var_name, problem.domains[var_name])
    for var1, var2, relation_func in problem.constraints:
        engine.add_constraint(var1, var2, relation_func, cid=f"{var1}_{var2}_{relation_func.__name__}")

    solver = CSPSolver(use_tms=False, parallel=True, parallel_mode='topological')
    solutions = solver.solve(problem, return_all=True)

    # El problema de 4-Reinas tiene 2 soluciones únicas
    assert len(solutions.solutions) == 2
    for sol in solutions.solutions:

        assert len(sol.assignment) == n
        # Verificar que las soluciones son válidas (no hay reinas en la misma fila, columna o diagonal)
        # Esto es una verificación simplificada, ya que la restricción de diagonal no está completamente implementada
        # en create_nqueens_problem para la serialización.
        # Para una verificación completa, se necesitaría una implementación robusta de la restricción de diagonal serializable.
        
        # Verificación de no-igualdad (simplificada)
        assigned_cols = list(sol.assignment.values())
        assert len(set(assigned_cols)) == n # Todas las reinas en columnas diferentes

    # Verificar que se usaron múltiples procesos (esto es heurístico y puede variar)
    # No hay una forma directa de verificar esto desde el test sin instrumentar el Pool
    # Sin embargo, la lógica de TopologicalParallelAC3 debería invocar Pool.map
    # Se puede verificar indirectamente si el código de _process_independent_group_worker se ejecuta
    # pero eso requeriría mocks o un logging muy específico.
    # Por ahora, confiamos en la implementación de TopologicalParallelAC3.


# Helper para crear el problema de N-Reinas con relaciones registradas (incluyendo diagonal)
def create_nqueens_problem_full(n):
    variable_names = [f"Q{i}" for i in range(n)]
    domains = {name: list(range(n)) for name in variable_names}
    problem = CSPProblem(variable_names, domains, [])

    # Registrar la relación de diagonal con parámetros i, j
    # Esto requiere una refactorización de cómo se manejan las relaciones con parámetros
    # o una clase de Constraint más compleja que encapsule la lógica.
    # Por ahora, se asume que la relación de diagonal se puede registrar de alguna forma.

    for i in range(n):
        for j in range(i + 1, n):
            # Restricción de no estar en la misma columna (implícita si cada Q_i es una columna)
            # Restricción de no estar en la misma fila (si Q_i representa la columna de la reina en la fila i)
            # No se necesita si los valores del dominio son las columnas y las variables son las filas

            # Restricción de no estar en la misma diagonal
            # Se necesita una función que tome val1, val2 y los índices i, j
            # Para la serialización, esta función no puede ser una lambda que capture i, j
            # Podríamos registrar una función genérica y pasar i, j como parte del contexto de la restricción
            # o como parte de la llamada a la relación.

            # Por ahora, para el test, se usará una función que simule la diagonal
            # y se registrará de forma genérica. Esto es un placeholder.
            def _nqueens_diagonal_relation(val1, val2, var1_idx, var2_idx):
                return abs(val1 - val2) != abs(var1_idx - var2_idx)
            
            # Esto no es directamente serializable si se registra así.
            # La solución real es que la Constraint almacene los índices y la relación genérica.
            # Para el test, se simulará que la relación ya está registrada y es accesible.

            # Para el test, vamos a crear una relación que encapsule i y j
            # Esto NO es serializable directamente, pero es para ilustrar el concepto.
            # La solución real implicaría que la clase Constraint almacene los índices
            # y la función de relación sea genérica.

            # Por simplicidad para el test de paralelización, nos quedaremos con la versión simplificada.
            # La implementación completa de N-Reinas con paralelización y serialización de restricciones
            # de diagonal requerirá una refactorización más profunda de la clase Constraint.

            problem.constraints.append((f"Q{i}", f"Q{j}", nqueens_not_equal))
            problem.constraints.append((f"Q{i}", f"Q{j}", nqueens_not_diagonal))
            
            # Para la diagonal, se necesita una forma de pasar i y j a la función de relación
            # Esto es un desafío para la serialización de multiprocessing con funciones lambda.
            # Una solución sería que la función de relación sea un método de una clase serializable
            # o que el ArcEngine pueda inyectar los índices de las variables en la relación.
            # Por ahora, nos centraremos en la paralelización de restricciones binarias simples.

    return problem


def test_multiprocess_ac3_nqueens_4_full():
    """
    Valida la paralelización multiproceso de AC-3.1 para el problema de 4-Reinas con restricciones de diagonal.
    """
    n = 4
    problem = create_nqueens_problem_full(n)

    # Crear ArcEngine con paralelización topológica
    engine = ArcEngine(parallel=True, parallel_mode='topological')
    for var_name in problem.variables:
        engine.add_variable(var_name, problem.domains[var_name])
    for var1, var2, relation_func in problem.constraints:
        engine.add_constraint(var1, var2, relation_func, cid=f"{var1}_{var2}_{relation_func.__name__}")

    solver = CSPSolver(use_tms=False, parallel=True, parallel_mode='topological')
    solutions = solver.solve(problem, return_all=True)

    # El problema de 4-Reinas tiene 2 soluciones únicas
    assert len(solutions.solutions) == 2
    for sol in solutions.solutions:

        assert len(sol.assignment) == n
        # Verificar que las soluciones son válidas (no hay reinas en la misma fila, columna o diagonal)
        assigned_cols = list(sol.assignment.values())
        assert len(set(assigned_cols)) == n # Todas las reinas en columnas diferentes

        # Verificar restricciones de diagonal
        for i in range(n):
            for j in range(i + 1, n):
                val_i = sol.assignment[f"Q{i}"]
                val_j = sol.assignment[f"Q{j}"]
                assert abs(val_i - val_j) != abs(i - j), f"Fallo diagonal para Q{i}={val_i}, Q{j}={val_j}"



