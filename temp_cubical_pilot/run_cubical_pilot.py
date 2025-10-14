from lattice_weaver.fibration.constraint_hierarchy import ConstraintHierarchy, Hardness
from lattice_weaver.fibration.fibration_adapters import CubicalEngineAdapter
from lattice_weaver.fibration.fibration_search_solver import FibrationSearchSolver
from lattice_weaver.fibration.energy_landscape_optimized import EnergyLandscapeOptimized
from lattice_weaver.fibration.hacification_engine import HacificationEngine
from temp_cubical_pilot.simple_cubical_problem import cubical_problem_definition
from numba import njit
import time

# 1. Definir el problema cúbico simple
problem_data = cubical_problem_definition()

# 2. Inicializar ConstraintHierarchy y el adaptador
ch = ConstraintHierarchy()
cubical_adapter = CubicalEngineAdapter(ch)

# 3. Traducir el problema cúbico a ConstraintHierarchy
cubical_adapter.translate_problem_to_constraints(problem_data)

print(f"\nConstraint Hierarchy después de la traducción: {ch.get_statistics()}")

# 4. Preparar Fibration Flow components
variables_domains = problem_data["variables"]

# Aplicar JIT a los predicados de las restricciones y funciones del EnergyLandscape
# Para este ejemplo, aplicaremos JIT a los predicados definidos en simple_cubical_problem.py
# y a las funciones de evaluación de EnergyLandscape si fueran funciones separadas.
# Aquí, los predicados ya están definidos como funciones Python, Numba puede compilarlos.

# Refactorizar los predicados para que Numba pueda compilarlos si es necesario
# Para este ejemplo, asumimos que los predicados son compatibles con Numba.

# JIT para la evaluación de restricciones (ejemplo conceptual)
# En una implementación real, esto se haría de forma más dinámica sobre los Callables
@njit
def jit_cubical_equality_predicate(point1, point2, interval):
    if interval == 0: # Numba no maneja strings directamente en njit, usar representación numérica
        return point1 == point2
    return True

@njit
def jit_cubical_type_predicate(point1, point2):
    return point1 in [0, 1] and point2 in [0, 1]

@njit
def jit_soft_preference_predicate(point1, point2):
    return abs(point1 - point2)

# Reconstruir la ConstraintHierarchy con predicados JITeados (ejemplo)
# En una implementación real, el adaptador se encargaría de esto.
ch_jit = ConstraintHierarchy()
for vars, pred_func in problem_data["hard_rules"]:
    if pred_func.__name__ == "cubical_type_predicate":
        ch_jit.add_hard_constraint((vars, jit_cubical_type_predicate))
    elif pred_func.__name__ == "cubical_equality_predicate":
        # Nota: Esto es una simplificación. 'interval' necesitaría ser mapeado a un int para njit.
        # Para un ejemplo real, se necesitaría un preprocesamiento de los datos.
        ch_jit.add_hard_constraint((vars, jit_cubical_equality_predicate))

for vars, pred_func, weight in problem_data["soft_preferences"]:
    if pred_func.__name__ == "<lambda>": # El lambda original
        ch_jit.add_soft_constraint((vars, jit_soft_preference_predicate), weight)


energy_landscape = EnergyLandscapeOptimized(ch_jit)
hacification_engine = HacificationEngine(ch_jit, energy_landscape, variables_domains)

# 5. Inicializar y ejecutar FibrationSearchSolver
solver = FibrationSearchSolver(
    variables=list(variables_domains.keys()),
    domains=variables_domains,
    hierarchy=ch_jit
)

print("\nIniciando búsqueda de solución con Fibration Flow...")
start_time = time.time()
solution = solver.solve()
end_time = time.time()

if solution:
    print("\nSolución encontrada:")
    print(solution)
    all_hard_satisfied, total_energy = ch_jit.evaluate_solution(solution)
    print(f"Hard constraints satisfechas: {all_hard_satisfied}")
    print(f"Energía total (violación de soft constraints): {total_energy}")
else:
    print("\nNo se encontró solución.")

print(f"Tiempo de ejecución: {end_time - start_time:.4f} segundos")

# 6. Benchmarking (ejemplo conceptual)
# Comparar el tiempo de ejecución con y sin JIT (si tuviéramos una versión no-JIT)
# Para este ejemplo, solo mostramos el tiempo con JIT.

# Ejemplo de cómo se usaría el adaptador para traducir la solución
if solution:
    cubical_solution_translated = cubical_adapter.translate_solution_from_constraints(solution)
    print("\nSolución de Fibration Flow traducida a formato cúbico:")
    print(cubical_solution_translated)


# --- Benchmarking de evaluación de predicados (conceptual) ---
print("\n--- Benchmarking de Predicados (Conceptual) ---")

# Predicado original (no JIT)
from typing import Dict, Any
def original_cubical_equality_predicate(assignment: Dict[str, Any]) -> bool:
    point1 = assignment.get('point1')
    point2 = assignment.get('point2')
    interval = assignment.get('interval')
    if interval == 'i0':
        return point1 == point2
    return True

# Predicado JIT (usando la versión njit_cubical_equality_predicate definida arriba)

# Simulación de datos para benchmarking
bench_data = [{'point1': i % 2, 'point2': (i + 1) % 2, 'interval': 'i0' if i % 3 == 0 else 'i1'} for i in range(10000)]

start_time_orig = time.time()
for assign in bench_data:
    original_cubical_equality_predicate(assign)
end_time_orig = time.time()
print(f"Tiempo de ejecución original (no JIT): {end_time_orig - start_time_orig:.6f} segundos")

# Para JIT, necesitamos mapear los strings a ints para el ejemplo simple
bench_data_jit = [{'point1': i % 2, 'point2': (i + 1) % 2, 'interval': 0 if i % 3 == 0 else 1} for i in range(10000)]

start_time_jit = time.time()
for assign in bench_data_jit:
    jit_cubical_equality_predicate(assign['point1'], assign['point2'], assign['interval'])
end_time_jit = time.time()
print(f"Tiempo de ejecución JIT (Numba): {end_time_jit - start_time_jit:.6f} segundos")

print("Nota: El benchmarking real requeriría una integración más profunda de Numba con la evaluación de ConstraintHierarchy.")

# --- Demostración del Sistema de Autoperturbación ---
print("\n--- Demostración del Sistema de Autoperturbación ---")

if solver.best_solution:
    print("Aplicando perturbación a la mejor solución encontrada...")
    original_assignment = solver.best_solution.copy()
    original_energy = solver.landscape.compute_energy(original_assignment).total_energy

    # Obtener acciones y niveles de perturbación disponibles
    potential_actions = solver.autoperturbation_system.get_potential_actions()
    perturbation_levels = solver.autoperturbation_system.get_perturbation_levels()

    print(f"Acciones de perturbación disponibles: {potential_actions}")
    print(f"Niveles de granularidad disponibles: {[level.name for level in perturbation_levels]}")

    # Ejemplo de perturbación: cambiar una variable aleatoria
    perturbed_assignment = solver.autoperturbation_system.apply_perturbation(
        original_assignment,
        perturbation_type="random_variable_change",
        level=perturbation_levels[0] # Usar el primer nivel disponible, por ejemplo LOCAL
    )
    perturbed_energy = solver.landscape.compute_energy(perturbed_assignment).total_energy

    feedback = solver.autoperturbation_system.observe_and_learn(
        original_assignment,
        perturbed_assignment,
        original_energy,
        perturbed_energy
    )

    print(f"Asignación original: {original_assignment}")
    print(f"Energía original: {original_energy:.2f}")
    print(f"Asignación perturbada: {perturbed_assignment}")
    print(f"Energía perturbada: {perturbed_energy:.2f}")
    print(f"Feedback de la perturbación: {feedback}")

    if feedback["perturbation_successful"]:
        print("La perturbación resultó en una mejora (o no empeoró significativamente) la energía.")
    else:
        print("La perturbación empeoró la energía.")

    # Ejemplo de perturbación a nivel de restricción SOFT (cambio de peso)
    print("\nAplicando perturbación a una restricción SOFT (cambio de peso)...")
    # Nota: Esta perturbación modifica la jerarquía de restricciones directamente.
    # Para ver su efecto, se debería re-ejecutar el solver o re-evaluar el paisaje.
    solver.autoperturbation_system.apply_perturbation(
        original_assignment, # La asignación no es relevante para este tipo de perturbación
        perturbation_type="soft_constraint_weight_change",
        level=perturbation_levels[2] # GLOBAL
    )
    print("Peso de restricción SOFT modificado. Esto afectaría búsquedas futuras.")

else:
    print("No se encontró una solución para aplicar perturbaciones.")

