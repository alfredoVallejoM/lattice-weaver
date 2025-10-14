from typing import Dict, List, Any, Optional
import time

from lattice_weaver.fibration import (
    ConstraintHierarchy,
    ConstraintLevel,
    Hardness,
    EnergyLandscapeOptimized,
    HacificationEngine,
    LandscapeModulator,
    FocusOnGlobalStrategy,
    AdaptiveStrategy,
    SimpleOptimizationSolver,
    FibrationSearchSolver,
    HillClimbingFibrationSolver # Importar el nuevo solver
)
from benchmarks.circuit_design_problem import create_circuit_design_problem


# --- Benchmark --- #
def run_benchmark():
    print("\n--- Benchmark: Flujo de Fibración con Restricciones SOFT (Integrado) ---")
    
    variables, domains, hierarchy = create_circuit_design_problem()
    
    # --- Solver Baseline (SimpleOptimizationSolver sin modulación ni hacificación) ---
    print("\nEjecutando Baseline Solver (SimpleOptimizationSolver)...")
    start_time = time.time()
    baseline_solver = SimpleOptimizationSolver(variables, domains, hierarchy)
    baseline_solver.max_solutions = 1000 # Explorar un número razonable de soluciones
    baseline_solver.find_best_solution = True
    baseline_best_solution = baseline_solver.solve()
    end_time = time.time()
    
    print(f"Tiempo de ejecución (Baseline): {end_time - start_time:.4f}s")
    
    baseline_best_energy = float("inf")
    if baseline_best_solution:
        landscape = EnergyLandscapeOptimized(hierarchy)
        baseline_best_energy = landscape.compute_energy(baseline_best_solution).total_energy
    
    print(f"Mejor energía total (Baseline): {baseline_best_energy:.3f}")
    # print(f"Mejor solución (Baseline): {baseline_best_solution}")

    # --- Flujo de Fibración con Modulación Adaptativa e Hacificación (FibrationSearchSolver) ---
    print("\nEjecutando Flujo de Fibración Integrado (FibrationSearchSolver)...")
    start_time = time.time()
    
    fibration_solver = FibrationSearchSolver(variables, domains, hierarchy)
    fibration_solver.max_solutions = 10 # Buscar las 10 mejores soluciones
    fibration_solver.max_iterations = 5000 # Aumentar iteraciones para exploración
    fibration_solver.max_backtracks = 20000 # Aumentar límite de retrocesos
    
    fibration_best_solution = fibration_solver.solve()
    end_time = time.time()
    
    print(f"Tiempo de ejecución (Fibración Integrado): {end_time - start_time:.4f}s")
    
    fibration_best_energy = float("inf")
    if fibration_best_solution:
        fibration_best_energy = fibration_solver.landscape.compute_energy(fibration_best_solution).total_energy
                
    print(f"Mejor energía total (Fibración Integrado): {fibration_best_energy:.3f}")
    # print(f"Mejor solución (Fibración Integrado): {fibration_best_solution}")
    print(f"Estadísticas del FibrationSearchSolver: {fibration_solver.get_statistics()}")

    # --- Hill Climbing Fibration Solver ---
    print("\nEjecutando Hill Climbing Fibration Solver...")
    hill_climbing_solver = HillClimbingFibrationSolver(variables, domains, hierarchy, max_iterations=1000, num_restarts=20)
    start_time = time.time()
    hill_climbing_solution = hill_climbing_solver.solve()
    end_time = time.time()
    hill_climbing_time = end_time - start_time
    hill_climbing_energy = float("inf")
    if hill_climbing_solution:
        landscape = EnergyLandscapeOptimized(hierarchy)
        hill_climbing_energy = landscape.compute_energy(hill_climbing_solution).total_energy

    print(f"Tiempo de ejecución (Hill Climbing): {hill_climbing_time:.4f}s")
    print(f"Mejor energía total (Hill Climbing): {hill_climbing_energy:.3f}")
    # print(f"Mejor solución (Hill Climbing): {hill_climbing_solution}")

    # --- Comparación de resultados ---
    print("\n--- Comparación Final ---")
    print(f"Baseline - Mejor Energía: {baseline_best_energy:.3f}")
    print(f"Fibración Search - Mejor Energía: {fibration_best_energy:.3f}")
    print(f"Hill Climbing - Mejor Energía: {hill_climbing_energy:.3f}")

    best_overall_energy = min(baseline_best_energy, fibration_best_energy, hill_climbing_energy)

    if best_overall_energy == baseline_best_energy:
        print("El Baseline Solver encontró una solución de mejor calidad.")
    elif best_overall_energy == fibration_best_energy:
        print("El Fibration Search Solver encontró una solución de mejor calidad.")
    else:
        print("El Hill Climbing Fibration Solver encontró una solución de mejor calidad.")

    # Calcular la mejora porcentual en la energía (reducción de violaciones SOFT)
    if baseline_best_energy != 0:
        improvement_fibration_search = ((baseline_best_energy - fibration_best_energy) / baseline_best_energy) * 100
        improvement_hill_climbing = ((baseline_best_energy - hill_climbing_energy) / baseline_best_energy) * 100
        print(f"Mejora en la calidad de la solución (Fibración Search vs Baseline): {improvement_fibration_search:.2f}%")
        print(f"Mejora en la calidad de la solución (Hill Climbing vs Baseline): {improvement_hill_climbing:.2f}%")
    else:
        print("No se puede calcular la mejora porcentual si la energía base es 0.")

if __name__ == "__main__":
    run_benchmark()
