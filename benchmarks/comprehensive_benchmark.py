import time
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

# Solvers del Flujo de Fibración
from lattice_weaver.fibration.simple_optimization_solver import SimpleOptimizationSolver
from lattice_weaver.fibration.fibration_search_solver import FibrationSearchSolver
from lattice_weaver.fibration.hill_climbing_solver import HillClimbingFibrationSolver
from lattice_weaver.fibration.constraint_hierarchy import ConstraintHierarchy, ConstraintLevel, Hardness
from lattice_weaver.fibration.energy_landscape_optimized import EnergyLandscapeOptimized

# Problemas de benchmark
from benchmarks.nqueens_problem import create_nqueens_problem
from benchmarks.circuit_design_problem import create_circuit_design_problem

# Solvers externos
from external_solvers.nqueens_python_constraint import solve_nqueens_python_constraint
from external_solvers.nqueens_ortools_cpsat import solve_nqueens_ortools_cpsat
from external_solvers.circuit_design_pymoo import solve_circuit_design_pymoo

class BenchmarkRunner:
    def __init__(self):
        self.results = []

    def run_benchmark(self, name: str, solver_func, *args, **kwargs):
        print(f"Ejecutando {name}...")
        start_time = time.time()
        solution = solver_func(*args, **kwargs)
        end_time = time.time()
        exec_time = end_time - start_time

        energy = float('inf')
        if solution and 'hierarchy' in kwargs:
            hierarchy = kwargs['hierarchy']
            landscape = EnergyLandscapeOptimized(hierarchy)
            energy_result = landscape.compute_energy(solution)
            energy = energy_result.total_energy
        elif solution and 'problem_type' in kwargs and kwargs['problem_type'] == 'nqueens':
            # Para N-Queens, la energía es 0 si hay solución, inf si no
            energy = 0.0
        else:
            energy = float('inf') # Si no hay solución o no se puede calcular la energía

        self.results.append({
            'solver': name,
            'time': exec_time,
            'solution': solution,
            'energy': energy
        })
        print(f"  Tiempo: {exec_time:.4f}s, Energía: {energy:.3f}")
        return solution

    def print_summary(self):
        print("\n--- Resumen del Benchmark ---")
        for res in self.results:
            print(f"Solver: {res['solver']}")
            print(f"  Tiempo: {res['time']:.4f}s")
            print(f"  Energía: {res['energy']:.3f}")
            print(f"  Solución encontrada: {'Sí' if res['solution'] else 'No'}")
            print("-" * 20)

def run_nqueens_benchmarks(n: int):
    print(f"\n--- Benchmark N-Queens (n={n}) ---")
    runner = BenchmarkRunner()

    # Problema para solvers de Fibración
    variables, domains, hierarchy = create_nqueens_problem(n)

    # 1. SimpleOptimizationSolver (Baseline Fibración)
    runner.run_benchmark(
        f"Fibration (SimpleOptimizationSolver) N={n}",
        lambda: SimpleOptimizationSolver(variables, domains, hierarchy).solve(),
        hierarchy=hierarchy,
        problem_type='nqueens'
    )

    # 2. FibrationSearchSolver
    runner.run_benchmark(
        f"Fibration (FibrationSearchSolver) N={n}",
        lambda: FibrationSearchSolver(variables, domains, hierarchy).solve(),
        hierarchy=hierarchy,
        problem_type='nqueens'
    )

    # 3. HillClimbingFibrationSolver
    runner.run_benchmark(
        f"Fibration (HillClimbingFibrationSolver) N={n}",
        lambda: HillClimbingFibrationSolver(variables, domains, hierarchy).solve(),
        hierarchy=hierarchy,
        problem_type='nqueens'
    )

    # 4. python-constraint
    runner.run_benchmark(
        f"Python-Constraint N={n}",
        lambda: solve_nqueens_python_constraint(n),
        problem_type='nqueens'
    )

    # 5. OR-Tools CP-SAT
    runner.run_benchmark(
        f"OR-Tools CP-SAT N={n}",
        lambda: solve_nqueens_ortools_cpsat(n),
        problem_type='nqueens'
    )
    runner.print_summary()

def run_circuit_design_benchmarks(n_gates: int, n_chips: int):
    print(f"\n--- Benchmark Diseño de Circuitos (G={n_gates}, C={n_chips}) ---")
    runner = BenchmarkRunner()

    # Problema para solvers de Fibración y Pymoo
    variables, domains, hierarchy = create_circuit_design_problem(n_gates, n_chips)

    # 1. SimpleOptimizationSolver (Baseline Fibración)
    runner.run_benchmark(
        f"Fibration (SimpleOptimizationSolver) G={n_gates}, C={n_chips}",
        lambda: SimpleOptimizationSolver(variables, domains, hierarchy).solve(),
        hierarchy=hierarchy
    )

    # 2. FibrationSearchSolver
    runner.run_benchmark(
        f"Fibration (FibrationSearchSolver) G={n_gates}, C={n_chips}",
        lambda: FibrationSearchSolver(variables, domains, hierarchy).solve(),
        hierarchy=hierarchy
    )

    # 3. HillClimbingFibrationSolver
    runner.run_benchmark(
        f"Fibration (HillClimbingFibrationSolver) G={n_gates}, C={n_chips}",
        lambda: HillClimbingFibrationSolver(variables, domains, hierarchy).solve(),
        hierarchy=hierarchy
    )

    # 4. Pymoo
    runner.run_benchmark(
        f"Pymoo G={n_gates}, C={n_chips}",
        lambda: solve_circuit_design_pymoo(n_gates, n_chips, hierarchy),
        hierarchy=hierarchy
    )
    runner.print_summary()

if __name__ == "__main__":
    # Benchmarks N-Queens
    run_nqueens_benchmarks(4)
    run_nqueens_benchmarks(8)
    run_nqueens_benchmarks(12)

    # Benchmarks Diseño de Circuitos
    run_circuit_design_benchmarks(5, 2)
    run_circuit_design_benchmarks(8, 3)

            hierarchy = kwargs['hierarchy']
            landscape = EnergyLandscapeOptimized(hierarchy)
            energy_result = landscape.compute_energy(solution)
            energy = energy_result.total_energy

        self.results.append({
            'solver': name,
            'time': exec_time,
            'solution': solution,
            'energy': energy
        })
        print(f"  Tiempo: {exec_time:.4f}s, Energía: {energy:.3f}")
        return solution

    def print_summary(self):
        print("\n--- Resumen del Benchmark ---")
        for res in self.results:
            print(f"Solver: {res['solver']}")
            print(f"  Tiempo: {res['time']:.4f}s")
            print(f"  Energía: {res['energy']:.3f}")
            print(f"  Solución encontrada: {'Sí' if res['solution'] else 'No'}")
            print("-" * 20)

def run_nqueens_benchmarks(n: int):
    print(f"\n--- Benchmark N-Queens (n={n}) ---")
    runner = BenchmarkRunner()

    # Problema para solvers de Fibración
    variables, domains, hierarchy = create_nqueens_problem(n)

    # 1. SimpleOptimizationSolver (Baseline Fibración)
    runner.run_benchmark(
        f"Fibration (SimpleOptimizationSolver) N={n}",
        lambda: SimpleOptimizationSolver(variables, domains, hierarchy).solve(),
        hierarchy=hierarchy
    )

    # 2. FibrationSearchSolver
    runner.run_benchmark(
        f"Fibration (FibrationSearchSolver) N={n}",
        lambda: FibrationSearchSolver(variables, domains, hierarchy).solve(),
        hierarchy=hierarchy
    )

    # 3. HillClimbingFibrationSolver
    runner.run_benchmark(
        f"Fibration (HillClimbingFibrationSolver) N={n}",
        lambda: HillClimbingFibrationSolver(variables, domains, hierarchy).solve(),
        hierarchy=hierarchy
    )

    # 4. python-constraint
    runner.run_benchmark(
        f"Python-Constraint N={n}",
        lambda: solve_nqueens_python_constraint(n)
    )

    # 5. OR-Tools CP-SAT
    runner.run_benchmark(
        f"OR-Tools CP-SAT N={n}",
        lambda: solve_nqueens_ortools_cpsat(n)
    )
    runner.print_summary()

def run_circuit_design_benchmarks(n_gates: int, n_chips: int):
    print(f"\n--- Benchmark Diseño de Circuitos ({n_gates} compuertas, {n_chips} chips) ---")
    runner = BenchmarkRunner()

    # Problema para solvers de Fibración y Pymoo
    variables, domains, hierarchy = create_circuit_design_problem(n_gates, n_chips)

    # 1. SimpleOptimizationSolver (Baseline Fibración)
    runner.run_benchmark(
        f"Fibration (SimpleOptimizationSolver) G={n_gates}, C={n_chips}",
        lambda: SimpleOptimizationSolver(variables, domains, hierarchy).solve(),
        hierarchy=hierarchy
    )

    # 2. FibrationSearchSolver
    runner.run_benchmark(
        f"Fibration (FibrationSearchSolver) G={n_gates}, C={n_chips}",
        lambda: FibrationSearchSolver(variables, domains, hierarchy).solve(),
        hierarchy=hierarchy
    )

    # 3. HillClimbingFibrationSolver
    runner.run_benchmark(
        f"Fibration (HillClimbingFibrationSolver) G={n_gates}, C={n_chips}",
        lambda: HillClimbingFibrationSolver(variables, domains, hierarchy).solve(),
        hierarchy=hierarchy
    )

    # 4. Pymoo
    runner.run_benchmark(
        f"Pymoo G={n_gates}, C={n_chips}",
        lambda: solve_circuit_design_pymoo(n_gates, n_chips, hierarchy),
        hierarchy=hierarchy
    )
    runner.print_summary()

if __name__ == "__main__":
    # Benchmarks N-Queens
    run_nqueens_benchmarks(4)
    run_nqueens_benchmarks(8)
    run_nqueens_benchmarks(12)

    # Benchmarks Diseño de Circuitos
    run_circuit_design_benchmarks(5, 2)
    run_circuit_design_benchmarks(8, 3)

