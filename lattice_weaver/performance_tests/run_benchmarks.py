import time
import json
import numpy as np

from ..fibration.constraint_hierarchy import ConstraintHierarchy, Hardness
from ..external_solvers.python_constraint_adapter import PythonConstraintAdapter
from ..external_solvers.ortools_cpsat_adapter import ORToolsCPSATAdapter
from ..external_solvers.pymoo_adapter import PymooAdapter
from .test_cases import get_test_cases
from ..external_solvers.fibration_flow_adapter import FibrationFlowAdapter

def run_benchmarks(output_file="/home/ubuntu/lattice-weaver/lattice_weaver/performance_tests/benchmark_results.json", solver_timeout=60):
    print("\n--- Ejecutando benchmarks de rendimiento ---")
    test_cases = get_test_cases()
    results = []
    # Cargar resultados existentes si el archivo ya existe
    try:
        with open(output_file, "r") as f:
            results = json.load(f)
        print(f"Cargados {len(results)} resultados existentes de {output_file}")
    except (FileNotFoundError, json.JSONDecodeError):
        print("No se encontraron resultados existentes o el archivo está vacío. Iniciando desde cero.")

    for i, test_case in enumerate(test_cases):
        # Saltar casos de prueba ya procesados
        if any(tc["test_case_name"] == test_case["name"] for tc in results):
            print(f"Saltando caso de prueba {test_case['name']} (ya procesado).")
            continue
        print(f'\nEjecutando caso de prueba {i+1}/{len(test_cases)}: {test_case["name"]}')
        case_results = {"test_case_name": test_case["name"], "solvers": {}}

        variables = test_case["variables"]
        domains = test_case["domains"]
        hierarchy = test_case["hierarchy"]

        # --- PythonConstraintAdapter ---
        print("  Probando PythonConstraintAdapter...")
        start_time = time.time()
        pc_adapter = PythonConstraintAdapter(variables, domains, hierarchy)
        pc_solutions = []
        try:
            solutions_found = pc_adapter.solve()
            if solutions_found is not None:
                pc_solutions = solutions_found
        except Exception as e:
            print(f"    Error en PythonConstraintAdapter: {e}")
        end_time = time.time()
        pc_time = end_time - start_time
        pc_found_solution = pc_solutions is not None and len(pc_solutions) > 0
        pc_solution_count = len(pc_solutions) if pc_solutions is not None else 0
        
        pc_valid_solutions = []
        if pc_solutions:
            pc_valid_solutions = pc_solutions

        case_results["solvers"]["python_constraint"] = {
            "time_seconds": pc_time,
            "found_solution": pc_found_solution,
            "solution_count": pc_solution_count,
            "valid_hard_solutions_count": len(pc_valid_solutions),
            "first_solution": pc_solutions[0] if pc_found_solution else None
        }
        print(f"    Tiempo: {pc_time:.4f}s, Soluciones encontradas: {pc_solution_count}")

        # --- FibrationFlowAdapter ---
        print("  Probando FibrationFlowAdapter...")
        start_time = time.time()
        fibration_flow_adapter = FibrationFlowAdapter(variables, domains, hierarchy)
        fibration_flow_solutions = None
        try:
            fibration_flow_solutions = fibration_flow_adapter.solve(time_limit_seconds=solver_timeout)
        except Exception as e:
            print(f"    Error en FibrationFlowAdapter: {e}")
        end_time = time.time()
        fibration_flow_time = end_time - start_time
        fibration_flow_found_solution = fibration_flow_solutions is not None and len(fibration_flow_solutions) > 0
        fibration_flow_objective_value = None
        fibration_flow_hard_violations = None

        if fibration_flow_found_solution:
            # Calcular violaciones SOFT y HARD para la primera solución encontrada
            first_solution = fibration_flow_solutions[0]
            soft_violations = 0
            hard_violations_count = 0
            for level in [level_enum for level_enum in hierarchy.constraints.keys()]:
                for constraint in hierarchy.get_constraints_at_level(level):
                    if constraint.hardness == Hardness.SOFT:
                        if not constraint.predicate(first_solution):
                            soft_violations += 1
                    elif constraint.hardness == Hardness.HARD:
                        if not constraint.predicate(first_solution):
                            hard_violations_count += 1
            fibration_flow_objective_value = soft_violations
            fibration_flow_hard_violations = hard_violations_count

        case_results["solvers"]["fibration_flow"] = {
            "time_seconds": fibration_flow_time,
            "found_solution": fibration_flow_found_solution,
            "solution": fibration_flow_solutions[0] if fibration_flow_found_solution else None,
            "objective_value": fibration_flow_objective_value,
            "hard_violations_count": fibration_flow_hard_violations
        }
        print(f"    Tiempo: {fibration_flow_time:.4f}s, Solución encontrada: {fibration_flow_found_solution}, Objetivo (violaciones SOFT): {fibration_flow_objective_value}, Violaciones HARD: {fibration_flow_hard_violations}")

        # --- ORToolsCPSATAdapter ---
        print("  Probando ORToolsCPSATAdapter...")
        start_time = time.time()
        ortools_adapter = ORToolsCPSATAdapter(variables, domains, hierarchy)
        ortools_solution = None
        try:
            ortools_solution = ortools_adapter.solve(time_limit_seconds=solver_timeout)
        except Exception as e:
            print(f"    Error en ORToolsCPSATAdapter: {e}")
        end_time = time.time()
        ortools_time = end_time - start_time
        ortools_found_solution = ortools_solution is not None
        ortools_objective_value = None
        if ortools_found_solution:
            soft_violations = 0
            for level in [level_enum for level_enum in hierarchy.constraints.keys()]:
                for constraint in hierarchy.get_constraints_at_level(level):
                    if constraint.hardness == Hardness.SOFT:
                        if constraint.predicate(ortools_solution) == 1:
                            soft_violations += 1
            ortools_objective_value = soft_violations

        case_results["solvers"]["ortools_cpsat"] = {
            "time_seconds": ortools_time,
            "found_solution": ortools_found_solution,
            "solution": ortools_solution,
            "objective_value": ortools_objective_value
        }
        print(f"    Tiempo: {ortools_time:.4f}s, Solución encontrada: {ortools_found_solution}, Objetivo (violaciones SOFT): {ortools_objective_value}")

        # --- PymooAdapter ---
        print("  Probando PymooAdapter...")
        start_time = time.time()
        pymoo_adapter = PymooAdapter(variables, domains, hierarchy)
        pymoo_solutions = None
        try:
            pymoo_solutions = pymoo_adapter.solve(time_limit_seconds=solver_timeout)
        except Exception as e:
            print(f"    Error en PymooAdapter: {e}")
        end_time = time.time()
        pymoo_time = end_time - start_time
        pymoo_found_solution = pymoo_solutions is not None and len(pymoo_solutions) > 0
        
        res = pymoo_adapter.res
        pymoo_objectives = None
        pymoo_constraints_violation = None

        if pymoo_found_solution:
            if res.F is not None and len(res.F) > 0:
                pymoo_objectives = res.F[0].tolist() if isinstance(res.F[0], np.ndarray) else res.F[0]
            
            if res.G is not None and len(res.G) > 0:
                pymoo_constraints_violation = res.G[0].tolist() if isinstance(res.G[0], np.ndarray) else res.G[0]

        case_results["solvers"]["pymoo"] = {
            "time_seconds": pymoo_time,
            "found_solution": pymoo_found_solution,
            "solutions_count": len(pymoo_solutions) if pymoo_solutions else 0,
            "first_solution": pymoo_solutions[0] if pymoo_found_solution else None,
            "first_solution_objectives": pymoo_objectives,
            "first_solution_hard_constraints_violation": pymoo_constraints_violation
        }
        print(f"    Tiempo: {pymoo_time:.4f}s, Soluciones encontradas: {pymoo_found_solution}, Objetivos (primera solución): {pymoo_objectives}, Violaciones HARD (primera solución): {pymoo_constraints_violation}")

        results.append(case_results)

        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)
        print(f"    Resultados guardados incrementalmente en {output_file}")

    print(f"\nBenchmarks completados. Resultados finales guardados en {output_file}")

if __name__ == "__main__":
   run_benchmarks(solver_timeout=1800)

