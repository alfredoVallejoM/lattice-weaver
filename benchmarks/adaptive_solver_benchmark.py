"""
Benchmark del Solver Adaptativo

Compara el rendimiento del solver adaptativo vs el solver enhanced en diferentes
tipos de problemas para validar las mejoras de rendimiento.

Autor: Agente Autónomo - Lattice Weaver
Fecha: 15 de Octubre, 2025
"""

import time
import logging
from typing import Dict, List, Any, Tuple

# Desactivar logging para benchmarks limpios
logging.disable(logging.CRITICAL)

from lattice_weaver.fibration.fibration_search_solver_enhanced import FibrationSearchSolverEnhanced
from lattice_weaver.fibration.fibration_search_solver_adaptive import FibrationSearchSolverAdaptive, SolverMode
from lattice_weaver.fibration.constraint_hierarchy import ConstraintHierarchy, Hardness
from lattice_weaver.fibration.energy_landscape_optimized import EnergyLandscapeOptimized
from lattice_weaver.arc_engine.core import ArcEngine


def create_nqueens_problem(n: int):
    """Crea problema N-Queens."""
    hierarchy = ConstraintHierarchy()
    variables = [f"Q{i}" for i in range(n)]
    domains = {var: list(range(n)) for var in variables}
    
    for i in range(n):
        for j in range(i + 1, n):
            # No misma columna
            def ne_constraint(assignment, i=i, j=j):
                qi, qj = f"Q{i}", f"Q{j}"
                if qi in assignment and qj in assignment:
                    return assignment[qi] != assignment[qj]
                return True
            
            hierarchy.add_local_constraint(
                var1=f"Q{i}", var2=f"Q{j}",
                predicate=ne_constraint,
                hardness=Hardness.HARD,
                metadata={"name": f"Q{i}_ne_Q{j}"}
            )
            
            # No misma diagonal
            def no_diagonal(assignment, i=i, j=j):
                qi, qj = f"Q{i}", f"Q{j}"
                if qi in assignment and qj in assignment:
                    return abs(assignment[qi] - assignment[qj]) != abs(i - j)
                return True
            
            hierarchy.add_local_constraint(
                var1=f"Q{i}", var2=f"Q{j}",
                predicate=no_diagonal,
                hardness=Hardness.HARD,
                metadata={"name": f"Q{i}_nodiag_Q{j}"}
            )
    
    return hierarchy, variables, domains


def benchmark_solver(solver_name: str, solver, n: int) -> Dict[str, Any]:
    """Ejecuta benchmark de un solver."""
    start_time = time.time()
    solution = solver.solve()
    elapsed = time.time() - start_time
    
    stats = solver.get_statistics()
    
    return {
        'solver': solver_name,
        'n': n,
        'time': elapsed,
        'solution_found': solution is not None,
        'backtracks': stats['search']['backtracks'],
        'nodes': stats['search']['nodes_explored'],
        'mode': stats.get('mode', 'N/A')
    }


def main():
    """Función principal de benchmarking."""
    print("=" * 100)
    print("BENCHMARK: SOLVER ADAPTATIVO VS SOLVER ENHANCED")
    print("=" * 100)
    print()
    
    results = []
    
    # Benchmark en diferentes tamaños
    for n in [4, 6, 8]:
        print(f"\n{n}-Queens:")
        print("-" * 100)
        
        # Crear problema
        hierarchy, variables, domains = create_nqueens_problem(n)
        landscape = EnergyLandscapeOptimized(hierarchy)
        
        # 1. Solver Enhanced (baseline)
        arc_engine_enhanced = ArcEngine(use_tms=True, parallel=False)
        solver_enhanced = FibrationSearchSolverEnhanced(
            hierarchy=hierarchy,
            landscape=landscape,
            arc_engine=arc_engine_enhanced,
            variables=variables,
            domains=domains,
            use_homotopy=True,
            use_tms=True,
            use_enhanced_heuristics=True,
            max_backtracks=10000,
            max_iterations=10000,
            time_limit_seconds=30.0
        )
        
        result_enhanced = benchmark_solver("Enhanced (FULL)", solver_enhanced, n)
        results.append(result_enhanced)
        print(f"  Enhanced (FULL):     {result_enhanced['time']:>7.3f}s  "
              f"Backtracks: {result_enhanced['backtracks']:>6}  "
              f"Nodos: {result_enhanced['nodes']:>6}")
        
        # 2. Solver Adaptive (auto-detect)
        arc_engine_adaptive = ArcEngine(use_tms=True, parallel=False)
        solver_adaptive = FibrationSearchSolverAdaptive(
            hierarchy=hierarchy,
            landscape=landscape,
            arc_engine=arc_engine_adaptive,
            variables=variables,
            domains=domains,
            mode=None,  # Auto-detect
            max_backtracks=10000,
            max_iterations=10000,
            time_limit_seconds=30.0
        )
        
        result_adaptive = benchmark_solver("Adaptive (AUTO)", solver_adaptive, n)
        results.append(result_adaptive)
        print(f"  Adaptive (AUTO):     {result_adaptive['time']:>7.3f}s  "
              f"Backtracks: {result_adaptive['backtracks']:>6}  "
              f"Nodos: {result_adaptive['nodes']:>6}  "
              f"Modo: {result_adaptive['mode']}")
        
        # 3. Solver Adaptive (LITE forzado)
        arc_engine_lite = ArcEngine(use_tms=False, parallel=False)
        solver_lite = FibrationSearchSolverAdaptive(
            hierarchy=hierarchy,
            landscape=landscape,
            arc_engine=arc_engine_lite,
            variables=variables,
            domains=domains,
            mode=SolverMode.LITE,
            max_backtracks=10000,
            max_iterations=10000,
            time_limit_seconds=30.0
        )
        
        result_lite = benchmark_solver("Adaptive (LITE)", solver_lite, n)
        results.append(result_lite)
        print(f"  Adaptive (LITE):     {result_lite['time']:>7.3f}s  "
              f"Backtracks: {result_lite['backtracks']:>6}  "
              f"Nodos: {result_lite['nodes']:>6}")
        
        # Calcular speedup
        speedup_auto = result_enhanced['time'] / result_adaptive['time'] if result_adaptive['time'] > 0 else float('inf')
        speedup_lite = result_enhanced['time'] / result_lite['time'] if result_lite['time'] > 0 else float('inf')
        
        print(f"\n  Speedup AUTO vs FULL: {speedup_auto:.2f}x")
        print(f"  Speedup LITE vs FULL: {speedup_lite:.2f}x")
    
    # Resumen final
    print("\n" + "=" * 100)
    print("RESUMEN")
    print("=" * 100)
    print(f"{'Solver':<25} {'Problema':<15} {'Tiempo':<12} {'Backtracks':<12} {'Modo':<10}")
    print("-" * 100)
    
    for r in results:
        print(f"{r['solver']:<25} {r['n']}-Queens{'':<7} {r['time']:>8.3f}s   "
              f"{r['backtracks']:>10}   {r['mode']:<10}")
    
    print("=" * 100)
    
    # Calcular mejoras promedio
    enhanced_times = [r['time'] for r in results if 'Enhanced' in r['solver']]
    adaptive_times = [r['time'] for r in results if 'AUTO' in r['solver']]
    lite_times = [r['time'] for r in results if 'LITE' in r['solver']]
    
    avg_speedup_auto = sum(e/a for e, a in zip(enhanced_times, adaptive_times)) / len(enhanced_times)
    avg_speedup_lite = sum(e/l for e, l in zip(enhanced_times, lite_times)) / len(enhanced_times)
    
    print(f"\nSpeedup promedio AUTO vs FULL: {avg_speedup_auto:.2f}x")
    print(f"Speedup promedio LITE vs FULL: {avg_speedup_lite:.2f}x")
    print()


if __name__ == "__main__":
    main()

