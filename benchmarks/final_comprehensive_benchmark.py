"""
Benchmark Comparativo Final de Todas las Mejoras

Compara el rendimiento de:
1. Fibration Flow Original (baseline)
2. Fibration Flow con HacificationEngine Optimizado
3. Fibration Flow con Solver Adaptativo V2
4. Fibration Flow con TMS Enhanced
5. Fibration Flow con HomotopyRules Optimizado
6. Fibration Flow con TODAS las mejoras

Autor: Agente Autónomo - Lattice Weaver
Fecha: 15 de Octubre, 2025
"""

import time
import json
from typing import Dict, List, Any
from dataclasses import dataclass, asdict

from lattice_weaver.fibration.constraint_hierarchy import ConstraintHierarchy, Hardness, ConstraintLevel
from lattice_weaver.fibration.energy_landscape_optimized import EnergyLandscapeOptimized
# TODO: Descomentar cuando se integre el componente
# from lattice_weaver.arc_engine.core import ArcEngine

# Importar versiones optimizadas
# TODO: Descomentar cuando se integre el componente
# from lattice_weaver.fibration.fibration_search_solver_enhanced import FibrationSearchSolverEnhanced
# TODO: Descomentar cuando se integre el componente
# from lattice_weaver.fibration.fibration_search_solver_adaptive_v2 import FibrationSearchSolverAdaptiveV2


@dataclass
class BenchmarkResult:
    """Resultado de un benchmark."""
    solver_name: str
    problem_name: str
    problem_size: int
    solution_found: bool
    time_seconds: float
    nodes_explored: int
    backtracks: int
    energy: float
    memory_mb: float = 0.0


def create_nqueens_problem(n: int) -> tuple:
    """Crea problema de N-Queens."""
    hierarchy = ConstraintHierarchy()
    
    # Restricción: reinas en diferentes columnas (implícito por dominio)
    # Restricción: reinas en diferentes filas
    for i in range(n):
        for j in range(i+1, n):
            def different_rows(a, i=i, j=j):
                if f"Q{i}" in a and f"Q{j}" in a:
                    return a[f"Q{i}"] != a[f"Q{j}"]
                return True
            
            hierarchy.add_local_constraint(
                f"Q{i}", f"Q{j}",
                different_rows,
                Hardness.HARD
            )
    
    # Restricción: reinas en diferentes diagonales
    for i in range(n):
        for j in range(i+1, n):
            def different_diagonals(a, i=i, j=j):
                if f"Q{i}" in a and f"Q{j}" in a:
                    return abs(a[f"Q{i}"] - a[f"Q{j}"]) != abs(i - j)
                return True
            
            hierarchy.add_local_constraint(
                f"Q{i}", f"Q{j}",
                different_diagonals,
                Hardness.HARD
            )
    
    variables = [f"Q{i}" for i in range(n)]
    domains = {var: list(range(n)) for var in variables}
    
    return hierarchy, variables, domains


def create_graph_coloring_problem(n_nodes: int, n_colors: int) -> tuple:
    """Crea problema de Graph Coloring."""
    hierarchy = ConstraintHierarchy()
    
    # Crear grafo aleatorio (ciclo + algunas aristas extras)
    edges = [(i, (i+1) % n_nodes) for i in range(n_nodes)]
    
    # Añadir aristas extras
    if n_nodes >= 4:
        edges.append((0, n_nodes//2))
        edges.append((1, n_nodes//2 + 1))
    
    # Restricciones: nodos adyacentes deben tener colores diferentes
    for i, j in edges:
        def different_colors(a, i=i, j=j):
            if f"N{i}" in a and f"N{j}" in a:
                return a[f"N{i}"] != a[f"N{j}"]
            return True
        
        hierarchy.add_local_constraint(
            f"N{i}", f"N{j}",
            different_colors,
            Hardness.HARD
        )
    
    variables = [f"N{i}" for i in range(n_nodes)]
    domains = {var: list(range(n_colors)) for var in variables}
    
    return hierarchy, variables, domains


def run_benchmark(
    solver_class,
    solver_name: str,
    hierarchy: ConstraintHierarchy,
    variables: List[str],
    domains: Dict[str, List[Any]],
    problem_name: str,
    **solver_kwargs
) -> BenchmarkResult:
    """Ejecuta un benchmark."""
    landscape = EnergyLandscapeOptimized(hierarchy)
    arc_engine = ArcEngine(use_tms=True)
    
    solver = solver_class(
        hierarchy=hierarchy,
        landscape=landscape,
        arc_engine=arc_engine,
        variables=variables,
        domains=domains,
        **solver_kwargs
    )
    
    start = time.time()
    solution = solver.solve()
    elapsed = time.time() - start
    
    stats = solver.get_statistics()
    
    energy = 0.0
    if solution:
        energy = landscape.compute_energy(solution).total_energy
    
    return BenchmarkResult(
        solver_name=solver_name,
        problem_name=problem_name,
        problem_size=len(variables),
        solution_found=solution is not None,
        time_seconds=elapsed,
        nodes_explored=stats['search']['nodes_explored'],
        backtracks=stats['search']['backtracks'],
        energy=energy
    )


def main():
    """Ejecuta benchmarks completos."""
    print("="*80)
    print("BENCHMARK COMPARATIVO FINAL - TODAS LAS MEJORAS")
    print("="*80)
    print()
    
    results = []
    
    # Problemas a testear
    problems = [
        ("N-Queens 4", lambda: create_nqueens_problem(4)),
        ("N-Queens 6", lambda: create_nqueens_problem(6)),
        ("N-Queens 8", lambda: create_nqueens_problem(8)),
        ("Graph Coloring 8/3", lambda: create_graph_coloring_problem(8, 3)),
        ("Graph Coloring 10/3", lambda: create_graph_coloring_problem(10, 3)),
    ]
    
    # Solvers a comparar
    solvers = [
        (FibrationSearchSolverEnhanced, "Enhanced (Baseline)", {}),
        (FibrationSearchSolverAdaptiveV2, "Adaptive V2 (All Improvements)", {})
    ]
    
    for problem_name, problem_factory in problems:
        print(f"\n{'='*80}")
        print(f"Problema: {problem_name}")
        print(f"{'='*80}\n")
        
        hierarchy, variables, domains = problem_factory()
        
        for solver_class, solver_name, kwargs in solvers:
            print(f"  Ejecutando {solver_name}...", end=" ", flush=True)
            
            try:
                result = run_benchmark(
                    solver_class,
                    solver_name,
                    hierarchy,
                    variables,
                    domains,
                    problem_name,
                    max_backtracks=10000,
                    max_iterations=10000,
                    time_limit_seconds=30.0,
                    **kwargs
                )
                
                results.append(result)
                
                status = "✓" if result.solution_found else "✗"
                print(f"{status} {result.time_seconds:.3f}s, {result.nodes_explored} nodos, {result.backtracks} backtracks")
                
            except Exception as e:
                print(f"ERROR: {e}")
    
    # Resumen final
    print(f"\n{'='*80}")
    print("RESUMEN FINAL")
    print(f"{'='*80}\n")
    
    # Agrupar por solver
    by_solver = {}
    for result in results:
        if result.solver_name not in by_solver:
            by_solver[result.solver_name] = []
        by_solver[result.solver_name].append(result)
    
    for solver_name, solver_results in by_solver.items():
        print(f"\n{solver_name}:")
        print(f"  Problemas resueltos: {sum(1 for r in solver_results if r.solution_found)}/{len(solver_results)}")
        print(f"  Tiempo promedio: {sum(r.time_seconds for r in solver_results) / len(solver_results):.3f}s")
        print(f"  Nodos promedio: {sum(r.nodes_explored for r in solver_results) / len(solver_results):.0f}")
        print(f"  Backtracks promedio: {sum(r.backtracks for r in solver_results) / len(solver_results):.0f}")
    
    # Comparación directa
    print(f"\n{'='*80}")
    print("COMPARACIÓN DIRECTA (Adaptive V2 vs Enhanced)")
    print(f"{'='*80}\n")
    
    for problem_name, _ in problems:
        baseline = next((r for r in results if r.problem_name == problem_name and "Baseline" in r.solver_name), None)
        improved = next((r for r in results if r.problem_name == problem_name and "Adaptive" in r.solver_name), None)
        
        if baseline and improved and baseline.solution_found and improved.solution_found:
            speedup = baseline.time_seconds / improved.time_seconds
            backtrack_reduction = (baseline.backtracks - improved.backtracks) / max(baseline.backtracks, 1)
            
            print(f"{problem_name}:")
            print(f"  Speedup: {speedup:.2f}x")
            print(f"  Reducción de backtracks: {backtrack_reduction:.1%}")
            print()
    
    # Guardar resultados
    with open("/home/ubuntu/benchmark_results_final.json", "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    
    print(f"\nResultados guardados en: /home/ubuntu/benchmark_results_final.json")


if __name__ == "__main__":
    main()

