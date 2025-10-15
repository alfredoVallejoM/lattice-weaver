"""
Benchmark para Fase 3: Integración FCA

Este script compara el rendimiento de diferentes estrategias de selección
de variables, incluyendo las nuevas estrategias guiadas por FCA.

Autor: Manus AI
Fecha: 15 de Octubre, 2025
"""

import sys
import time
from typing import Dict, List, Any
from dataclasses import dataclass

# Añadir el directorio raíz al path
sys.path.insert(0, '/home/ubuntu/lattice-weaver')

from lattice_weaver.core.csp_problem import CSP, Constraint
from lattice_weaver.core.csp_engine.solver import CSPSolver
from lattice_weaver.core.csp_engine.strategies.fca_guided import (
    FCAGuidedSelector,
    FCAOnlySelector,
    FCAClusterSelector
)


@dataclass
class BenchmarkResult:
    """Resultado de un benchmark."""
    problem_name: str
    strategy_name: str
    solved: bool
    num_solutions: int
    nodes_explored: int
    backtracks: int
    time_seconds: float
    fca_analysis_time: float = 0.0


def create_nqueens(n: int) -> CSP:
    """Crea un problema N-Queens."""
    variables = [f'Q{i}' for i in range(n)]
    domains = {var: frozenset(range(n)) for var in variables}
    
    constraints = []
    for i in range(n):
        for j in range(i + 1, n):
            # No misma columna
            constraints.append(
                Constraint(
                    scope=frozenset([f'Q{i}', f'Q{j}']),
                    relation=lambda vi, vj, i=i, j=j: vi != vj
                )
            )
            # No misma diagonal
            constraints.append(
                Constraint(
                    scope=frozenset([f'Q{i}', f'Q{j}']),
                    relation=lambda vi, vj, i=i, j=j: abs(vi - vj) != abs(i - j)
                )
            )
    
    return CSP(variables=frozenset(variables), domains=domains, constraints=constraints)


def create_graph_coloring(num_nodes: int, edges: List[tuple], num_colors: int = 3) -> CSP:
    """Crea un problema de coloreo de grafos."""
    variables = [f'V{i}' for i in range(num_nodes)]
    domains = {var: frozenset(range(num_colors)) for var in variables}
    
    constraints = []
    for i, j in edges:
        constraints.append(
            Constraint(
                scope=frozenset([f'V{i}', f'V{j}']),
                relation=lambda vi, vj: vi != vj
            )
        )
    
    return CSP(variables=frozenset(variables), domains=domains, constraints=constraints)


def create_sudoku_4x4() -> CSP:
    """Crea un Sudoku 4x4 simplificado."""
    variables = [f'C{i}{j}' for i in range(4) for j in range(4)]
    domains = {var: frozenset([1, 2, 3, 4]) for var in variables}
    
    constraints = []
    
    # Restricciones de fila
    for i in range(4):
        for j1 in range(4):
            for j2 in range(j1 + 1, 4):
                constraints.append(
                    Constraint(
                        scope=frozenset([f'C{i}{j1}', f'C{i}{j2}']),
                        relation=lambda v1, v2: v1 != v2
                    )
                )
    
    # Restricciones de columna
    for j in range(4):
        for i1 in range(4):
            for i2 in range(i1 + 1, 4):
                constraints.append(
                    Constraint(
                        scope=frozenset([f'C{i1}{j}', f'C{i2}{j}']),
                        relation=lambda v1, v2: v1 != v2
                    )
                )
    
    # Restricciones de bloque 2x2
    for block_row in range(2):
        for block_col in range(2):
            cells = []
            for i in range(2):
                for j in range(2):
                    cells.append(f'C{block_row*2 + i}{block_col*2 + j}')
            
            for c1_idx in range(len(cells)):
                for c2_idx in range(c1_idx + 1, len(cells)):
                    constraints.append(
                        Constraint(
                            scope=frozenset([cells[c1_idx], cells[c2_idx]]),
                            relation=lambda v1, v2: v1 != v2
                        )
                    )
    
    return CSP(variables=frozenset(variables), domains=domains, constraints=constraints)


def run_benchmark(csp: CSP, problem_name: str, strategy_name: str, selector) -> BenchmarkResult:
    """Ejecuta un benchmark con una estrategia específica."""
    try:
        # Medir tiempo de análisis FCA (si aplica)
        fca_time = 0.0
        if 'FCA' in strategy_name:
            fca_start = time.time()
            # El análisis FCA se realiza en la primera llamada a select()
            # Lo medimos ejecutando una selección dummy
            from lattice_weaver.core.csp_engine.fca_analyzer import FCAAnalyzer
            analyzer = FCAAnalyzer(csp)
            analyzer.analyze()
            fca_time = time.time() - fca_start
        
        # Crear solver con la estrategia
        solver = CSPSolver(csp)
        
        # Reemplazar el método de selección si es necesario
        if selector:
            def new_select(current_domains):
                # Adaptar la API: el selector necesita assignment, pero el solver no lo pasa
                # Usamos solver.assignment que es accesible
                return selector.select(csp, solver.assignment, current_domains)
            
            solver._select_unassigned_variable = new_select
        
        # Resolver
        start_time = time.time()
        stats = solver.solve(all_solutions=False, max_solutions=1)
        end_time = time.time()
        
        return BenchmarkResult(
            problem_name=problem_name,
            strategy_name=strategy_name,
            solved=len(stats.solutions) > 0,
            num_solutions=len(stats.solutions),
            nodes_explored=stats.nodes_explored,
            backtracks=stats.backtracks,
            time_seconds=end_time - start_time,
            fca_analysis_time=fca_time
        )
    
    except Exception as e:
        print(f"Error en {problem_name} con {strategy_name}: {e}")
        return BenchmarkResult(
            problem_name=problem_name,
            strategy_name=strategy_name,
            solved=False,
            num_solutions=0,
            nodes_explored=0,
            backtracks=0,
            time_seconds=0.0
        )


def main():
    """Ejecuta todos los benchmarks."""
    print("=" * 80)
    print("BENCHMARK FASE 3: INTEGRACIÓN FCA")
    print("=" * 80)
    print()
    
    # Definir problemas de prueba
    problems = [
        ("N-Queens 4x4", create_nqueens(4)),
        ("N-Queens 6x6", create_nqueens(6)),
        ("N-Queens 8x8", create_nqueens(8)),
        ("Graph Coloring (6 nodes, cycle)", create_graph_coloring(6, [(0,1), (1,2), (2,3), (3,4), (4,5), (5,0)])),
        ("Graph Coloring (8 nodes, complete)", create_graph_coloring(8, [(i,j) for i in range(8) for j in range(i+1, 8)], num_colors=8)),
        ("Sudoku 4x4", create_sudoku_4x4()),
    ]
    
    # Definir estrategias
    strategies = [
        ("Baseline (First Unassigned)", None),
        ("FCA-Guided (MRV + FCA)", FCAGuidedSelector()),
        ("FCA-Only", FCAOnlySelector()),
        ("FCA-Cluster", FCAClusterSelector()),
    ]
    
    # Ejecutar benchmarks
    results = []
    total_benchmarks = len(problems) * len(strategies)
    current = 0
    
    for problem_name, csp in problems:
        print(f"\n{'='*80}")
        print(f"Problema: {problem_name}")
        print(f"{'='*80}")
        
        for strategy_name, selector in strategies:
            current += 1
            print(f"[{current}/{total_benchmarks}] Ejecutando {strategy_name}...", end=" ")
            
            result = run_benchmark(csp, problem_name, strategy_name, selector)
            results.append(result)
            
            if result.solved:
                print(f"✓ Resuelto en {result.time_seconds:.4f}s "
                      f"({result.nodes_explored} nodos, {result.backtracks} backtracks)")
                if result.fca_analysis_time > 0:
                    print(f"  └─ Análisis FCA: {result.fca_analysis_time:.4f}s")
            else:
                print("✗ No resuelto")
    
    # Generar resumen
    print("\n" + "=" * 80)
    print("RESUMEN DE RESULTADOS")
    print("=" * 80)
    
    # Agrupar por estrategia
    strategy_stats = {}
    for result in results:
        if result.strategy_name not in strategy_stats:
            strategy_stats[result.strategy_name] = {
                'solved': 0,
                'total': 0,
                'total_nodes': 0,
                'total_backtracks': 0,
                'total_time': 0.0,
                'total_fca_time': 0.0,
            }
        
        stats = strategy_stats[result.strategy_name]
        stats['total'] += 1
        if result.solved:
            stats['solved'] += 1
            stats['total_nodes'] += result.nodes_explored
            stats['total_backtracks'] += result.backtracks
            stats['total_time'] += result.time_seconds
            stats['total_fca_time'] += result.fca_analysis_time
    
    # Imprimir tabla resumen
    print()
    print(f"{'Estrategia':<40} {'Resueltos':<12} {'Avg Nodos':<12} {'Avg Time (s)':<15}")
    print("-" * 80)
    
    for strategy_name, stats in strategy_stats.items():
        solved_count = stats['solved']
        total_count = stats['total']
        avg_nodes = stats['total_nodes'] / solved_count if solved_count > 0 else 0
        avg_time = stats['total_time'] / solved_count if solved_count > 0 else 0
        
        print(f"{strategy_name:<40} {solved_count}/{total_count:<10} "
              f"{avg_nodes:<12.1f} {avg_time:<15.4f}")
    
    # Comparación detallada
    print("\n" + "=" * 80)
    print("COMPARACIÓN DETALLADA POR PROBLEMA")
    print("=" * 80)
    
    for problem_name, _ in problems:
        print(f"\n{problem_name}:")
        problem_results = [r for r in results if r.problem_name == problem_name]
        
        if not problem_results:
            continue
        
        print(f"  {'Estrategia':<35} {'Nodos':<10} {'Backtracks':<12} {'Tiempo (s)':<12}")
        print("  " + "-" * 70)
        
        for result in problem_results:
            if result.solved:
                print(f"  {result.strategy_name:<35} "
                      f"{result.nodes_explored:<10} "
                      f"{result.backtracks:<12} "
                      f"{result.time_seconds:<12.4f}")
            else:
                print(f"  {result.strategy_name:<35} {'NO RESUELTO':<10}")
    
    # Análisis de mejora
    print("\n" + "=" * 80)
    print("ANÁLISIS DE MEJORA (vs Baseline)")
    print("=" * 80)
    
    baseline_stats = strategy_stats.get("Baseline (First Unassigned)", {})
    baseline_avg_nodes = baseline_stats.get('total_nodes', 0) / max(baseline_stats.get('solved', 1), 1)
    baseline_avg_time = baseline_stats.get('total_time', 0) / max(baseline_stats.get('solved', 1), 1)
    
    print()
    print(f"{'Estrategia':<40} {'Mejora Nodos':<15} {'Mejora Tiempo':<15}")
    print("-" * 80)
    
    for strategy_name, stats in strategy_stats.items():
        if strategy_name == "Baseline (First Unassigned)":
            continue
        
        solved_count = stats['solved']
        if solved_count == 0:
            continue
        
        avg_nodes = stats['total_nodes'] / solved_count
        avg_time = stats['total_time'] / solved_count
        
        nodes_improvement = ((baseline_avg_nodes - avg_nodes) / baseline_avg_nodes * 100) if baseline_avg_nodes > 0 else 0
        time_improvement = ((baseline_avg_time - avg_time) / baseline_avg_time * 100) if baseline_avg_time > 0 else 0
        
        print(f"{strategy_name:<40} {nodes_improvement:>13.1f}% {time_improvement:>13.1f}%")
    
    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETADO")
    print("=" * 80)


if __name__ == '__main__':
    main()

