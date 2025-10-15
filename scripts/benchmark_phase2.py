#!/usr/bin/env python3.11
"""
Benchmark para Fase 2: Sistema de Estrategias Modulares

Este script compara el rendimiento de diferentes combinaciones de estrategias
y valida que no hay regresión respecto al comportamiento por defecto.
"""

import time
from typing import List, Tuple
from lattice_weaver.core.csp_problem import CSP, Constraint
from lattice_weaver.core.csp_engine.solver import CSPSolver
from lattice_weaver.core.csp_engine.strategies import (
    FirstUnassignedSelector,
    MRVSelector,
    DegreeSelector,
    MRVDegreeSelector,
    NaturalOrderer,
    LCVOrderer,
)


# ============================================================================
# Generadores de Problemas
# ============================================================================

def generate_nqueens(n: int) -> CSP:
    """Genera un problema N-Queens de tamaño n."""
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


def generate_graph_coloring(n_nodes: int, n_colors: int, edges: List[Tuple[int, int]]) -> CSP:
    """Genera un problema de coloreo de grafos."""
    variables = [f'N{i}' for i in range(n_nodes)]
    domains = {var: frozenset(range(n_colors)) for var in variables}
    
    constraints = [
        Constraint(
            scope=frozenset([f'N{i}', f'N{j}']),
            relation=lambda ci, cj: ci != cj
        )
        for i, j in edges
    ]
    
    return CSP(variables=frozenset(variables), domains=domains, constraints=constraints)


def generate_sudoku_like(size: int) -> CSP:
    """Genera un problema tipo Sudoku simplificado (solo restricciones de fila/columna)."""
    variables = [f'C{i}_{j}' for i in range(size) for j in range(size)]
    domains = {var: frozenset(range(1, size + 1)) for var in variables}
    
    constraints = []
    
    # Restricciones de fila
    for i in range(size):
        for j1 in range(size):
            for j2 in range(j1 + 1, size):
                constraints.append(
                    Constraint(
                        scope=frozenset([f'C{i}_{j1}', f'C{i}_{j2}']),
                        relation=lambda v1, v2: v1 != v2
                    )
                )
    
    # Restricciones de columna
    for j in range(size):
        for i1 in range(size):
            for i2 in range(i1 + 1, size):
                constraints.append(
                    Constraint(
                        scope=frozenset([f'C{i1}_{j}', f'C{i2}_{j}']),
                        relation=lambda v1, v2: v1 != v2
                    )
                )
    
    return CSP(variables=frozenset(variables), domains=domains, constraints=constraints)


# ============================================================================
# Benchmark
# ============================================================================

def benchmark_strategy(csp: CSP, var_selector, val_orderer, problem_name: str):
    """Ejecuta benchmark para una combinación de estrategias."""
    solver = CSPSolver(csp, variable_selector=var_selector, value_orderer=val_orderer)
    
    start_time = time.perf_counter()
    stats = solver.solve()
    end_time = time.perf_counter()
    
    return {
        'problem': problem_name,
        'var_selector': var_selector.__class__.__name__,
        'val_orderer': val_orderer.__class__.__name__,
        'solved': len(stats.solutions) > 0,
        'solutions': len(stats.solutions),
        'nodes': stats.nodes_explored,
        'backtracks': stats.backtracks,
        'time': end_time - start_time,
        'efficiency': 100.0 if stats.backtracks == 0 and stats.nodes_explored > 0 else 
                     (1 - stats.backtracks / max(stats.nodes_explored, 1)) * 100
    }


def run_benchmarks():
    """Ejecuta suite completa de benchmarks."""
    print("=" * 80)
    print("BENCHMARK FASE 2: SISTEMA DE ESTRATEGIAS MODULARES")
    print("=" * 80)
    print()
    
    # Definir estrategias a probar
    strategies = [
        (FirstUnassignedSelector(), NaturalOrderer(), "Baseline"),
        (MRVSelector(), NaturalOrderer(), "MRV"),
        (DegreeSelector(), NaturalOrderer(), "Degree"),
        (MRVDegreeSelector(), NaturalOrderer(), "MRV+Degree"),
        (FirstUnassignedSelector(), LCVOrderer(), "LCV"),
        (MRVDegreeSelector(), LCVOrderer(), "MRV+Degree+LCV"),
    ]
    
    # Definir problemas
    problems = [
        (generate_nqueens(4), "N-Queens 4x4"),
        (generate_nqueens(5), "N-Queens 5x5"),
        (generate_nqueens(6), "N-Queens 6x6"),
        (generate_nqueens(7), "N-Queens 7x7"),
        (generate_nqueens(8), "N-Queens 8x8"),
        (generate_graph_coloring(5, 3, [(0,1), (1,2), (2,3), (3,4), (4,0)]), "Graph 5/3 (cycle)"),
        (generate_graph_coloring(8, 3, [(0,1), (1,2), (2,3), (3,4), (4,5), (5,6), (6,7), (7,0)]), "Graph 8/3 (cycle)"),
        (generate_graph_coloring(10, 4, [(i, (i+1)%10) for i in range(10)] + [(i, (i+2)%10) for i in range(10)]), "Graph 10/4 (dense)"),
        (generate_sudoku_like(3), "Sudoku-like 3x3"),
        (generate_sudoku_like(4), "Sudoku-like 4x4"),
    ]
    
    results = []
    
    for csp, problem_name in problems:
        print(f"\n{'='*80}")
        print(f"PROBLEMA: {problem_name}")
        print(f"{'='*80}")
        print(f"{'Estrategia':<30} {'Solved':<8} {'Nodes':<8} {'Backtracks':<12} {'Time (s)':<12} {'Efficiency':<10}")
        print(f"{'-'*80}")
        
        for var_sel, val_ord, strategy_name in strategies:
            result = benchmark_strategy(csp, var_sel, val_ord, problem_name)
            results.append(result)
            
            solved_str = "✓" if result['solved'] else "✗"
            print(f"{strategy_name:<30} {solved_str:<8} {result['nodes']:<8} {result['backtracks']:<12} {result['time']:<12.6f} {result['efficiency']:<10.1f}%")
    
    # Resumen
    print(f"\n{'='*80}")
    print("RESUMEN GENERAL")
    print(f"{'='*80}")
    
    # Agrupar por estrategia
    from collections import defaultdict
    by_strategy = defaultdict(list)
    for result in results:
        strategy_key = f"{result['var_selector']}+{result['val_orderer']}"
        by_strategy[strategy_key].append(result)
    
    print(f"\n{'Estrategia':<40} {'Problemas':<12} {'Resueltos':<12} {'Avg Nodes':<12} {'Avg Time (s)':<15}")
    print(f"{'-'*80}")
    
    for strategy_key, strategy_results in sorted(by_strategy.items()):
        total_problems = len(strategy_results)
        solved_problems = sum(1 for r in strategy_results if r['solved'])
        avg_nodes = sum(r['nodes'] for r in strategy_results) / total_problems
        avg_time = sum(r['time'] for r in strategy_results) / total_problems
        
        print(f"{strategy_key:<40} {total_problems:<12} {solved_problems:<12} {avg_nodes:<12.1f} {avg_time:<15.6f}")
    
    # Validación de no-regresión
    print(f"\n{'='*80}")
    print("VALIDACIÓN DE NO-REGRESIÓN")
    print(f"{'='*80}")
    
    baseline_results = [r for r in results if r['var_selector'] == 'FirstUnassignedSelector' and r['val_orderer'] == 'NaturalOrderer']
    best_results = [r for r in results if r['var_selector'] == 'MRVDegreeSelector' and r['val_orderer'] == 'LCVOrderer']
    
    print(f"\nBaseline (FirstUnassigned + Natural):")
    print(f"  - Problemas resueltos: {sum(1 for r in baseline_results if r['solved'])}/{len(baseline_results)}")
    print(f"  - Promedio nodos: {sum(r['nodes'] for r in baseline_results) / len(baseline_results):.1f}")
    print(f"  - Promedio backtracks: {sum(r['backtracks'] for r in baseline_results) / len(baseline_results):.1f}")
    
    print(f"\nMejor estrategia (MRV+Degree + LCV):")
    print(f"  - Problemas resueltos: {sum(1 for r in best_results if r['solved'])}/{len(best_results)}")
    print(f"  - Promedio nodos: {sum(r['nodes'] for r in best_results) / len(best_results):.1f}")
    print(f"  - Promedio backtracks: {sum(r['backtracks'] for r in best_results) / len(best_results):.1f}")
    
    improvement_nodes = (1 - (sum(r['nodes'] for r in best_results) / len(best_results)) / 
                        (sum(r['nodes'] for r in baseline_results) / len(baseline_results))) * 100
    
    print(f"\nMejora en nodos explorados: {improvement_nodes:.1f}%")
    
    if improvement_nodes > 0:
        print("✓ NO HAY REGRESIÓN: Las estrategias avanzadas mejoran el rendimiento")
    else:
        print("✗ POSIBLE REGRESIÓN: Las estrategias avanzadas no mejoran el rendimiento")
    
    print(f"\n{'='*80}")
    print("BENCHMARK COMPLETADO")
    print(f"{'='*80}")


if __name__ == '__main__':
    run_benchmarks()

