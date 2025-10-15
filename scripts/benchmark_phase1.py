#!/usr/bin/env python3.11
"""
Benchmark para Fase 1: Heurísticas MRV/Degree/LCV

Este script compara el rendimiento del CSPSolver con las nuevas heurísticas
integradas contra problemas de referencia.
"""

import time
import sys
from pathlib import Path

# Añadir el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lattice_weaver.core.csp_problem import CSP, Constraint
from lattice_weaver.core.csp_engine.solver import CSPSolver


def create_nqueens_csp(n):
    """Crea un CSP para el problema de N-Queens"""
    variables = [f'Q{i}' for i in range(n)]
    domains = {var: frozenset(range(n)) for var in variables}
    
    constraints = []
    for i in range(n):
        for j in range(i + 1, n):
            # Restricción: reinas no en misma fila
            constraints.append(
                Constraint(
                    scope=(f'Q{i}', f'Q{j}'),
                    relation=lambda qi, qj: qi != qj
                )
            )
            # Restricción: reinas no en misma diagonal
            constraints.append(
                Constraint(
                    scope=(f'Q{i}', f'Q{j}'),
                    relation=lambda qi, qj, i=i, j=j: abs(qi - qj) != abs(i - j)
                )
            )
    
    return CSP(variables=variables, domains=domains, constraints=constraints)


def create_graph_coloring_csp(n_nodes, n_colors, density=0.5):
    """Crea un CSP para coloreo de grafos"""
    import random
    random.seed(42)  # Reproducibilidad
    
    variables = [f'N{i}' for i in range(n_nodes)]
    domains = {var: frozenset(range(n_colors)) for var in variables}
    
    constraints = []
    # Crear aristas aleatorias
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if random.random() < density:
                constraints.append(
                    Constraint(
                        scope=(f'N{i}', f'N{j}'),
                        relation=lambda ci, cj: ci != cj
                    )
                )
    
    return CSP(variables=variables, domains=domains, constraints=constraints)


def create_sudoku_csp_simple(size=4):
    """Crea un CSP simplificado tipo Sudoku (size x size)"""
    # Para simplificar, solo restricciones de fila y columna
    variables = [f'C{i}_{j}' for i in range(size) for j in range(size)]
    domains = {var: frozenset(range(1, size + 1)) for var in variables}
    
    constraints = []
    
    # Restricciones de fila
    for i in range(size):
        for j1 in range(size):
            for j2 in range(j1 + 1, size):
                constraints.append(
                    Constraint(
                        scope=(f'C{i}_{j1}', f'C{i}_{j2}'),
                        relation=lambda v1, v2: v1 != v2
                    )
                )
    
    # Restricciones de columna
    for j in range(size):
        for i1 in range(size):
            for i2 in range(i1 + 1, size):
                constraints.append(
                    Constraint(
                        scope=(f'C{i1}_{j}', f'C{i2}_{j}'),
                        relation=lambda v1, v2: v1 != v2
                    )
                )
    
    return CSP(variables=variables, domains=domains, constraints=constraints)


def benchmark_problem(csp, name, timeout=30):
    """
    Ejecuta benchmark de un problema CSP
    
    Args:
        csp: El problema CSP
        name: Nombre descriptivo del problema
        timeout: Tiempo máximo en segundos
    
    Returns:
        Dict con resultados del benchmark
    """
    solver = CSPSolver(csp)
    
    start = time.perf_counter()
    try:
        stats = solver.solve(all_solutions=False, max_solutions=1)
        elapsed = time.perf_counter() - start
        
        if elapsed > timeout:
            return {
                'name': name,
                'status': 'TIMEOUT',
                'time': elapsed,
                'nodes': stats.nodes_explored,
                'backtracks': stats.backtracks,
                'constraints': stats.constraints_checked,
                'solutions': len(stats.solutions)
            }
        
        return {
            'name': name,
            'status': 'SUCCESS' if len(stats.solutions) > 0 else 'NO_SOLUTION',
            'time': elapsed,
            'nodes': stats.nodes_explored,
            'backtracks': stats.backtracks,
            'constraints': stats.constraints_checked,
            'solutions': len(stats.solutions)
        }
    except Exception as e:
        elapsed = time.perf_counter() - start
        return {
            'name': name,
            'status': 'ERROR',
            'time': elapsed,
            'error': str(e),
            'nodes': 0,
            'backtracks': 0,
            'constraints': 0,
            'solutions': 0
        }


def print_result(result):
    """Imprime resultado de benchmark formateado"""
    status_symbol = {
        'SUCCESS': '✓',
        'NO_SOLUTION': '✗',
        'TIMEOUT': '⏱',
        'ERROR': '⚠'
    }
    
    symbol = status_symbol.get(result['status'], '?')
    print(f"{symbol} {result['name']:35s} | "
          f"{result['time']:8.4f}s | "
          f"{result['nodes']:8d} nodos | "
          f"{result['backtracks']:8d} backtracks | "
          f"{result['constraints']:10d} checks")


def main():
    print("=" * 100)
    print("BENCHMARK FASE 1: Heurísticas MRV/Degree/LCV")
    print("=" * 100)
    print()
    
    results = []
    
    # N-Queens
    print("📋 N-Queens Problems:")
    print("-" * 100)
    for n in [4, 5, 6, 7, 8]:
        csp = create_nqueens_csp(n)
        result = benchmark_problem(csp, f"N-Queens {n}x{n}")
        print_result(result)
        results.append(result)
    
    print()
    
    # Graph Coloring
    print("📋 Graph Coloring Problems:")
    print("-" * 100)
    for nodes in [5, 8, 10]:
        for colors in [3, 4]:
            csp = create_graph_coloring_csp(nodes, colors, density=0.4)
            result = benchmark_problem(csp, f"Graph Coloring {nodes} nodes, {colors} colors")
            print_result(result)
            results.append(result)
    
    print()
    
    # Sudoku-like
    print("📋 Sudoku-like Problems:")
    print("-" * 100)
    for size in [3, 4]:
        csp = create_sudoku_csp_simple(size)
        result = benchmark_problem(csp, f"Sudoku-like {size}x{size}")
        print_result(result)
        results.append(result)
    
    print()
    print("=" * 100)
    print("RESUMEN DE RESULTADOS")
    print("=" * 100)
    
    successful = [r for r in results if r['status'] == 'SUCCESS']
    failed = [r for r in results if r['status'] != 'SUCCESS']
    
    print(f"\n✓ Problemas resueltos: {len(successful)}/{len(results)}")
    print(f"✗ Problemas no resueltos: {len(failed)}/{len(results)}")
    
    if successful:
        avg_time = sum(r['time'] for r in successful) / len(successful)
        avg_nodes = sum(r['nodes'] for r in successful) / len(successful)
        avg_backtracks = sum(r['backtracks'] for r in successful) / len(successful)
        
        print(f"\nPromedios (problemas resueltos):")
        print(f"  Tiempo: {avg_time:.4f}s")
        print(f"  Nodos explorados: {avg_nodes:.1f}")
        print(f"  Backtracks: {avg_backtracks:.1f}")
    
    print()
    print("=" * 100)
    print("ANÁLISIS DE EFICIENCIA")
    print("=" * 100)
    
    # Análisis de eficiencia de heurísticas
    for r in successful:
        if r['nodes'] > 0:
            efficiency = (1 - r['backtracks'] / r['nodes']) * 100
            print(f"{r['name']:35s} | Eficiencia: {efficiency:5.1f}% "
                  f"(backtracks/nodos = {r['backtracks']}/{r['nodes']})")
    
    print()
    print("Nota: Eficiencia = (1 - backtracks/nodos) * 100%")
    print("      Mayor eficiencia indica mejor selección de variables/valores")
    print()


if __name__ == '__main__':
    main()

