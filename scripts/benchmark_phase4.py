"""
Benchmark completo de Fase 4: Integración Topológica y Estrategias Híbridas.

Este script compara todas las estrategias de Fases 2-4:
- Baseline: FirstUnassigned + Natural
- Fase 2: MRV, Degree, MRV+Degree
- Fase 3: FCA-Guided, FCA-Cluster
- Fase 4: Topology-Guided, Component-Based, Hybrid, Adaptive

Autor: Manus AI
Fecha: 15 de Octubre de 2025
"""

import time
from typing import Dict, List, Tuple
from lattice_weaver.core.csp_problem import CSP, Constraint
from lattice_weaver.core.csp_engine.solver import CSPSolver
from lattice_weaver.core.csp_engine.strategies import (
    FirstUnassignedSelector,
    MRVSelector,
    DegreeSelector,
    MRVDegreeSelector,
    FCAGuidedSelector,
    FCAClusterSelector,
    TopologyGuidedSelector,
    ComponentBasedSelector,
    HybridFCATopologySelector,
    AdaptiveMultiscaleSelector
)


def create_nqueens(n: int) -> CSP:
    """Crea problema N-Queens"""
    variables = [f'Q{i}' for i in range(n)]
    domains = {v: frozenset(range(n)) for v in variables}
    
    constraints = []
    for i in range(n):
        for j in range(i + 1, n):
            # No misma fila
            constraints.append(
                Constraint(
                    scope=(f'Q{i}', f'Q{j}'),
                    relation=lambda qi, qj, i=i, j=j: qi != qj
                )
            )
            # No misma diagonal
            constraints.append(
                Constraint(
                    scope=(f'Q{i}', f'Q{j}'),
                    relation=lambda qi, qj, i=i, j=j: abs(qi - qj) != abs(i - j)
                )
            )
    
    return CSP(variables=variables, domains=domains, constraints=constraints)


def create_graph_coloring(num_nodes: int, num_colors: int = 3) -> CSP:
    """Crea problema de coloración de grafo completo"""
    variables = [f'N{i}' for i in range(num_nodes)]
    domains = {v: frozenset(range(num_colors)) for v in variables}
    
    constraints = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            constraints.append(
                Constraint(
                    scope=(f'N{i}', f'N{j}'),
                    relation=lambda ci, cj: ci != cj
                )
            )
    
    return CSP(variables=variables, domains=domains, constraints=constraints)


def benchmark_strategy(csp: CSP, strategy_name: str, selector, max_time: float = 30.0) -> Dict:
    """
    Ejecuta benchmark de una estrategia.
    
    Args:
        csp: Problema a resolver
        strategy_name: Nombre de la estrategia
        selector: Instancia del selector
        max_time: Tiempo máximo en segundos
    
    Returns:
        Diccionario con resultados
    """
    solver = CSPSolver(csp, variable_selector=selector)
    
    start_time = time.time()
    try:
        result_stats = solver.solve(all_solutions=False, max_solutions=1)
        elapsed = time.time() - start_time
        
        # Verificar timeout
        if elapsed > max_time:
            return {
                'strategy': strategy_name,
                'solved': False,
                'timeout': True,
                'time': elapsed,
                'nodes': result_stats.nodes_explored,
                'backtracks': result_stats.backtracks
            }
        
        return {
            'strategy': strategy_name,
            'solved': len(result_stats.solutions) > 0,
            'timeout': False,
            'time': elapsed,
            'nodes': result_stats.nodes_explored,
            'backtracks': result_stats.backtracks,
            'solutions': len(result_stats.solutions)
        }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            'strategy': strategy_name,
            'solved': False,
            'timeout': False,
            'error': str(e),
            'time': elapsed,
            'nodes': getattr(solver, 'nodes_explored', 0),
            'backtracks': getattr(solver, 'backtracks', 0)
        }


def run_benchmarks():
    """Ejecuta suite completa de benchmarks"""
    
    # Definir estrategias a comparar
    strategies = [
        ('Baseline (FirstUnassigned)', FirstUnassignedSelector()),
        ('MRV', MRVSelector()),
        ('Degree', DegreeSelector()),
        ('MRV+Degree', MRVDegreeSelector()),
        ('FCA-Guided', FCAGuidedSelector()),
        ('FCA-Cluster', FCAClusterSelector()),
        ('Topology-Guided', TopologyGuidedSelector()),
        ('Component-Based', ComponentBasedSelector()),
        ('Hybrid (FCA+Topo)', HybridFCATopologySelector(fca_weight=0.5, topology_weight=0.5)),
        ('Adaptive', AdaptiveMultiscaleSelector())
    ]
    
    # Definir problemas a resolver
    problems = [
        ('N-Queens 4x4', create_nqueens(4)),
        ('N-Queens 6x6', create_nqueens(6)),
        ('N-Queens 8x8', create_nqueens(8)),
        ('Graph Coloring (4 nodes, 3 colors)', create_graph_coloring(4, 3)),
        ('Graph Coloring (5 nodes, 3 colors)', create_graph_coloring(5, 3)),
        ('Graph Coloring (6 nodes, 3 colors)', create_graph_coloring(6, 3)),
    ]
    
    print("=" * 100)
    print("BENCHMARK FASE 4: INTEGRACIÓN TOPOLÓGICA Y ESTRATEGIAS HÍBRIDAS")
    print("=" * 100)
    print()
    
    all_results = []
    
    for problem_name, csp in problems:
        print(f"\n{'=' * 100}")
        print(f"Problema: {problem_name}")
        print(f"Variables: {len(csp.variables)}, Restricciones: {len(csp.constraints)}")
        print(f"{'=' * 100}\n")
        
        problem_results = []
        
        for strategy_name, selector in strategies:
            print(f"Ejecutando: {strategy_name}...", end=' ', flush=True)
            result = benchmark_strategy(csp, strategy_name, selector, max_time=30.0)
            problem_results.append(result)
            
            if result.get('timeout'):
                print(f"TIMEOUT ({result['time']:.2f}s)")
            elif result.get('error'):
                print(f"ERROR: {result['error']}")
            elif result['solved']:
                print(f"✓ {result['time']:.4f}s, {result['nodes']} nodos, {result['backtracks']} backtracks")
            else:
                print(f"✗ No resuelto")
        
        # Mostrar tabla comparativa para este problema
        print(f"\n{'Estrategia':<30} {'Tiempo (s)':<12} {'Nodos':<10} {'Backtracks':<12} {'Estado':<15}")
        print("-" * 100)
        
        for result in problem_results:
            strategy = result['strategy']
            time_str = f"{result['time']:.4f}" if 'time' in result else "N/A"
            nodes_str = str(result.get('nodes', 'N/A'))
            backtracks_str = str(result.get('backtracks', 'N/A'))
            
            if result.get('timeout'):
                status = "TIMEOUT"
            elif result.get('error'):
                status = "ERROR"
            elif result.get('solved'):
                status = "✓ Resuelto"
            else:
                status = "✗ No resuelto"
            
            print(f"{strategy:<30} {time_str:<12} {nodes_str:<10} {backtracks_str:<12} {status:<15}")
        
        all_results.append((problem_name, problem_results))
    
    # Resumen global
    print(f"\n\n{'=' * 100}")
    print("RESUMEN GLOBAL")
    print(f"{'=' * 100}\n")
    
    # Calcular estadísticas por estrategia
    strategy_stats = {}
    for problem_name, results in all_results:
        for result in results:
            strategy = result['strategy']
            if strategy not in strategy_stats:
                strategy_stats[strategy] = {
                    'total_problems': 0,
                    'solved': 0,
                    'total_time': 0.0,
                    'total_nodes': 0,
                    'total_backtracks': 0
                }
            
            stats = strategy_stats[strategy]
            stats['total_problems'] += 1
            if result.get('solved'):
                stats['solved'] += 1
                stats['total_time'] += result.get('time', 0)
                stats['total_nodes'] += result.get('nodes', 0)
                stats['total_backtracks'] += result.get('backtracks', 0)
    
    # Mostrar tabla de resumen
    print(f"{'Estrategia':<30} {'Resueltos':<12} {'Avg Tiempo (s)':<15} {'Avg Nodos':<12} {'Avg Backtracks':<15}")
    print("-" * 100)
    
    for strategy, stats in sorted(strategy_stats.items(), key=lambda x: x[1]['total_time'] / max(x[1]['solved'], 1)):
        solved = stats['solved']
        total = stats['total_problems']
        avg_time = stats['total_time'] / solved if solved > 0 else float('inf')
        avg_nodes = stats['total_nodes'] / solved if solved > 0 else 0
        avg_backtracks = stats['total_backtracks'] / solved if solved > 0 else 0
        
        print(f"{strategy:<30} {solved}/{total:<10} {avg_time:<15.4f} {avg_nodes:<12.1f} {avg_backtracks:<15.1f}")
    
    print("\n" + "=" * 100)
    print("BENCHMARK COMPLETADO")
    print("=" * 100)


if __name__ == '__main__':
    run_benchmarks()

