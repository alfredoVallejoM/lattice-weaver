"""
Suite de benchmarks completa para LatticeWeaver.

Compara:
1. SimpleBacktrackingSolver (sin AC-3)
2. CSPSolver con ArcEngine (con AC-3)
3. Compilador Multiescala (sin ArcEngine integrado)
4. Compilador Multiescala con ArcEngine (nueva implementación)
"""

import pytest
import time
import json
from typing import Dict, Any

from lattice_weaver.core.csp_problem import CSP
from lattice_weaver.core.simple_backtracking_solver import solve_csp_backtracking
from lattice_weaver.arc_engine.csp_solver import CSPSolver, CSPProblem
from lattice_weaver.arc_engine.constraints import register_relation
from lattice_weaver.benchmarks.generators import (
    generate_nqueens,
    generate_sudoku,
    generate_graph_coloring,
    generate_simple_csp
)


# ============================================================================
# Registro de Relaciones
# ============================================================================

def not_attack_queens(a, b, metadata):
    """Relación para N-Queens."""
    i = metadata.get('var1_idx')
    j = metadata.get('var2_idx')
    if i is None or j is None:
        return False
    if a == b:  # Misma columna
        return False
    if abs(a - b) == abs(i - j):  # Misma diagonal
        return False
    return True

def not_equal_rel(a, b, metadata):
    """Relación de desigualdad."""
    return a != b

# Registrar relaciones
try:
    register_relation("not_equal", not_equal_rel)
    register_relation("not_attack_queens", not_attack_queens)
except ValueError:
    pass  # Ya registradas


# ============================================================================
# Funciones de Conversión
# ============================================================================

def csp_to_csp_problem(csp: CSP, relation_name: str = "not_equal") -> CSPProblem:
    """
    Convierte un CSP a CSPProblem para usar con CSPSolver.
    
    Args:
        csp: Problema CSP
        relation_name: Nombre de la relación a usar para todas las restricciones
        
    Returns:
        CSPProblem configurado
    """
    variables_list = sorted(list(csp.variables))
    domains_dict = {var: list(csp.domains[var]) for var in variables_list}
    
    # Inferir el nombre de relación según el tipo de problema
    constraints_list = []
    for constraint in csp.constraints:
        if len(constraint.scope) == 2:
            scope_list = list(constraint.scope)
            var1, var2 = scope_list[0], scope_list[1]
            
            # Detectar N-Queens
            if 'Q' in var1 and 'Q' in var2:
                rel_name = "not_attack_queens"
            else:
                rel_name = relation_name
            
            constraints_list.append((var1, var2, rel_name))
    
    return CSPProblem(variables_list, domains_dict, constraints_list)


# ============================================================================
# Funciones de Resolución
# ============================================================================

def solve_with_simple_backtracking(csp: CSP, timeout: float = 300.0) -> Dict[str, Any]:
    """Resuelve con solve_csp_backtracking."""
    start_time = time.time()
    
    try:
        solution = solve_csp_backtracking(csp)
        
        elapsed_time = time.time() - start_time
        
        return {
            'success': solution is not None,
            'time': elapsed_time,
            'solution': solution,
            'timeout': elapsed_time >= timeout,
            'method': 'SimpleBacktracking'
        }
    
    except Exception as e:
        elapsed_time = time.time() - start_time
        return {
            'success': False,
            'time': elapsed_time,
            'solution': None,
            'error': str(e),
            'timeout': False,
            'method': 'SimpleBacktracking'
        }


def solve_with_arc_engine(csp: CSP, timeout: float = 300.0, parallel: bool = False) -> Dict[str, Any]:
    """Resuelve con CSPSolver (ArcEngine + AC-3)."""
    start_time = time.time()
    
    try:
        # Convertir a CSPProblem
        problem = csp_to_csp_problem(csp)
        
        # Crear solver
        solver = CSPSolver(use_tms=False, parallel=parallel)
        
        # Resolver
        result = solver.solve(problem, return_all=False, max_solutions=1)
        
        elapsed_time = time.time() - start_time
        
        return {
            'success': len(result.solutions) > 0,
            'time': elapsed_time,
            'nodes_explored': result.nodes_explored,
            'solution': result.solutions[0].assignment if result.solutions else None,
            'timeout': elapsed_time >= timeout,
            'method': f'ArcEngine{"_Parallel" if parallel else ""}'
        }
    
    except Exception as e:
        elapsed_time = time.time() - start_time
        return {
            'success': False,
            'time': elapsed_time,
            'nodes_explored': 0,
            'solution': None,
            'error': str(e),
            'timeout': False,
            'method': f'ArcEngine{"_Parallel" if parallel else ""}'
        }


# ============================================================================
# Tests de Benchmarking
# ============================================================================

class TestCompleteBenchmarks:
    """Suite completa de benchmarks."""
    
    @pytest.mark.parametrize("size", [4, 6, 8, 10])
    def test_nqueens_comparison(self, size):
        """Compara todos los métodos en N-Queens."""
        print(f"\n{'='*70}")
        print(f"N-Queens {size}x{size} - Comparación de Métodos")
        print(f"{'='*70}")
        
        # Generar problema
        csp = generate_nqueens(size)
        
        results = {}
        
        # 1. SimpleBacktracking
        print("\n[1/3] SimpleBacktracking...")
        results['simple'] = solve_with_simple_backtracking(csp, timeout=60.0)
        
        # 2. ArcEngine (secuencial)
        print("[2/3] ArcEngine (secuencial)...")
        results['arc_seq'] = solve_with_arc_engine(csp, timeout=60.0, parallel=False)
        
        # 3. ArcEngine (paralelo)
        print("[3/3] ArcEngine (paralelo)...")
        results['arc_par'] = solve_with_arc_engine(csp, timeout=60.0, parallel=True)
        
        # Mostrar resultados
        print(f"\n{'Método':<25} {'Tiempo (s)':<12} {'Éxito':<8} {'Nodos':<10}")
        print(f"{'-'*70}")
        
        for key, result in results.items():
            method = result['method']
            time_val = f"{result['time']:.4f}"
            success = "✓" if result['success'] else "✗"
            nodes = result.get('nodes_explored', 'N/A')
            
            print(f"{method:<25} {time_val:<12} {success:<8} {nodes:<10}")
        
        # Calcular speedups
        if results['simple']['success'] and results['arc_seq']['success']:
            speedup_seq = results['simple']['time'] / results['arc_seq']['time']
            print(f"\nSpeedup ArcEngine vs Simple: {speedup_seq:.2f}x")
        
        if results['simple']['success'] and results['arc_par']['success']:
            speedup_par = results['simple']['time'] / results['arc_par']['time']
            print(f"Speedup ArcEngine_Parallel vs Simple: {speedup_par:.2f}x")
        
        # Guardar resultados
        output_file = f"/home/ubuntu/lattice-weaver-repo/benchmark_results/nqueens_{size}_comparison.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nResultados guardados en: {output_file}")
        
        # Verificar que al menos un método encontró solución
        assert any(r['success'] for r in results.values()), \
            f"Ningún método encontró solución para N-Queens {size}x{size}"
    
    @pytest.mark.parametrize("size", [4])
    def test_sudoku_comparison(self, size):
        """Compara todos los métodos en Sudoku."""
        print(f"\n{'='*70}")
        print(f"Sudoku {size}x{size} - Comparación de Métodos")
        print(f"{'='*70}")
        
        # Generar problema
        csp = generate_sudoku(size)
        
        results = {}
        
        # 1. SimpleBacktracking
        print("\n[1/3] SimpleBacktracking...")
        results['simple'] = solve_with_simple_backtracking(csp, timeout=60.0)
        
        # 2. ArcEngine (secuencial)
        print("[2/3] ArcEngine (secuencial)...")
        results['arc_seq'] = solve_with_arc_engine(csp, timeout=60.0, parallel=False)
        
        # 3. ArcEngine (paralelo)
        print("[3/3] ArcEngine (paralelo)...")
        results['arc_par'] = solve_with_arc_engine(csp, timeout=60.0, parallel=True)
        
        # Mostrar resultados
        print(f"\n{'Método':<25} {'Tiempo (s)':<12} {'Éxito':<8} {'Nodos':<10}")
        print(f"{'-'*70}")
        
        for key, result in results.items():
            method = result['method']
            time_val = f"{result['time']:.4f}"
            success = "✓" if result['success'] else "✗"
            nodes = result.get('nodes_explored', 'N/A')
            
            print(f"{method:<25} {time_val:<12} {success:<8} {nodes:<10}")
        
        # Guardar resultados
        output_file = f"/home/ubuntu/lattice-weaver-repo/benchmark_results/sudoku_{size}_comparison.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nResultados guardados en: {output_file}")
    
    @pytest.mark.parametrize("num_nodes,density", [(10, 0.2), (15, 0.3)])
    def test_graph_coloring_comparison(self, num_nodes, density):
        """Compara todos los métodos en Graph Coloring."""
        print(f"\n{'='*70}")
        print(f"Graph Coloring {num_nodes} nodos, densidad {density} - Comparación")
        print(f"{'='*70}")
        
        # Generar problema
        csp = generate_graph_coloring(num_nodes, edge_probability=density, num_colors=3)
        
        results = {}
        
        # 1. SimpleBacktracking
        print("\n[1/3] SimpleBacktracking...")
        results['simple'] = solve_with_simple_backtracking(csp, timeout=60.0)
        
        # 2. ArcEngine (secuencial)
        print("[2/3] ArcEngine (secuencial)...")
        results['arc_seq'] = solve_with_arc_engine(csp, timeout=60.0, parallel=False)
        
        # 3. ArcEngine (paralelo)
        print("[3/3] ArcEngine (paralelo)...")
        results['arc_par'] = solve_with_arc_engine(csp, timeout=60.0, parallel=True)
        
        # Mostrar resultados
        print(f"\n{'Método':<25} {'Tiempo (s)':<12} {'Éxito':<8} {'Nodos':<10}")
        print(f"{'-'*70}")
        
        for key, result in results.items():
            method = result['method']
            time_val = f"{result['time']:.4f}"
            success = "✓" if result['success'] else "✗"
            nodes = result.get('nodes_explored', 'N/A')
            
            print(f"{method:<25} {time_val:<12} {success:<8} {nodes:<10}")
        
        # Guardar resultados
        output_file = f"/home/ubuntu/lattice-weaver-repo/benchmark_results/graph_coloring_{num_nodes}_{density}_comparison.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nResultados guardados en: {output_file}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

