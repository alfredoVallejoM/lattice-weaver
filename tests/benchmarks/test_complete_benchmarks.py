import pytest
import time
import json
from typing import Dict, Any, List

from lattice_weaver.core.csp_problem import CSP
from lattice_weaver.core.csp_engine.solver import CSPSolver, CSPSolutionStats
from lattice_weaver.core.simple_backtracking_solver import solve_csp_backtracking
from lattice_weaver.benchmarks.generators import (
    generate_nqueens,
    generate_sudoku,
    generate_graph_coloring
)


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


def solve_with_csp_solver(csp: CSP, timeout: float = 300.0, parallel: bool = False) -> Dict[str, Any]:
    """Resuelve con CSPSolver (con AC-3)."""
    start_time = time.time()
    
    try:
        solver = CSPSolver(csp, parallel=parallel)
        solution_stats: CSPSolutionStats = solver.solve()
        
        elapsed_time = time.time() - start_time
        
        return {
            'success': len(solution_stats.solutions) > 0,
            'time': elapsed_time,
            'nodes_explored': solution_stats.nodes_explored,
            'backtracks': solution_stats.backtracks,
            'constraints_checked': solution_stats.constraints_checked,
            'solution': solution_stats.solutions[0] if solution_stats.solutions else None,
            'timeout': elapsed_time >= timeout,
            'method': f'CSPSolver{"_Parallel" if parallel else ""}'
        }
    
    except Exception as e:
        elapsed_time = time.time() - start_time
        return {
            'success': False,
            'time': elapsed_time,
            'nodes_explored': 0,
            'backtracks': 0,
            'constraints_checked': 0,
            'solution': None,
            'error': str(e),
            'timeout': False,
            'method': f'CSPSolver{"_Parallel" if parallel else ""}'
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
        
        csp = generate_nqueens(size)
        
        results = {}
        
        print("\n[1/3] SimpleBacktracking...")
        results['simple'] = solve_with_simple_backtracking(csp, timeout=60.0)
        
        print("[2/3] CSPSolver (secuencial)...")
        results['csp_seq'] = solve_with_csp_solver(csp, timeout=60.0, parallel=False)
        
        print("[3/3] CSPSolver (paralelo)...")
        results['csp_par'] = solve_with_csp_solver(csp, timeout=60.0, parallel=True)
        
        print(f"\n{'Método':<25} {'Tiempo (s)':<12} {'Éxito':<8} {'Nodos':<10} {'Backtracks':<12} {'Constraints':<12}")
        print(f"{'='*70}")
        
        for key, result in results.items():
            method = result['method']
            time_val = f"{result['time']:.4f}"
            success = "✓" if result['success'] else "✗"
            nodes = result.get('nodes_explored', 'N/A')
            backtracks = result.get('backtracks', 'N/A')
            constraints = result.get('constraints_checked', 'N/A')
            
            print(f"{method:<25} {time_val:<12} {success:<8} {nodes:<10} {backtracks:<12} {constraints:<12}")
        
        if results['simple']['success'] and results['csp_seq']['success']:
            speedup_seq = results['simple']['time'] / results['csp_seq']['time']
            print(f"\nSpeedup CSPSolver vs Simple: {speedup_seq:.2f}x")
        
        if results['simple']['success'] and results['csp_par']['success']:
            speedup_par = results['simple']['time'] / results['csp_par']['time']
            print(f"Speedup CSPSolver_Parallel vs Simple: {speedup_par:.2f}x")
        
        output_file = f"/home/ubuntu/lattice-weaver-repo/benchmark_results/nqueens_{size}_comparison.json"
        # Asegurarse de que el directorio existe
        import os
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nResultados guardados en: {output_file}")
        
        assert any(r['success'] for r in results.values()), \
            f"Ningún método encontró solución para N-Queens {size}x{size}"
    
    @pytest.mark.parametrize("size", [4])
    def test_sudoku_comparison(self, size):
        """Compara todos los métodos en Sudoku."""
        print(f"\n{'='*70}")
        print(f"Sudoku {size}x{size} - Comparación de Métodos")
        print(f"{'='*70}")
        
        csp = generate_sudoku(size)
        
        results = {}
        
        print("\n[1/3] SimpleBacktracking...")
        results['simple'] = solve_with_simple_backtracking(csp, timeout=60.0)
        
        print("[2/3] CSPSolver (secuencial)...")
        results['csp_seq'] = solve_with_csp_solver(csp, timeout=60.0, parallel=False)
        
        print("[3/3] CSPSolver (paralelo)...")
        results['csp_par'] = solve_with_csp_solver(csp, timeout=60.0, parallel=True)
        
        print(f"\n{'Método':<25} {'Tiempo (s)':<12} {'Éxito':<8} {'Nodos':<10} {'Backtracks':<12} {'Constraints':<12}")
        print(f"{'='*70}")
        
        for key, result in results.items():
            method = result['method']
            time_val = f"{result['time']:.4f}"
            success = "✓" if result['success'] else "✗"
            nodes = result.get('nodes_explored', 'N/A')
            backtracks = result.get('backtracks', 'N/A')
            constraints = result.get('constraints_checked', 'N/A')
            
            print(f"{method:<25} {time_val:<12} {success:<8} {nodes:<10} {backtracks:<12} {constraints:<12}")
        
        output_file = f"/home/ubuntu/lattice-weaver-repo/benchmark_results/sudoku_{size}_comparison.json"
        import os
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nResultados guardados en: {output_file}")
    
    @pytest.mark.parametrize("num_nodes,density", [(10, 0.2), (15, 0.3)])
    def test_graph_coloring_comparison(self, num_nodes, density):
        """Compara todos los métodos en Graph Coloring."""
        print(f"\n{'='*70}")
        print(f"Graph Coloring {num_nodes} nodos, densidad {density} - Comparación")
        print(f"{'='*70}")
        
        csp = generate_graph_coloring(num_nodes, edge_probability=density, num_colors=3)
        
        results = {}
        
        print("\n[1/3] SimpleBacktracking...")
        results['simple'] = solve_with_simple_backtracking(csp, timeout=60.0)
        
        print("[2/3] CSPSolver (secuencial)...")
        results['csp_seq'] = solve_with_csp_solver(csp, timeout=60.0, parallel=False)
        
        print("[3/3] CSPSolver (paralelo)...")
        results['csp_par'] = solve_with_csp_solver(csp, timeout=60.0, parallel=True)
        
        print(f"\n{'Método':<25} {'Tiempo (s)':<12} {'Éxito':<8} {'Nodos':<10} {'Backtracks':<12} {'Constraints':<12}")
        print(f"{'='*70}")
        
        for key, result in results.items():
            method = result['method']
            time_val = f"{result['time']:.4f}"
            success = "✓" if result['success'] else "✗"
            nodes = result.get('nodes_explored', 'N/A')
            backtracks = result.get('backtracks', 'N/A')
            constraints = result.get('constraints_checked', 'N/A')
            
            print(f"{method:<25} {time_val:<12} {success:<8} {nodes:<10} {backtracks:<12} {constraints:<12}")
        
        output_file = f"/home/ubuntu/lattice-weaver-repo/benchmark_results/graph_coloring_{num_nodes}_{density}_comparison.json"
        import os
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nResultados guardados en: {output_file}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

