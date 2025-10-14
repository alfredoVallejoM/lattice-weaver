import pytest
import time
from typing import Dict, Any, List

from lattice_weaver.core.csp_problem import CSP
from lattice_weaver.core.csp_engine.solver import CSPSolver
from lattice_weaver.benchmarks.generators import (
    generate_nqueens,
    generate_sudoku,
    generate_graph_coloring,
    generate_simple_csp
)
from lattice_weaver.benchmarks.orchestrator import (
    BenchmarkMetrics,
    NoCompilationStrategy,
    FixedLevelStrategy
)


# ============================================================================
# Funciones de Conversión CSP -> CSPSolver
# ============================================================================

def csp_to_csp_solver(csp: CSP, parallel: bool = False) -> CSPSolver:
    """
    Convierte un CSP a un CSPSolver.
    
    Args:
        csp: Problema CSP
        parallel: Si True, habilita paralelización
        
    Returns:
        CSPSolver configurado con el problema
    """
    solver = CSPSolver(csp, parallel=parallel, parallel_mode='topological')
    
    # Las variables y dominios ya están en el objeto CSP
    # Las restricciones ya están en el objeto CSP
    # El CSPSolver opera directamente sobre el objeto CSP
    return solver


# ============================================================================
# Funciones de Resolución
# ============================================================================

def solve_with_csp_solver(csp: CSP, timeout: float = 300.0, parallel: bool = False) -> Dict[str, Any]:
    """
    Resuelve un CSP usando el CSPSolver.
    
    Args:
        csp: Problema CSP
        timeout: Tiempo máximo de ejecución (segundos)
        parallel: Si True, habilita paralelización en AC-3
        
    Returns:
        Diccionario con métricas de rendimiento
    """
    start_time = time.time()
    
    try:
        # Crear solver
        solver = csp_to_csp_solver(csp, parallel=parallel)
        
        # Resolver
        result = solver.solve(return_all=False, max_solutions=1)
        
        elapsed_time = time.time() - start_time
        
        return {
            'success': len(result.solutions) > 0,
            'time': elapsed_time,
            'nodes_explored': result.nodes_explored,
            'solution': result.solutions[0] if result.solutions else None,
            'timeout': elapsed_time >= timeout
        }
    
    except Exception as e:
        elapsed_time = time.time() - start_time
        return {
            'success': False,
            'time': elapsed_time,
            'nodes_explored': 0,
            'solution': None,
            'error': str(e),
            'timeout': False
        }


# ============================================================================
# Tests de Benchmarking
# ============================================================================

class TestCSPSolverBenchmarks:
    """Suite de benchmarks usando el CSPSolver."""
    
    @pytest.mark.parametrize("size", [4, 6, 8, 10])
    def test_nqueens_csp_solver(self, size):
        """Benchmark de N-Queens con CSPSolver."""
        print(f"\n{'='*60}")
        print(f"N-Queens {size}x{size} con CSPSolver")
        print(f"{'='*60}")
        
        # Generar problema
        csp = generate_nqueens(size)
        
        # Resolver con CSPSolver (secuencial)
        print("\n--- CSPSolver Secuencial ---")
        result_seq = solve_with_csp_solver(csp, parallel=False)
        print(f"Tiempo: {result_seq['time']:.4f}s")
        print(f"Nodos explorados: {result_seq['nodes_explored']}")
        print(f"Solución encontrada: {result_seq['success']}")
        
        # Resolver con CSPSolver (paralelo)
        print("\n--- CSPSolver Paralelo ---")
        result_par = solve_with_csp_solver(csp, parallel=True)
        print(f"Tiempo: {result_par['time']:.4f}s")
        print(f"Nodos explorados: {result_par['nodes_explored']}")
        print(f"Solución encontrada: {result_par['success']}")
        
        # Verificar que se encontró solución
        assert result_seq['success'] or result_par['success'], \
            f"No se encontró solución para N-Queens {size}x{size}"
    
    @pytest.mark.parametrize("size", [4, 9])
    def test_sudoku_csp_solver(self, size):
        """Benchmark de Sudoku con CSPSolver."""
        print(f"\n{'='*60}")
        print(f"Sudoku {size}x{size} con CSPSolver")
        print(f"{'='*60}")
        
        # Generar problema
        csp = generate_sudoku(size)
        
        # Resolver con CSPSolver
        result = solve_with_csp_solver(csp, timeout=60.0)
        print(f"Tiempo: {result['time']:.4f}s")
        print(f"Nodos explorados: {result['nodes_explored']}")
        print(f"Solución encontrada: {result['success']}")
        
        # Para Sudoku 4x4 debe encontrar solución rápidamente
        if size == 4:
            assert result['success'], f"No se encontró solución para Sudoku {size}x{size}"
            assert result['time'] < 10.0, f"Sudoku {size}x{size} tomó demasiado tiempo"
    
    @pytest.mark.parametrize("num_nodes,density", [(10, 0.2), (15, 0.3), (20, 0.2)])
    def test_graph_coloring_csp_solver(self, num_nodes, density):
        """Benchmark de Graph Coloring con CSPSolver."""
        print(f"\n{'='*60}")
        print(f"Graph Coloring {num_nodes} nodos, densidad {density} con CSPSolver")
        print(f"{'='*60}")
        
        # Generar problema
        csp = generate_graph_coloring(num_nodes, num_colors=3, edge_density=density)
        
        # Resolver con CSPSolver
        result = solve_with_csp_solver(csp, timeout=60.0)
        print(f"Tiempo: {result['time']:.4f}s")
        print(f"Nodos explorados: {result['nodes_explored']}")
        print(f"Solución encontrada: {result['success']}")
    
    @pytest.mark.parametrize("size,density", [(5, 0.2), (10, 0.3)])
    def test_simple_csp_solver(self, size, density):
        """Benchmark de Simple CSP con CSPSolver."""
        print(f"\n{'='*60}")
        print(f"Simple CSP {size} variables, densidad {density} con CSPSolver")
        print(f"{'='*60}")
        
        # Generar problema
        csp = generate_simple_csp(size, density)
        
        # Resolver con CSPSolver
        result = solve_with_csp_solver(csp, timeout=60.0)
        print(f"Tiempo: {result['time']:.4f}s")
        print(f"Nodos explorados: {result['nodes_explored']}")
        print(f"Solución encontrada: {result['success']}")


# ============================================================================
# Tests de Comparación: CSPSolver vs SimpleBacktracking
# ============================================================================

class TestCSPSolverVsSimple:
    """Comparación entre CSPSolver y SimpleBacktracking."""
    
    def test_nqueens_comparison(self):
        """Compara CSPSolver vs SimpleBacktracking en N-Queens."""
        from lattice_weaver.core.simple_backtracking_solver import SimpleBacktrackingSolver
        
        sizes = [4, 6, 8]
        
        print(f"\n{'='*60}")
        print(f"Comparación CSPSolver vs SimpleBacktracking (N-Queens)")
        print(f"{'='*60}")
        print(f"{'Tamaño':<10} {'CSPSolver (s)':<15} {'Simple (s)':<15} {'Speedup':<10}")
        print(f"{'--'*30}")
        
        for size in sizes:
            csp = generate_nqueens(size)
            
            # CSPSolver
            result_arc = solve_with_csp_solver(csp, parallel=False)
            time_arc = result_arc['time']
            
            # SimpleBacktracking
            start = time.time()
            solver_simple = SimpleBacktrackingSolver()
            solution_simple = solver_simple.solve(csp)
            time_simple = time.time() - start
            
            speedup = time_simple / time_arc if time_arc > 0 else float('inf')
            
            print(f"{size:<10} {time_arc:<15.4f} {time_simple:<15.4f} {speedup:<10.2f}x")
        
        print(f"{'--'*30}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

