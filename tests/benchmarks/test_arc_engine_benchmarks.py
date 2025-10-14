"""
Suite de benchmarks usando el ArcEngine real de LatticeWeaver.

Este módulo evalúa el rendimiento del compilador multiescala cuando se integra
correctamente con el ArcEngine (AC-3.1 optimizado) y el CSPSolver.
"""

import pytest
import time
from typing import Dict, Any, List

from lattice_weaver.core.csp_problem import CSP
from lattice_weaver.arc_engine.core import ArcEngine
from lattice_weaver.arc_engine.csp_solver import CSPSolver
from lattice_weaver.arc_engine.constraints import register_relation
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
# Registro de Relaciones para ArcEngine
# ============================================================================

def not_equal(a, b):
    """Relación de desigualdad."""
    return a != b

def not_attack_queens(a, b, metadata):
    """
    Relación para N-Queens: dos reinas no se atacan.
    
    Args:
        a: Columna de la primera reina
        b: Columna de la segunda reina
        metadata: Diccionario con 'var1_idx' y 'var2_idx' (filas)
    """
    i = metadata.get('var1_idx')
    j = metadata.get('var2_idx')
    
    if i is None or j is None:
        return False
    
    # No misma columna
    if a == b:
        return False
    
    # No misma diagonal
    if abs(a - b) == abs(i - j):
        return False
    
    return True

def not_equal_rel(a, b, metadata):
    """Relación de desigualdad."""
    return a != b

# Registrar relaciones
register_relation("not_equal", not_equal_rel)
register_relation("not_attack_queens", not_attack_queens)


# ============================================================================
# Funciones de Conversión CSP -> ArcEngine
# ============================================================================

def csp_to_arc_engine(csp: CSP, parallel: bool = False) -> ArcEngine:
    """
    Convierte un CSP a un ArcEngine.
    
    Args:
        csp: Problema CSP
        parallel: Si True, habilita paralelización
        
    Returns:
        ArcEngine configurado con el problema
    """
    engine = ArcEngine(parallel=parallel, parallel_mode='topological')
    
    # Añadir variables y dominios
    for var in csp.variables:
        engine.add_variable(var, csp.domains[var])
    
    # Añadir restricciones
    for i, constraint in enumerate(csp.constraints):
        if len(constraint.scope) == 2:
            scope_list = list(constraint.scope)
            var1, var2 = scope_list[0], scope_list[1]
            
            # Determinar el nombre de la relación
            if hasattr(constraint, 'relation_name'):
                relation_name = constraint.relation_name
            else:
                # Inferir el nombre de la relación según el tipo de problema
                if 'Q' in var1 and 'Q' in var2:  # N-Queens
                    relation_name = "not_attack_queens"
                else:
                    relation_name = "not_equal"
            
            # Extraer metadatos si es N-Queens
            metadata = {}
            if 'Q' in var1 and 'Q' in var2:
                # Extraer índices de fila de los nombres de variables Q0, Q1, etc.
                var1_idx = int(var1[1:]) if var1[1:].isdigit() else None
                var2_idx = int(var2[1:]) if var2[1:].isdigit() else None
                if var1_idx is not None and var2_idx is not None:
                    metadata['var1_idx'] = var1_idx
                    metadata['var2_idx'] = var2_idx
            
            engine.add_constraint(var1, var2, relation_name, metadata=metadata, cid=f"c{i}")
    
    return engine


# ============================================================================
# Funciones de Resolución
# ============================================================================

def solve_with_arc_engine(csp: CSP, timeout: float = 300.0, parallel: bool = False) -> Dict[str, Any]:
    """
    Resuelve un CSP usando el ArcEngine y CSPSolver.
    
    Args:
        csp: Problema CSP
        timeout: Tiempo máximo de ejecución (segundos)
        parallel: Si True, habilita paralelización en AC-3
        
    Returns:
        Diccionario con métricas de rendimiento
    """
    start_time = time.time()
    
    try:
        # Convertir CSP a ArcEngine
        engine = csp_to_arc_engine(csp, parallel=parallel)
        
        # Crear solver
        solver = CSPSolver(engine)
        
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

class TestArcEngineBenchmarks:
    """Suite de benchmarks usando el ArcEngine real."""
    
    @pytest.mark.parametrize("size", [4, 6, 8, 10])
    def test_nqueens_arc_engine(self, size):
        """Benchmark de N-Queens con ArcEngine."""
        print(f"\n{'='*60}")
        print(f"N-Queens {size}x{size} con ArcEngine")
        print(f"{'='*60}")
        
        # Generar problema
        csp = generate_nqueens(size)
        
        # Resolver con ArcEngine (secuencial)
        print("\n--- ArcEngine Secuencial ---")
        result_seq = solve_with_arc_engine(csp, parallel=False)
        print(f"Tiempo: {result_seq['time']:.4f}s")
        print(f"Nodos explorados: {result_seq['nodes_explored']}")
        print(f"Solución encontrada: {result_seq['success']}")
        
        # Resolver con ArcEngine (paralelo)
        print("\n--- ArcEngine Paralelo ---")
        result_par = solve_with_arc_engine(csp, parallel=True)
        print(f"Tiempo: {result_par['time']:.4f}s")
        print(f"Nodos explorados: {result_par['nodes_explored']}")
        print(f"Solución encontrada: {result_par['success']}")
        
        # Verificar que se encontró solución
        assert result_seq['success'] or result_par['success'], \
            f"No se encontró solución para N-Queens {size}x{size}"
    
    @pytest.mark.parametrize("size", [4, 9])
    def test_sudoku_arc_engine(self, size):
        """Benchmark de Sudoku con ArcEngine."""
        print(f"\n{'='*60}")
        print(f"Sudoku {size}x{size} con ArcEngine")
        print(f"{'='*60}")
        
        # Generar problema
        csp = generate_sudoku(size)
        
        # Resolver con ArcEngine
        result = solve_with_arc_engine(csp, timeout=60.0)
        print(f"Tiempo: {result['time']:.4f}s")
        print(f"Nodos explorados: {result['nodes_explored']}")
        print(f"Solución encontrada: {result['success']}")
        
        # Para Sudoku 4x4 debe encontrar solución rápidamente
        if size == 4:
            assert result['success'], f"No se encontró solución para Sudoku {size}x{size}"
            assert result['time'] < 10.0, f"Sudoku {size}x{size} tomó demasiado tiempo"
    
    @pytest.mark.parametrize("num_nodes,density", [(10, 0.2), (15, 0.3), (20, 0.2)])
    def test_graph_coloring_arc_engine(self, num_nodes, density):
        """Benchmark de Graph Coloring con ArcEngine."""
        print(f"\n{'='*60}")
        print(f"Graph Coloring {num_nodes} nodos, densidad {density} con ArcEngine")
        print(f"{'='*60}")
        
        # Generar problema
        csp = generate_graph_coloring(num_nodes, num_colors=3, edge_density=density)
        
        # Resolver con ArcEngine
        result = solve_with_arc_engine(csp, timeout=60.0)
        print(f"Tiempo: {result['time']:.4f}s")
        print(f"Nodos explorados: {result['nodes_explored']}")
        print(f"Solución encontrada: {result['success']}")
    
    @pytest.mark.parametrize("size,density", [(5, 0.2), (10, 0.3)])
    def test_simple_csp_arc_engine(self, size, density):
        """Benchmark de Simple CSP con ArcEngine."""
        print(f"\n{'='*60}")
        print(f"Simple CSP {size} variables, densidad {density} con ArcEngine")
        print(f"{'='*60}")
        
        # Generar problema
        csp = generate_simple_csp(size, density)
        
        # Resolver con ArcEngine
        result = solve_with_arc_engine(csp, timeout=60.0)
        print(f"Tiempo: {result['time']:.4f}s")
        print(f"Nodos explorados: {result['nodes_explored']}")
        print(f"Solución encontrada: {result['success']}")


# ============================================================================
# Tests de Comparación: ArcEngine vs SimpleBacktracking
# ============================================================================

class TestArcEngineVsSimple:
    """Comparación entre ArcEngine y SimpleBacktracking."""
    
    def test_nqueens_comparison(self):
        """Compara ArcEngine vs SimpleBacktracking en N-Queens."""
        from lattice_weaver.core.simple_backtracking_solver import SimpleBacktrackingSolver
        
        sizes = [4, 6, 8]
        
        print(f"\n{'='*60}")
        print(f"Comparación ArcEngine vs SimpleBacktracking (N-Queens)")
        print(f"{'='*60}")
        print(f"{'Tamaño':<10} {'ArcEngine (s)':<15} {'Simple (s)':<15} {'Speedup':<10}")
        print(f"{'-'*60}")
        
        for size in sizes:
            csp = generate_nqueens(size)
            
            # ArcEngine
            result_arc = solve_with_arc_engine(csp, parallel=False)
            time_arc = result_arc['time']
            
            # SimpleBacktracking
            start = time.time()
            solver_simple = SimpleBacktrackingSolver()
            solution_simple = solver_simple.solve(csp)
            time_simple = time.time() - start
            
            speedup = time_simple / time_arc if time_arc > 0 else float('inf')
            
            print(f"{size:<10} {time_arc:<15.4f} {time_simple:<15.4f} {speedup:<10.2f}x")
        
        print(f"{'-'*60}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

