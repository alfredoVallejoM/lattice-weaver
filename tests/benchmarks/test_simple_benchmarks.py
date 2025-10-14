"""
Pruebas integradas simplificadas para validar la mejora del sistema con el compilador multiescala.

Esta versión evita dependencias problemáticas importando solo lo necesario.
"""

import pytest
import sys
import os
from typing import Dict, Optional

# Añadir el directorio raíz al path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from lattice_weaver.core.csp_problem import CSP
from lattice_weaver.core.simple_backtracking_solver import solve_csp_backtracking
from lattice_weaver.benchmarks.orchestrator import (
    Orchestrator,
    NoCompilationStrategy,
    FixedLevelStrategy
)
from lattice_weaver.benchmarks.generators import (
    generate_nqueens,
    generate_simple_csp
)


def simple_solver(csp: CSP) -> tuple[Optional[Dict], int, int]:
    """
    Wrapper del solucionador simple para el formato esperado por el orquestador.
    
    Args:
        csp: Problema CSP a resolver.
        
    Returns:
        Tupla con (solución, nodos explorados, backtracks).
    """
    solution = solve_csp_backtracking(csp)
    # Por ahora, no tenemos métricas de nodos y backtracks
    # Retornamos valores dummy
    return solution, 0, 0


class TestSimpleBenchmarks:
    """Pruebas integradas simplificadas del compilador multiescala."""
    
    def setup_method(self):
        """Configuración inicial para cada prueba."""
        self.orchestrator = Orchestrator(simple_solver)
    
    def test_nqueens_8_no_compilation(self):
        """Prueba N-Reinas (8x8) sin compilación."""
        csp = generate_nqueens(8)
        strategy = NoCompilationStrategy()
        
        metrics = self.orchestrator.run_benchmark(csp, strategy)
        
        assert metrics.error is None, f"Error: {metrics.error}"
        assert metrics.solution_found
        assert metrics.total_time > 0
        assert metrics.compilation_time == 0
        assert metrics.solving_time > 0
        print(f"\n  Total time: {metrics.total_time:.4f}s")
        print(f"  Solving time: {metrics.solving_time:.4f}s")
    
    def test_nqueens_8_with_compilation_l1(self):
        """Prueba N-Reinas (8x8) con compilación a L1."""
        csp = generate_nqueens(8)
        strategy = FixedLevelStrategy(1)
        
        metrics = self.orchestrator.run_benchmark(csp, strategy)
        
        assert metrics.error is None, f"Error: {metrics.error}"
        assert metrics.solution_found
        assert metrics.total_time > 0
        assert metrics.compilation_time > 0
        assert metrics.solving_time > 0
        assert metrics.compilation_level == "L1"
        print(f"\n  Total time: {metrics.total_time:.4f}s")
        print(f"  Compilation time: {metrics.compilation_time:.4f}s")
        print(f"  Solving time: {metrics.solving_time:.4f}s")
    
    def test_nqueens_8_comparison(self):
        """Comparación de N-Reinas (8x8) con diferentes estrategias."""
        csp = generate_nqueens(8)
        strategies = [
            NoCompilationStrategy(),
            FixedLevelStrategy(1),
            FixedLevelStrategy(2)
        ]
        
        results = self.orchestrator.run_comparison(csp, strategies)
        
        assert len(results) == 3
        assert all(metrics.error is None for metrics in results.values()), \
            f"Errors: {[m.error for m in results.values() if m.error]}"
        assert all(metrics.solution_found for metrics in results.values())
        
        # Imprimir resumen
        print("\n=== Resumen de Comparación ===")
        for strategy_name, metrics in results.items():
            print(f"\n{strategy_name}:")
            print(f"  Total time: {metrics.total_time:.4f}s")
            print(f"  Compilation time: {metrics.compilation_time:.4f}s")
            print(f"  Solving time: {metrics.solving_time:.4f}s")
            print(f"  Peak memory: {metrics.peak_memory:.2f} MB")
    
    def test_simple_csp_comparison(self):
        """Comparación de CSP simple con diferentes estrategias."""
        csp = generate_simple_csp(10, 5, 0.3)
        strategies = [
            NoCompilationStrategy(),
            FixedLevelStrategy(1)
        ]
        
        results = self.orchestrator.run_comparison(csp, strategies)
        
        assert len(results) == 2
        # Algunos CSPs aleatorios pueden no tener solución
        # Solo verificamos que no haya errores
        assert all(metrics.error is None for metrics in results.values()), \
            f"Errors: {[m.error for m in results.values() if m.error]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

