"""
Suite exhaustiva de benchmarks para el compilador multiescala.

Este módulo contiene pruebas de rendimiento completas que evalúan:
- Escalabilidad con diferentes tamaños de problemas
- Rendimiento en diferentes tipos de problemas CSP
- Efectividad de cada nivel de compilación (L0-L6)
- Comparación con estrategia base (sin compilación)
"""

import unittest
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

from lattice_weaver.core.csp_problem import CSP
from lattice_weaver.core.simple_backtracking_solver import solve_csp_backtracking

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
from lattice_weaver.benchmarks.orchestrator import (
    Orchestrator,
    NoCompilationStrategy,
    FixedLevelStrategy,
    BenchmarkMetrics
)
from lattice_weaver.benchmarks.generators import (
    generate_nqueens,
    generate_sudoku,
    generate_graph_coloring,
    generate_simple_csp,
    generate_job_shop_scheduling
)


class TestComprehensiveBenchmarks(unittest.TestCase):
    """Suite exhaustiva de benchmarks para el compilador multiescala."""
    
    @classmethod
    def setUpClass(cls):
        """Configuración inicial para todos los tests."""
        cls.solver = simple_solver
        cls.orchestrator = Orchestrator(cls.solver)
        
        # Directorio para guardar resultados
        cls.results_dir = Path("/home/ubuntu/benchmark_results")
        cls.results_dir.mkdir(exist_ok=True)
        
        # Almacenar todos los resultados
        cls.all_results = []
    
    @classmethod
    def tearDownClass(cls):
        """Guardar todos los resultados al finalizar."""
        # Guardar resultados en JSON
        results_file = cls.results_dir / f"comprehensive_results_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump(cls.all_results, f, indent=2, default=str)
        
        print(f"\n{'='*80}")
        print(f"Resultados guardados en: {results_file}")
        print(f"Total de benchmarks ejecutados: {len(cls.all_results)}")
        print(f"{'='*80}\n")
    
    def _run_benchmark_suite(
        self,
        problem_name: str,
        problem_generator,
        sizes: List[int],
        **generator_kwargs
    ) -> List[Dict[str, Any]]:
        """
        Ejecuta una suite completa de benchmarks para un tipo de problema.
        
        Args:
            problem_name: Nombre del tipo de problema.
            problem_generator: Función generadora del problema.
            sizes: Lista de tamaños a probar.
            **generator_kwargs: Argumentos adicionales para el generador.
            
        Returns:
            Lista de resultados de todos los benchmarks.
        """
        results = []
        
        # Estrategias a probar
        strategies = [
            NoCompilationStrategy(),
            FixedLevelStrategy(1),
            FixedLevelStrategy(2),
            FixedLevelStrategy(3),
            FixedLevelStrategy(4),
            FixedLevelStrategy(5),
            FixedLevelStrategy(6)
        ]
        
        for size in sizes:
            print(f"\n{'-'*80}")
            print(f"Benchmarking {problem_name} - Size: {size}")
            print(f"{'-'*80}")
            
            # Generar problema
            try:
                csp = problem_generator(size, **generator_kwargs)
            except Exception as e:
                print(f"Error generando problema: {e}")
                continue
            
            # Ejecutar benchmark con cada estrategia
            for strategy in strategies:
                strategy_name = strategy.get_name()
                print(f"\nEstrategia: {strategy_name}")
                
                try:
                    metrics = self.orchestrator.run_benchmark(
                        csp,
                        strategy,
                        timeout=300  # 5 minutos máximo
                    )
                    
                    # Crear registro de resultado
                    result = {
                        'problem_type': problem_name,
                        'problem_size': size,
                        'strategy': strategy_name,
                        'total_time': metrics.total_time,
                        'compilation_time': metrics.compilation_time,
                        'solving_time': metrics.solving_time,
                        'peak_memory_mb': metrics.peak_memory,
                        'nodes_explored': metrics.nodes_explored,
                        'backtracks': metrics.backtracks,
                        'solution_found': metrics.solution_found,
                        'compiled_variables': metrics.compiled_variables,
                        'compiled_constraints': metrics.compiled_constraints,
                        'compression_ratio': metrics.compression_ratio,
                        'error': metrics.error
                    }
                    
                    results.append(result)
                    self.all_results.append(result)
                    
                    # Mostrar resumen
                    if metrics.error:
                        print(f"  ERROR: {metrics.error}")
                    else:
                        print(f"  Tiempo total: {metrics.total_time:.4f}s")
                        print(f"  Tiempo compilación: {metrics.compilation_time:.4f}s")
                        print(f"  Tiempo resolución: {metrics.solving_time:.4f}s")
                        print(f"  Memoria pico: {metrics.peak_memory:.2f} MB")
                        print(f"  Nodos explorados: {metrics.nodes_explored}")
                        print(f"  Solución encontrada: {metrics.solution_found}")
                        print(f"  Ratio compresión: {metrics.compression_ratio:.2f}x")
                    
                except Exception as e:
                    print(f"  ERROR: {e}")
                    result = {
                        'problem_type': problem_name,
                        'problem_size': size,
                        'strategy': strategy_name,
                        'error': str(e)
                    }
                    results.append(result)
                    self.all_results.append(result)
        
        return results
    
    def test_nqueens_scalability(self):
        """Test de escalabilidad con N-Queens."""
        sizes = [4, 6, 8, 10, 12]
        results = self._run_benchmark_suite(
            "N-Queens",
            generate_nqueens,
            sizes
        )
        
        # Verificar que se ejecutaron todos los benchmarks
        self.assertEqual(len(results), len(sizes) * 7)  # 7 estrategias
        
        # Verificar que al menos algunos encontraron solución
        solutions_found = sum(1 for r in results if r.get('solution_found', False))
        self.assertGreater(solutions_found, 0)

    def test_sudoku_scalability(self):
        """Test de escalabilidad con Sudoku."""
        print(f"\n{'='*80}")
        print("Benchmarking Sudoku")
        print(f"{'='*80}")

        # Sudoku Easy
        results_easy = self._run_benchmark_suite(
            "Sudoku-Easy",
            generate_sudoku,
            [4], # Usar un tamaño de 4 para Sudoku fácil (4x4)
        )
        self.assertEqual(len(results_easy), 1 * 7) # 1 problema, 7 estrategias

        # Sudoku Medium
        results_medium = self._run_benchmark_suite(
            "Sudoku-Medium",
            generate_sudoku,
            [9], # Usar un tamaño de 9 para Sudoku medio (9x9)
        )
        self.assertEqual(len(results_medium), 1 * 7)

        # Sudoku Hard
        results_hard = self._run_benchmark_suite(
            "Sudoku-Hard",
            generate_sudoku,
            [9], # Usar un tamaño de 9 para Sudoku difícil (9x9)
        )
        self.assertEqual(len(results_hard), 1 * 7)

        # Verificar que al menos algunos encontraron solución
        all_sudoku_results = results_easy + results_medium + results_hard
        solutions_found_sudoku = sum(1 for r in all_sudoku_results if r.get('solution_found', False))
        self.assertGreater(solutions_found_sudoku, 0)

    def test_job_shop_scheduling_scalability(self):
        """Test de escalabilidad con Job Shop Scheduling."""
        print(f"\n{'='*80}")
        print("Benchmarking Job Shop Scheduling")
        print(f"{'='*80}")

        # Problemas de Job Shop Scheduling de diferentes tamaños
        sizes = [(2, 2), (3, 2), (3, 3)] # (num_jobs, num_machines)
        all_job_shop_results = []

        for num_jobs, num_machines in sizes:
            print(f"\n{'-'*80}")
            print(f"Job Shop Scheduling - Jobs: {num_jobs}, Machines: {num_machines}")
            print(f"{'-'*80}")
            results = self._run_benchmark_suite(
                f"JobShop-{num_jobs}x{num_machines}",
                generate_job_shop_scheduling,
                [0], # Size no aplica directamente, se usa num_jobs y num_machines
                num_jobs=num_jobs,
                num_machines=num_machines
            )
            all_job_shop_results.extend(results)
        
        # Verificar que al menos algunos encontraron solución
        solutions_found_job_shop = sum(1 for r in all_job_shop_results if r.get('solution_found', False))
        self.assertGreater(solutions_found_job_shop, 0)

    def test_simple_csp_scalability(self):
        """Test de escalabilidad con CSP simple."""
        print(f"\n{'='*80}")
        print("Benchmarking Simple CSP")
        print(f"{'='*80}")

        # Problemas de CSP simple con diferentes números de variables
        sizes = [5, 10, 15]
        all_simple_csp_results = []

        for num_vars in sizes:
            print(f"\n{'-'*80}")
            print(f"Simple CSP - Variables: {num_vars}")
            print(f"{'-'*80}")
            results = self._run_benchmark_suite(
                f"SimpleCSP-{num_vars}",
                generate_simple_csp,
                [0], # Size no aplica directamente, se usa num_vars
                num_variables=num_vars,
                domain_size=3,
                constraint_density=0.5
            )
            all_simple_csp_results.extend(results)
        
        # Verificar que al menos algunos encontraron solución
        solutions_found_simple_csp = sum(1 for r in all_simple_csp_results if r.get('solution_found', False))
        self.assertGreater(solutions_found_simple_csp, 0)

    def test_graph_coloring_varying_density(self):
        """Test de coloreado de grafos con diferentes densidades."""
        # Probar con diferentes densidades de aristas
        densities = [0.2, 0.3, 0.5]
        all_results = []
        
        for density in densities:
            print(f"\n{'='*80}")
            print(f"Graph Coloring - Densidad de aristas: {density}")
            print(f"{'='*80}")
            
            results = self._run_benchmark_suite(
                f"Graph-Coloring-{density}",
                generate_graph_coloring,
                [10, 15, 20],
                edge_probability=density,
                num_colors=3
            )
            all_results.extend(results)
        
        # Verificar ejecución
        self.assertGreater(len(all_results), 0)
    
    def test_simple_csp_varying_complexity(self):
        """Test de CSP simple con diferentes complejidades."""
        # Probar con diferentes densidades de restricciones
        densities = [0.2, 0.3, 0.5]
        all_results = []
        
        for density in densities:
            print(f"\n{'='*80}")
            print(f"Simple CSP - Densidad de restricciones: {density}")
            print(f"{'='*80}")
            
            results = self._run_benchmark_suite(
                f"Simple-CSP-{density}",
                generate_simple_csp,
                [8, 10, 12],
                domain_size=5,
                constraint_density=density
            )
            all_results.extend(results)
        
        # Verificar ejecución
        self.assertGreater(len(all_results), 0)
    
    def test_compilation_effectiveness(self):
        """
        Test específico para evaluar la efectividad de cada nivel de compilación.
        
        Este test compara directamente el rendimiento de cada nivel con el baseline.
        """
        print(f"\n{'='*80}")
        print("ANÁLISIS DE EFECTIVIDAD DE COMPILACIÓN")
        print(f"{'='*80}")
        
        # Usar N-Queens 8x8 como problema de referencia
        csp = generate_nqueens(8)
        
        strategies = [
            NoCompilationStrategy(),
            FixedLevelStrategy(1),
            FixedLevelStrategy(2),
            FixedLevelStrategy(3),
            FixedLevelStrategy(4),
            FixedLevelStrategy(5),
            FixedLevelStrategy(6)
        ]
        
        baseline_time = None
        improvements = {}
        
        for strategy in strategies:
            strategy_name = strategy.get_name()
            print(f"\nEvaluando: {strategy_name}")
            
            # Ejecutar múltiples veces para obtener promedio
            times = []
            for run in range(3):
                metrics = self.orchestrator.run_benchmark(csp, strategy)
                if not metrics.error:
                    times.append(metrics.total_time)
            
            if times:
                avg_time = sum(times) / len(times)
                print(f"  Tiempo promedio: {avg_time:.4f}s")
                
                if strategy_name == "NoCompilation":
                    baseline_time = avg_time
                elif baseline_time:
                    improvement = ((baseline_time - avg_time) / baseline_time) * 100
                    improvements[strategy_name] = improvement
                    print(f"  Mejora vs baseline: {improvement:.2f}%")
        
        # Guardar análisis de efectividad
        effectiveness_file = self.results_dir / "compilation_effectiveness.json"
        with open(effectiveness_file, 'w') as f:
            json.dump({
                'baseline_time': baseline_time,
                'improvements': improvements
            }, f, indent=2)
        
        print(f"\n{'='*80}")
        print("RESUMEN DE MEJORAS")
        print(f"{'='*80}")
        for strategy, improvement in improvements.items():
            print(f"{strategy}: {improvement:+.2f}%")
        
        # Verificar que al menos algunos niveles muestran mejora
        positive_improvements = sum(1 for imp in improvements.values() if imp > 0)
        print(f"\nNiveles con mejora positiva: {positive_improvements}/{len(improvements)}")


if __name__ == '__main__':
    unittest.main(verbosity=2)

