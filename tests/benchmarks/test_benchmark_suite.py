"""
Suite de tests de benchmarking.

Este módulo ejecuta benchmarks comparativos con algoritmos SOTA.
"""
import pytest
from .problems import STANDARD_SUITE, create_nqueens
from .algorithms import get_solver
from .runner import BenchmarkRunner


@pytest.mark.benchmark
class TestBenchmarkSuite:
    """Tests de benchmarking comparativo."""
    
    def test_nqueens_comparison(self):
        """
        Benchmark: N-Reinas (n=4, 6, 8) - Comparación de algoritmos.
        
        Compara el rendimiento de diferentes algoritmos en el problema
        de N-Reinas con diferentes tamaños.
        """
        print("\n" + "="*70)
        print("  BENCHMARK: N-Reinas - Comparación de Algoritmos")
        print("="*70)
        
        runner = BenchmarkRunner(warmup_runs=1, benchmark_runs=3)
        
        # Problemas a testear
        problems = [create_nqueens(n) for n in [4, 6, 8]]
        
        # Algoritmos a comparar
        algorithms = {
            "Backtracking": get_solver("backtracking"),
            "Forward Checking": get_solver("forward_checking"),
            "AC-3": get_solver("ac3"),
        }
        
        # Ejecutar benchmarks
        for problem in problems:
            print(f"\n{problem.name}:")
            print("-" * 70)
            
            results = runner.compare_algorithms(problem, algorithms, max_solutions=1)
            
            # Encontrar baseline (el más lento)
            baseline_time = max(r.time_mean_ms for r in results.values())
            
            # Imprimir resultados
            print(f"{'Algoritmo':<20} {'Tiempo (ms)':<15} {'Nodos':<10} {'Backtracks':<12} {'Speedup':<10}")
            print("-" * 70)
            
            for algo_name, result in sorted(results.items(), key=lambda x: x[1].time_mean_ms):
                speedup = baseline_time / result.time_mean_ms if result.time_mean_ms > 0 else 0.0
                
                print(f"{algo_name:<20} "
                      f"{result.time_mean_ms:>7.2f} ± {result.time_std_ms:<5.2f} "
                      f"{result.nodes_mean:>9.0f} "
                      f"{result.backtracks_mean:>11.0f} "
                      f"{speedup:.2f}x")
            
            # Memoria
            print(f"\nMemoria (MB):")
            for algo_name, result in sorted(results.items()):
                print(f"  {algo_name:<20} {result.memory_mean_mb:.2f}")
            
            # Validar que encontraron soluciones
            for algo_name, result in results.items():
                assert result.solutions_found > 0, f"{algo_name} debe encontrar al menos una solución"
                assert result.success_rate == 1.0, f"{algo_name} debe tener 100% de éxito"
        
        # Resumen global
        print("\n" + "="*70)
        print("  RESUMEN GLOBAL")
        print("="*70)
        runner.print_summary(baseline_algorithm="AC-3")
    
    def test_graph_coloring_comparison(self):
        """
        Benchmark: Graph Coloring - Comparación de algoritmos.
        
        Compara el rendimiento en problemas de coloreo de grafos.
        """
        print("\n" + "="*70)
        print("  BENCHMARK: Graph Coloring - Comparación de Algoritmos")
        print("="*70)
        
        runner = BenchmarkRunner(warmup_runs=1, benchmark_runs=3)
        
        # Obtener problemas de coloreo
        from .problems import get_problems_by_category
        problems = get_problems_by_category("graph_coloring")
        
        # Algoritmos a comparar
        algorithms = {
            "Backtracking": get_solver("backtracking"),
            "Forward Checking": get_solver("forward_checking"),
            "AC-3": get_solver("ac3"),
        }
        
        # Ejecutar benchmarks
        for problem in problems:
            print(f"\n{problem.name}:")
            print("-" * 70)
            
            results = runner.compare_algorithms(problem, algorithms, max_solutions=1)
            
            # Encontrar baseline
            baseline_time = max(r.time_mean_ms for r in results.values())
            
            # Imprimir resultados
            print(f"{'Algoritmo':<20} {'Tiempo (ms)':<15} {'Speedup':<10}")
            print("-" * 70)
            
            for algo_name, result in sorted(results.items(), key=lambda x: x[1].time_mean_ms):
                speedup = baseline_time / result.time_mean_ms if result.time_mean_ms > 0 else 0.0
                
                print(f"{algo_name:<20} "
                      f"{result.time_mean_ms:>7.2f} ± {result.time_std_ms:<5.2f} "
                      f"{speedup:.2f}x")
            
            # Validar
            for algo_name, result in results.items():
                assert result.solutions_found > 0, f"{algo_name} debe encontrar solución"
    
    @pytest.mark.slow
    def test_full_benchmark_suite(self):
        """
        Benchmark: Suite completa de problemas.
        
        Ejecuta benchmarks en todos los problemas estándar.
        """
        print("\n" + "="*70)
        print("  BENCHMARK: Suite Completa")
        print("="*70)
        
        runner = BenchmarkRunner(warmup_runs=1, benchmark_runs=3)
        
        # Algoritmos a comparar
        algorithms = {
            "Backtracking": get_solver("backtracking"),
            "Forward Checking": get_solver("forward_checking"),
            "AC-3": get_solver("ac3"),
        }
        
        # Ejecutar en todos los problemas
        for problem in STANDARD_SUITE:
            print(f"\n{problem.name} [{problem.difficulty}]:")
            print("-" * 70)
            
            results = runner.compare_algorithms(problem, algorithms, max_solutions=1)
            
            # Imprimir resultados compactos
            for algo_name, result in sorted(results.items(), key=lambda x: x[1].time_mean_ms):
                print(f"  {algo_name:<20} {result.time_mean_ms:>7.2f}ms  "
                      f"({result.nodes_mean:.0f} nodos)")
        
        # Resumen final
        print("\n" + "="*70)
        print("  RESUMEN FINAL")
        print("="*70)
        runner.print_summary(baseline_algorithm="Backtracking")
        
        # Guardar resultados
        runner.save_results("tests/data/benchmark_results.json")
        print("\n✅ Resultados guardados en tests/data/benchmark_results.json")
    
    def test_scalability_analysis(self):
        """
        Benchmark: Análisis de escalabilidad.
        
        Analiza cómo escala el rendimiento con el tamaño del problema.
        """
        print("\n" + "="*70)
        print("  BENCHMARK: Análisis de Escalabilidad (N-Reinas)")
        print("="*70)
        
        runner = BenchmarkRunner(warmup_runs=1, benchmark_runs=3)
        
        # Diferentes tamaños
        sizes = [4, 5, 6, 7, 8]
        
        # Un solo algoritmo para análisis de escalabilidad
        algorithm = get_solver("forward_checking")
        
        print(f"\n{'N':<5} {'Tiempo (ms)':<15} {'Nodos':<12} {'Backtracks':<12}")
        print("-" * 50)
        
        for n in sizes:
            problem = create_nqueens(n)
            result = runner.run_benchmark(problem, algorithm, "Forward Checking", max_solutions=1)
            
            print(f"{n:<5} "
                  f"{result.time_mean_ms:>7.2f} ± {result.time_std_ms:<5.2f} "
                  f"{result.nodes_mean:>11.0f} "
                  f"{result.backtracks_mean:>11.0f}")
        
        print("\n📊 Observación: El tiempo crece exponencialmente con N")


@pytest.mark.benchmark
def test_quick_benchmark():
    """
    Test rápido de benchmarking para validación.
    
    Ejecuta un benchmark simple para verificar que todo funciona.
    """
    print("\n" + "="*70)
    print("  QUICK BENCHMARK: N-Reinas (n=4)")
    print("="*70)
    
    runner = BenchmarkRunner(warmup_runs=0, benchmark_runs=1)
    problem = create_nqueens(4)
    
    algorithms = {
        "Backtracking": get_solver("backtracking"),
        "Forward Checking": get_solver("forward_checking"),
    }
    
    results = runner.compare_algorithms(problem, algorithms, max_solutions=1)
    
    print(f"\n{'Algoritmo':<20} {'Tiempo (ms)':<15} {'Soluciones':<12}")
    print("-" * 50)
    
    for algo_name, result in results.items():
        print(f"{algo_name:<20} {result.time_mean_ms:>7.2f} {result.solutions_found:>11}")
        
        # Validaciones
        assert result.success_rate == 1.0, f"{algo_name} debe completarse exitosamente"
        assert result.solutions_found > 0, f"{algo_name} debe encontrar soluciones"
    
    print("\n✅ Benchmark completado exitosamente")

