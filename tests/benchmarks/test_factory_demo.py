"""
Demostración de uso de ProblemFactory.

Este módulo muestra cómo usar la factory para generar problemas
de forma automatizada y ejecutar benchmarks masivos.
"""
import pytest
from .factory import (
    ProblemFactory,
    ProblemGenerator,
    ProblemConfig,
    ProblemType,
    quick_problem,
    batch_problems,
    scalability_suite,
    get_quick_suite,
    get_nqueens_suite,
    get_comparison_suite,
)
from .algorithms import get_solver
from .runner import BenchmarkRunner


@pytest.mark.benchmark
class TestProblemFactory:
    """Tests de demostración de ProblemFactory."""
    
    def test_quick_problem_creation(self):
        """
        Demo: Creación rápida de problemas individuales.
        """
        print("\n" + "="*70)
        print("  DEMO: Creación Rápida de Problemas")
        print("="*70)
        
        # Crear problemas individuales
        problems = [
            quick_problem("nqueens", 6),
            quick_problem("nqueens", 8),
            quick_problem("graph_coloring", 10, difficulty="medium"),
            quick_problem("scheduling", 7, difficulty="easy"),
        ]
        
        print(f"\n✅ Creados {len(problems)} problemas:")
        for p in problems:
            print(f"  - {p.name} [{p.difficulty}]")
        
        assert len(problems) == 4
        assert all(p.problem is not None for p in problems)
    
    def test_batch_creation(self):
        """
        Demo: Creación en batch de múltiples problemas.
        """
        print("\n" + "="*70)
        print("  DEMO: Creación en Batch")
        print("="*70)
        
        # Crear batch de N-Reinas
        sizes = [4, 6, 8, 10, 12]
        problems = batch_problems("nqueens", sizes)
        
        print(f"\n✅ Creados {len(problems)} problemas N-Reinas:")
        for p in problems:
            print(f"  - {p.name}")
        
        assert len(problems) == len(sizes)
    
    def test_scalability_suite(self):
        """
        Demo: Suite para análisis de escalabilidad.
        """
        print("\n" + "="*70)
        print("  DEMO: Suite de Escalabilidad")
        print("="*70)
        
        # Crear suite de escalabilidad
        problems = scalability_suite("nqueens", min_size=4, max_size=10, step=2)
        
        print(f"\n✅ Suite de escalabilidad (step=2):")
        for p in problems:
            print(f"  - {p.name}")
        
        assert len(problems) == 4  # [4, 6, 8, 10]
    
    def test_predefined_suites(self):
        """
        Demo: Suites predefinidas.
        """
        print("\n" + "="*70)
        print("  DEMO: Suites Predefinidas")
        print("="*70)
        
        # Suite rápida
        quick = get_quick_suite()
        print(f"\n✅ Suite Rápida ({len(quick)} problemas):")
        for p in quick:
            print(f"  - {p.name} [{p.difficulty}]")
        
        # Suite de N-Reinas
        nqueens = get_nqueens_suite(max_n=8)
        print(f"\n✅ Suite N-Reinas ({len(nqueens)} problemas):")
        for p in nqueens:
            print(f"  - {p.name}")
        
        # Suite de comparación
        comparison = get_comparison_suite()
        print(f"\n✅ Suite de Comparación ({len(comparison)} categorías):")
        for category, probs in comparison.items():
            print(f"  - {category}: {len(probs)} problemas")
        
        assert len(quick) == 5
        assert len(nqueens) == 5  # [4, 5, 6, 7, 8]
        assert len(comparison) == 3  # small, medium, large
    
    def test_automated_benchmark_with_factory(self):
        """
        Demo: Benchmark automatizado usando factory.
        
        Genera problemas automáticamente y ejecuta benchmarks.
        """
        print("\n" + "="*70)
        print("  DEMO: Benchmark Automatizado con Factory")
        print("="*70)
        
        # Generar problemas automáticamente
        problems = batch_problems("nqueens", [4, 6, 8])
        
        # Configurar runner
        runner = BenchmarkRunner(warmup_runs=0, benchmark_runs=3)
        
        # Algoritmo a testear
        solver = get_solver("backtracking")
        
        print(f"\n{'Problema':<20} {'Tiempo (ms)':<15} {'Nodos':<10}")
        print("-"*50)
        
        # Ejecutar benchmarks
        for problem in problems:
            result = runner.run_benchmark(problem, solver, "Backtracking", max_solutions=1)
            
            print(f"{problem.name:<20} "
                  f"{result.time_mean_ms:>7.2f} ± {result.time_std_ms:<5.2f} "
                  f"{result.nodes_mean:>9.0f}")
            
            assert result.success_rate == 1.0
            assert result.solutions_found > 0
        
        print("\n✅ Benchmark automatizado completado")
    
    def test_massive_suite_generation(self):
        """
        Demo: Generación masiva de problemas.
        
        Muestra cómo generar grandes cantidades de problemas
        para testing exhaustivo.
        """
        print("\n" + "="*70)
        print("  DEMO: Generación Masiva de Problemas")
        print("="*70)
        
        factory = ProblemFactory(seed=42)  # Reproducible
        
        # Generar múltiples tipos de problemas
        configs = []
        
        # N-Reinas: 4 a 12
        for n in range(4, 13):
            configs.append(ProblemConfig(ProblemType.NQUEENS, size=n))
        
        # Graph Coloring: diferentes tamaños y dificultades
        for size in [5, 10, 15]:
            for difficulty in ["easy", "medium", "hard"]:
                configs.append(ProblemConfig(
                    ProblemType.GRAPH_COLORING,
                    size=size,
                    difficulty=difficulty
                ))
        
        # Scheduling: diferentes configuraciones
        for size in [5, 8, 10]:
            configs.append(ProblemConfig(ProblemType.SCHEDULING, size=size))
        
        # Generar todos los problemas
        problems = factory.create_batch(configs)
        
        print(f"\n✅ Generados {len(problems)} problemas:")
        
        # Contar por tipo
        by_type = {}
        for p in problems:
            category = p.category
            by_type[category] = by_type.get(category, 0) + 1
        
        for category, count in by_type.items():
            print(f"  - {category}: {count} problemas")
        
        assert len(problems) == 9 + 9 + 3  # 21 problemas
        
        print(f"\n📊 Total: {len(problems)} problemas listos para testing")
    
    def test_custom_problem_generation(self):
        """
        Demo: Generación de problemas personalizados.
        """
        print("\n" + "="*70)
        print("  DEMO: Problemas Personalizados")
        print("="*70)
        
        factory = ProblemFactory()
        
        # Graph coloring con parámetros personalizados
        config = ProblemConfig(
            problem_type=ProblemType.GRAPH_COLORING,
            size=10,
            custom_params={
                'num_colors': 3,
                'density': 0.5
            }
        )
        
        problem = factory.create(config)
        
        print(f"\n✅ Problema personalizado creado:")
        print(f"  - {problem.name}")
        print(f"  - Dificultad: {problem.difficulty}")
        print(f"  - Variables: {len(problem.problem.variables)}")
        print(f"  - Restricciones: {len(problem.problem.constraints)}")
        
        assert problem is not None
        assert len(problem.problem.variables) == 10


@pytest.mark.benchmark
def test_factory_integration_example():
    """
    Ejemplo completo: Factory + Runner + Comparación.
    
    Muestra un flujo completo de generación automatizada,
    ejecución de benchmarks y comparación de algoritmos.
    """
    print("\n" + "="*70)
    print("  EJEMPLO COMPLETO: Factory + Benchmark + Comparación")
    print("="*70)
    
    # 1. Generar problemas automáticamente
    print("\n1️⃣ Generando problemas...")
    problems = batch_problems("nqueens", [4, 6, 8])
    print(f"   ✅ {len(problems)} problemas generados")
    
    # 2. Configurar algoritmos
    print("\n2️⃣ Configurando algoritmos...")
    algorithms = {
        "Backtracking": get_solver("backtracking"),
        "Forward Checking": get_solver("forward_checking"),
    }
    print(f"   ✅ {len(algorithms)} algoritmos listos")
    
    # 3. Ejecutar benchmarks
    print("\n3️⃣ Ejecutando benchmarks...")
    runner = BenchmarkRunner(warmup_runs=1, benchmark_runs=3)
    
    all_results = {}
    for problem in problems:
        print(f"\n   📊 {problem.name}:")
        results = runner.compare_algorithms(problem, algorithms, max_solutions=1)
        all_results[problem.name] = results
        
        # Mostrar resultados
        for algo_name, result in sorted(results.items(), key=lambda x: x[1].time_mean_ms):
            print(f"      {algo_name:<20} {result.time_mean_ms:>7.2f}ms")
    
    # 4. Análisis global
    print("\n4️⃣ Análisis global:")
    print("-"*70)
    runner.print_summary(baseline_algorithm="Backtracking")
    
    print("\n✅ Flujo completo ejecutado exitosamente")
    
    # Validaciones
    assert len(all_results) == 3
    for problem_results in all_results.values():
        for result in problem_results.values():
            assert result.success_rate == 1.0
            assert result.solutions_found > 0


if __name__ == "__main__":
    # Ejecutar demos directamente
    print("Ejecutando demos de ProblemFactory...")
    pytest.main([__file__, "-v", "-s"])

