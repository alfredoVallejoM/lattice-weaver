"""
Runner de benchmarks con captura de métricas detalladas.

Este módulo ejecuta benchmarks y captura métricas de rendimiento.
"""
import time
import tracemalloc
import statistics
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import json

from .problems import BenchmarkProblem
from .algorithms import CSPSolver, SolutionStats


@dataclass
class BenchmarkResult:
    """
    Resultado de un benchmark.
    
    Attributes:
        problem_name: Nombre del problema
        algorithm: Nombre del algoritmo
        time_ms: Tiempo de ejecución en milisegundos
        memory_mb: Memoria pico en MB
        nodes_explored: Nodos explorados
        backtracks: Backtracks realizados
        solutions_found: Número de soluciones encontradas
        success: Si se completó exitosamente
    """
    problem_name: str
    algorithm: str
    time_ms: float
    memory_mb: float
    nodes_explored: int
    backtracks: int
    solutions_found: int
    success: bool
    error_message: Optional[str] = None


@dataclass
class AggregatedResult:
    """
    Resultado agregado de múltiples ejecuciones.
    
    Attributes:
        problem_name: Nombre del problema
        algorithm: Nombre del algoritmo
        time_mean_ms: Tiempo promedio
        time_std_ms: Desviación estándar del tiempo
        time_min_ms: Tiempo mínimo
        time_max_ms: Tiempo máximo
        memory_mean_mb: Memoria promedio
        memory_std_mb: Desviación estándar de memoria
        nodes_mean: Nodos promedio
        backtracks_mean: Backtracks promedio
        solutions_found: Soluciones encontradas
        runs: Número de ejecuciones
        success_rate: Tasa de éxito
    """
    problem_name: str
    algorithm: str
    time_mean_ms: float
    time_std_ms: float
    time_min_ms: float
    time_max_ms: float
    memory_mean_mb: float
    memory_std_mb: float
    nodes_mean: float
    backtracks_mean: float
    solutions_found: int
    runs: int
    success_rate: float


class BenchmarkRunner:
    """
    Runner de benchmarks con captura de métricas.
    """
    
    def __init__(self, warmup_runs: int = 1, benchmark_runs: int = 3):
        """
        Inicializa el runner.
        
        Args:
            warmup_runs: Número de ejecuciones de calentamiento
            benchmark_runs: Número de ejecuciones de benchmark
        """
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs
        self.results: List[BenchmarkResult] = []
    
    def run_single(self, problem: BenchmarkProblem, solver: CSPSolver, 
                   algorithm_name: str, max_solutions: int = 1) -> BenchmarkResult:
        """
        Ejecuta un benchmark individual.
        
        Args:
            problem: Problema a resolver
            solver: Solver a usar
            algorithm_name: Nombre del algoritmo
            max_solutions: Número máximo de soluciones
        
        Returns:
            Resultado del benchmark
        """
        try:
            # Iniciar medición de memoria
            tracemalloc.start()
            
            # Medir tiempo
            start_time = time.perf_counter()
            stats = solver.solve(problem.problem, max_solutions=max_solutions)
            end_time = time.perf_counter()
            
            # Obtener memoria pico
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            time_ms = (end_time - start_time) * 1000
            memory_mb = peak / (1024 * 1024)
            
            return BenchmarkResult(
                problem_name=problem.name,
                algorithm=algorithm_name,
                time_ms=time_ms,
                memory_mb=memory_mb,
                nodes_explored=stats.nodes_explored,
                backtracks=stats.backtracks,
                solutions_found=len(stats.solutions),
                success=True
            )
        
        except Exception as e:
            tracemalloc.stop()
            return BenchmarkResult(
                problem_name=problem.name,
                algorithm=algorithm_name,
                time_ms=0.0,
                memory_mb=0.0,
                nodes_explored=0,
                backtracks=0,
                solutions_found=0,
                success=False,
                error_message=str(e)
            )
    
    def run_benchmark(self, problem: BenchmarkProblem, solver: CSPSolver,
                     algorithm_name: str, max_solutions: int = 1) -> AggregatedResult:
        """
        Ejecuta un benchmark con múltiples ejecuciones.
        
        Args:
            problem: Problema a resolver
            solver: Solver a usar
            algorithm_name: Nombre del algoritmo
            max_solutions: Número máximo de soluciones
        
        Returns:
            Resultado agregado
        """
        # Calentamiento
        for _ in range(self.warmup_runs):
            self.run_single(problem, solver, algorithm_name, max_solutions)
        
        # Ejecuciones de benchmark
        results = []
        for _ in range(self.benchmark_runs):
            result = self.run_single(problem, solver, algorithm_name, max_solutions)
            results.append(result)
            self.results.append(result)
        
        # Agregar resultados
        successful_results = [r for r in results if r.success]
        
        if not successful_results:
            # Todos fallaron
            return AggregatedResult(
                problem_name=problem.name,
                algorithm=algorithm_name,
                time_mean_ms=0.0,
                time_std_ms=0.0,
                time_min_ms=0.0,
                time_max_ms=0.0,
                memory_mean_mb=0.0,
                memory_std_mb=0.0,
                nodes_mean=0.0,
                backtracks_mean=0.0,
                solutions_found=0,
                runs=len(results),
                success_rate=0.0
            )
        
        times = [r.time_ms for r in successful_results]
        memories = [r.memory_mb for r in successful_results]
        nodes = [r.nodes_explored for r in successful_results]
        backtracks = [r.backtracks for r in successful_results]
        
        return AggregatedResult(
            problem_name=problem.name,
            algorithm=algorithm_name,
            time_mean_ms=statistics.mean(times),
            time_std_ms=statistics.stdev(times) if len(times) > 1 else 0.0,
            time_min_ms=min(times),
            time_max_ms=max(times),
            memory_mean_mb=statistics.mean(memories),
            memory_std_mb=statistics.stdev(memories) if len(memories) > 1 else 0.0,
            nodes_mean=statistics.mean(nodes),
            backtracks_mean=statistics.mean(backtracks),
            solutions_found=successful_results[0].solutions_found,
            runs=len(results),
            success_rate=len(successful_results) / len(results)
        )
    
    def compare_algorithms(self, problem: BenchmarkProblem, 
                          solvers: Dict[str, CSPSolver],
                          max_solutions: int = 1) -> Dict[str, AggregatedResult]:
        """
        Compara múltiples algoritmos en el mismo problema.
        
        Args:
            problem: Problema a resolver
            solvers: Diccionario {nombre: solver}
            max_solutions: Número máximo de soluciones
        
        Returns:
            Diccionario {algoritmo: resultado}
        """
        results = {}
        
        for algo_name, solver in solvers.items():
            result = self.run_benchmark(problem, solver, algo_name, max_solutions)
            results[algo_name] = result
        
        return results
    
    def save_results(self, filepath: str):
        """
        Guarda los resultados en un archivo JSON.
        
        Args:
            filepath: Ruta del archivo
        """
        data = [asdict(r) for r in self.results]
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_results(self, filepath: str):
        """
        Carga resultados desde un archivo JSON.
        
        Args:
            filepath: Ruta del archivo
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.results = [BenchmarkResult(**r) for r in data]
    
    def get_speedup_matrix(self, baseline_algorithm: str) -> Dict[str, Dict[str, float]]:
        """
        Calcula matriz de speedups relativos a un baseline.
        
        Args:
            baseline_algorithm: Algoritmo baseline
        
        Returns:
            Diccionario {problema: {algoritmo: speedup}}
        """
        speedups = {}
        
        # Agrupar resultados por problema
        by_problem = {}
        for result in self.results:
            if result.problem_name not in by_problem:
                by_problem[result.problem_name] = {}
            
            if result.algorithm not in by_problem[result.problem_name]:
                by_problem[result.problem_name][result.algorithm] = []
            
            by_problem[result.problem_name][result.algorithm].append(result.time_ms)
        
        # Calcular speedups
        for problem, algos in by_problem.items():
            if baseline_algorithm not in algos:
                continue
            
            baseline_time = statistics.mean(algos[baseline_algorithm])
            speedups[problem] = {}
            
            for algo, times in algos.items():
                algo_time = statistics.mean(times)
                speedup = baseline_time / algo_time if algo_time > 0 else 0.0
                speedups[problem][algo] = speedup
        
        return speedups
    
    def print_summary(self, baseline_algorithm: Optional[str] = None):
        """
        Imprime un resumen de los resultados.
        
        Args:
            baseline_algorithm: Algoritmo baseline para calcular speedups
        """
        if not self.results:
            print("No hay resultados disponibles")
            return
        
        # Agrupar por problema
        by_problem = {}
        for result in self.results:
            if result.problem_name not in by_problem:
                by_problem[result.problem_name] = []
            by_problem[result.problem_name].append(result)
        
        # Imprimir cada problema
        for problem_name, results in by_problem.items():
            print(f"\n{'='*70}")
            print(f"  {problem_name}")
            print(f"{'='*70}\n")
            
            # Agrupar por algoritmo
            by_algo = {}
            for r in results:
                if r.algorithm not in by_algo:
                    by_algo[r.algorithm] = []
                by_algo[r.algorithm].append(r)
            
            # Calcular baseline
            baseline_time = None
            if baseline_algorithm and baseline_algorithm in by_algo:
                baseline_time = statistics.mean([r.time_ms for r in by_algo[baseline_algorithm]])
            
            # Imprimir tabla
            print(f"{'Algoritmo':<20} {'Tiempo (ms)':<15} {'Nodos':<10} {'Backtracks':<12} {'Speedup':<10}")
            print("-" * 70)
            
            for algo, algo_results in sorted(by_algo.items()):
                times = [r.time_ms for r in algo_results]
                nodes = [r.nodes_explored for r in algo_results]
                backtracks = [r.backtracks for r in algo_results]
                
                time_mean = statistics.mean(times)
                time_std = statistics.stdev(times) if len(times) > 1 else 0.0
                nodes_mean = statistics.mean(nodes)
                backtracks_mean = statistics.mean(backtracks)
                
                speedup_str = ""
                if baseline_time and baseline_time > 0:
                    speedup = baseline_time / time_mean if time_mean > 0 else 0.0
                    speedup_str = f"{speedup:.2f}x"
                
                print(f"{algo:<20} {time_mean:>7.2f} ± {time_std:<5.2f} {nodes_mean:>9.0f} {backtracks_mean:>11.0f} {speedup_str:<10}")
            
            # Memoria
            print(f"\n{'Memoria (MB):'}")
            for algo, algo_results in sorted(by_algo.items()):
                memories = [r.memory_mb for r in algo_results]
                memory_mean = statistics.mean(memories)
                print(f"  {algo:<20} {memory_mean:.2f}")

