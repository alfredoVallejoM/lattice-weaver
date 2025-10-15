"""
Simplified Validation Benchmark - Validación Conceptual de Optimizaciones

Benchmark simplificado que valida el impacto conceptual de las optimizaciones
sin requerir integración completa.

Autor: Agente Autónomo - Lattice Weaver
Fecha: 15 de Octubre, 2025
"""

import time
import json
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Importar solo lo necesario
from lattice_weaver.utils.sparse_set import SparseSet
from lattice_weaver.utils.object_pool import ObjectPool
from lattice_weaver.utils.jit_compiler import (
    domain_intersection_jit, domain_difference_jit, mrv_score_jit
)
from lattice_weaver.utils.numpy_vectorization import NumpyVectorizer


@dataclass
class OptimizationBenchmark:
    """Resultado de benchmark de una optimización."""
    
    optimization_name: str
    operation: str
    problem_size: int
    
    baseline_time: float
    optimized_time: float
    speedup: float
    
    iterations: int


class SimplifiedValidationSuite:
    """Suite simplificada de validación de optimizaciones."""
    
    def __init__(self):
        """Inicializa suite."""
        self.results: List[OptimizationBenchmark] = []
    
    # ========================================================================
    # Benchmarks de Fase 1
    # ========================================================================
    
    def benchmark_sparse_set(self, sizes: List[int], iterations: int = 10000):
        """Benchmark de Sparse Set vs list."""
        print("\n=== Sparse Set Benchmark ===\n")
        
        for size in sizes:
            # Baseline: list operations
            values = list(range(size))
            
            start = time.time()
            for _ in range(iterations):
                # Simular operaciones comunes
                val = values[0] if values else None
                if val is not None:
                    _ = val in values
            baseline_time = time.time() - start
            
            # Optimized: SparseSet
            sparse = SparseSet(universe=values)
            
            start = time.time()
            for _ in range(iterations):
                lst = sparse.to_list()
                val = lst[0] if lst else None
                if val is not None:
                    _ = val in sparse.universe
            optimized_time = time.time() - start
            
            speedup = baseline_time / optimized_time if optimized_time > 0 else 0
            
            result = OptimizationBenchmark(
                optimization_name="Sparse Set",
                operation="contains + get_first",
                problem_size=size,
                baseline_time=baseline_time,
                optimized_time=optimized_time,
                speedup=speedup,
                iterations=iterations
            )
            
            self.results.append(result)
            print(f"Size {size}: {speedup:.2f}x speedup ({baseline_time:.4f}s -> {optimized_time:.4f}s)")
    
    def benchmark_object_pool(self, sizes: List[int], iterations: int = 10000):
        """Benchmark de Object Pool vs new allocations."""
        print("\n=== Object Pool Benchmark ===\n")
        
        for size in sizes:
            # Baseline: crear y destruir listas
            start = time.time()
            for _ in range(iterations):
                lst = [i for i in range(size)]
                del lst
            baseline_time = time.time() - start
            
            # Optimized: Object Pool
            pool = ObjectPool(factory=list, max_size=100)
            
            start = time.time()
            for _ in range(iterations):
                lst = pool.acquire()
                lst.extend(range(size))
                lst.clear()
                pool.release(lst)
            optimized_time = time.time() - start
            
            speedup = baseline_time / optimized_time if optimized_time > 0 else 0
            
            result = OptimizationBenchmark(
                optimization_name="Object Pool",
                operation="acquire + release",
                problem_size=size,
                baseline_time=baseline_time,
                optimized_time=optimized_time,
                speedup=speedup,
                iterations=iterations
            )
            
            self.results.append(result)
            print(f"Size {size}: {speedup:.2f}x speedup ({baseline_time:.4f}s -> {optimized_time:.4f}s)")
    
    # ========================================================================
    # Benchmarks de Fase 3
    # ========================================================================
    
    def benchmark_jit_compiler(self, sizes: List[int], iterations: int = 1000):
        """Benchmark de JIT compilation."""
        print("\n=== JIT Compiler Benchmark ===\n")
        
        for size in sizes:
            domain1 = np.array(range(size))
            domain2 = np.array(range(size // 2, size + size // 2))
            
            # Baseline: Python loops
            start = time.time()
            for _ in range(iterations):
                result = []
                for val in domain1:
                    if val in domain2:
                        result.append(val)
                _ = np.array(result)
            baseline_time = time.time() - start
            
            # Optimized: JIT compiled
            start = time.time()
            for _ in range(iterations):
                _ = domain_intersection_jit(domain1, domain2)
            optimized_time = time.time() - start
            
            speedup = baseline_time / optimized_time if optimized_time > 0 else 0
            
            result = OptimizationBenchmark(
                optimization_name="JIT Compiler",
                operation="domain_intersection",
                problem_size=size,
                baseline_time=baseline_time,
                optimized_time=optimized_time,
                speedup=speedup,
                iterations=iterations
            )
            
            self.results.append(result)
            print(f"Size {size}: {speedup:.2f}x speedup ({baseline_time:.4f}s -> {optimized_time:.4f}s)")
    
    def benchmark_numpy_vectorization(self, sizes: List[int], iterations: int = 1000):
        """Benchmark de NumPy vectorization."""
        print("\n=== NumPy Vectorization Benchmark ===\n")
        
        vectorizer = NumpyVectorizer(max_domain_size=max(sizes) * 2)
        
        for size in sizes:
            # Crear dominios
            domains1 = {f"V{i}": list(range(size)) for i in range(size)}
            domains2 = {f"V{i}": list(range(size // 2, size + size // 2)) for i in range(size)}
            
            # Baseline: Python loops
            start = time.time()
            for _ in range(iterations):
                result = {}
                for var in domains1:
                    d1 = set(domains1[var])
                    d2 = set(domains2[var])
                    result[var] = list(d1 & d2)
            baseline_time = time.time() - start
            
            # Optimized: NumPy vectorized
            vec1 = vectorizer.vectorize_domains(domains1)
            vec2 = vectorizer.vectorize_domains(domains2)
            
            start = time.time()
            for _ in range(iterations):
                result_vec = vectorizer.intersection_vectorized(vec1, vec2)
                _ = vectorizer.devectorize_domains(result_vec)
            optimized_time = time.time() - start
            
            speedup = baseline_time / optimized_time if optimized_time > 0 else 0
            
            result = OptimizationBenchmark(
                optimization_name="NumPy Vectorization",
                operation="intersection (multiple domains)",
                problem_size=size,
                baseline_time=baseline_time,
                optimized_time=optimized_time,
                speedup=speedup,
                iterations=iterations
            )
            
            self.results.append(result)
            print(f"Size {size}: {speedup:.2f}x speedup ({baseline_time:.4f}s -> {optimized_time:.4f}s)")
    
    # ========================================================================
    # Reporte
    # ========================================================================
    
    def generate_report(self) -> str:
        """Genera reporte de resultados."""
        report = []
        report.append("# Simplified Validation Benchmark Report")
        report.append("")
        report.append(f"**Total benchmarks**: {len(self.results)}")
        report.append("")
        
        # Agrupar por optimización
        by_optimization = {}
        for result in self.results:
            opt = result.optimization_name
            if opt not in by_optimization:
                by_optimization[opt] = []
            by_optimization[opt].append(result)
        
        for opt_name, results in sorted(by_optimization.items()):
            report.append(f"## {opt_name}")
            report.append("")
            report.append("| Problem Size | Baseline Time | Optimized Time | Speedup |")
            report.append("|--------------|---------------|----------------|---------|")
            
            for result in results:
                report.append(
                    f"| {result.problem_size} | "
                    f"{result.baseline_time:.4f}s | "
                    f"{result.optimized_time:.4f}s | "
                    f"**{result.speedup:.2f}x** |"
                )
            
            report.append("")
            
            # Estadísticas
            speedups = [r.speedup for r in results]
            avg_speedup = np.mean(speedups)
            max_speedup = np.max(speedups)
            
            report.append(f"**Average Speedup**: {avg_speedup:.2f}x")
            report.append(f"**Maximum Speedup**: {max_speedup:.2f}x")
            report.append("")
        
        # Resumen general
        report.append("## Overall Summary")
        report.append("")
        
        all_speedups = [r.speedup for r in self.results]
        report.append(f"- **Total benchmarks**: {len(self.results)}")
        report.append(f"- **Average speedup**: {np.mean(all_speedups):.2f}x")
        report.append(f"- **Median speedup**: {np.median(all_speedups):.2f}x")
        report.append(f"- **Maximum speedup**: {np.max(all_speedups):.2f}x")
        report.append(f"- **Minimum speedup**: {np.min(all_speedups):.2f}x")
        report.append("")
        
        return "\n".join(report)
    
    def save_results(self, output_dir: str = "/home/ubuntu/benchmark_results"):
        """Guarda resultados."""
        os.makedirs(output_dir, exist_ok=True)
        
        # JSON
        results_file = os.path.join(output_dir, "simplified_benchmark_results.json")
        with open(results_file, 'w') as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)
        
        # Markdown
        report_file = os.path.join(output_dir, "simplified_benchmark_report.md")
        with open(report_file, 'w') as f:
            f.write(self.generate_report())
        
        print(f"\nResults saved to {output_dir}/")


def main():
    """Ejecuta suite simplificada."""
    suite = SimplifiedValidationSuite()
    
    # Tamaños de problema
    small_sizes = [10, 50, 100]
    medium_sizes = [10, 50, 100, 500]
    large_sizes = [10, 50, 100, 500, 1000]
    
    # Fase 1
    suite.benchmark_sparse_set(small_sizes, iterations=10000)
    suite.benchmark_object_pool(small_sizes, iterations=10000)
    
    # Fase 3
    suite.benchmark_jit_compiler(medium_sizes, iterations=1000)
    suite.benchmark_numpy_vectorization(small_sizes, iterations=100)
    
    # Guardar y mostrar
    suite.save_results()
    print("\n" + suite.generate_report())


if __name__ == "__main__":
    main()

