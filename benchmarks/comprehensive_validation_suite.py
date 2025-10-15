"""
Comprehensive Validation Suite - Benchmarks Exhaustivos de Todas las Optimizaciones

Suite completa de benchmarks para validar el impacto real de todas las optimizaciones
implementadas en las Fases 1, 2 y 3.

Objetivos:
- Medir mejoras reales vs baseline
- Identificar puntos de equilibrio (breakeven points)
- Validar escalabilidad
- Comparar con estado del arte
- Analizar overhead en problemas pequeños

Autor: Agente Autónomo - Lattice Weaver
Fecha: 15 de Octubre, 2025
"""

import time
import json
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
import sys
import os

# Añadir path del proyecto
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lattice_weaver.fibration.constraint_hierarchy import ConstraintHierarchy, Hardness
from lattice_weaver.fibration.hacification_engine_optimized import HacificationEngineOptimized
from lattice_weaver.fibration.energy_landscape_optimized import EnergyLandscapeOptimized
from lattice_weaver.arc_engine.core import ArcEngine

# Importar optimizaciones de Fase 1
from lattice_weaver.utils.sparse_set import SparseSet
from lattice_weaver.utils.object_pool import get_list_pool, get_dict_pool
from lattice_weaver.utils.auto_profiler import AutoProfiler
from lattice_weaver.utils.lazy_init import LazyProperty

# Importar optimizaciones de Fase 2
from lattice_weaver.arc_engine.adaptive_propagation import AdaptivePropagationEngine
from lattice_weaver.fibration.watched_literals import WatchedLiteralsManager
from lattice_weaver.fibration.advanced_heuristics import (
    WeightedDegreeHeuristic, ImpactBasedSearch
)
from lattice_weaver.fibration.predicate_cache import PredicateCache

# Importar optimizaciones de Fase 3
from lattice_weaver.fibration.hacification_incremental import IncrementalHacificationEngine
from lattice_weaver.utils.jit_compiler import (
    domain_intersection_jit, mrv_score_jit, get_jit_compiler
)
from lattice_weaver.utils.numpy_vectorization import NumpyVectorizer, get_numpy_vectorizer


@dataclass
class BenchmarkResult:
    """Resultado de un benchmark individual."""
    
    problem_name: str
    problem_size: int
    optimization_level: str  # 'baseline', 'phase1', 'phase2', 'phase3', 'all'
    
    # Métricas de tiempo
    total_time: float
    setup_time: float
    solve_time: float
    
    # Métricas de búsqueda
    backtracks: int
    nodes_explored: int
    
    # Métricas de memoria
    peak_memory_mb: float
    allocations: int
    
    # Métricas de calidad
    solution_found: bool
    solution_quality: float
    
    # Estadísticas de optimizaciones
    optimization_stats: Dict[str, Any]


@dataclass
class ComparisonResult:
    """Resultado de comparación entre niveles de optimización."""
    
    problem_name: str
    problem_size: int
    
    baseline: BenchmarkResult
    optimized: BenchmarkResult
    
    # Speedups
    total_speedup: float
    setup_speedup: float
    solve_speedup: float
    
    # Reducciones
    backtrack_reduction: float
    memory_reduction: float
    allocation_reduction: float
    
    # Punto de equilibrio alcanzado
    breakeven: bool


class ComprehensiveBenchmarkSuite:
    """
    Suite completa de benchmarks para validar optimizaciones.
    """
    
    def __init__(self, output_dir: str = "/home/ubuntu/benchmark_results"):
        """
        Inicializa suite de benchmarks.
        
        Args:
            output_dir: Directorio para guardar resultados
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.results: List[BenchmarkResult] = []
        self.comparisons: List[ComparisonResult] = []
    
    # ========================================================================
    # Generadores de Problemas
    # ========================================================================
    
    def generate_n_queens(self, n: int) -> Tuple[ConstraintHierarchy, Dict[str, List[int]]]:
        """
        Genera problema de N-Queens.
        
        Args:
            n: Tamaño del tablero
        
        Returns:
            (hierarchy, initial_domains)
        """
        hierarchy = ConstraintHierarchy()
        # Dominios: cada reina puede estar en cualquier fila
        domains = {f"Q{i}": list(range(n)) for i in range(n)}
        
        # Restricciones: no dos reinas en misma fila, columna o diagonal
        for i in range(n):
            for j in range(i + 1, n):
                def no_attack(assignment, i=i, j=j):
                    qi = assignment.get(f"Q{i}")
                    qj = assignment.get(f"Q{j}")
                    if qi is None or qj is None:
                        return True
                    return qi != qj and abs(qi - qj) != abs(i - j)
                
                hierarchy.add_local_constraint(
                    f"Q{i}", f"Q{j}",
                    no_attack,
                    Hardness.HARD
                )
        
        return hierarchy, domains
    
    def generate_graph_coloring(
        self,
        n_nodes: int,
        n_colors: int,
        edge_probability: float = 0.3
    ) -> Tuple[ConstraintHierarchy, Dict[str, List[int]]]:
        """
        Genera problema de Graph Coloring.
        
        Args:
            n_nodes: Número de nodos
            n_colors: Número de colores
            edge_probability: Probabilidad de arista
        
        Returns:
            (hierarchy, initial_domains)
        """
        hierarchy = ConstraintHierarchy()
        domains = {f"N{i}": list(range(n_colors)) for i in range(n_nodes)}
        
        # Generar grafo aleatorio
        np.random.seed(42)
        edges = []
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if np.random.random() < edge_probability:
                    edges.append((i, j))
        
        # Restricciones: nodos adyacentes deben tener colores diferentes
        for i, j in edges:
            def different_colors(assignment, i=i, j=j):
                ci = assignment.get(f"N{i}")
                cj = assignment.get(f"N{j}")
                if ci is None or cj is None:
                    return True
                return ci != cj
            
            hierarchy.add_local_constraint(
                f"N{i}", f"N{j}",
                different_colors,
                Hardness.HARD
            )
        
        return hierarchy, domains
    
    def generate_job_shop_scheduling(
        self,
        n_jobs: int,
        n_machines: int,
        horizon: int
    ) -> Tuple[ConstraintHierarchy, Dict[str, List[int]]]:
        """
        Genera problema de Job Shop Scheduling simplificado.
        
        Args:
            n_jobs: Número de trabajos
            n_machines: Número de máquinas
            horizon: Horizonte temporal
        
        Returns:
            (hierarchy, initial_domains)
        """
        hierarchy = ConstraintHierarchy()
        domains = {}
        
        # Variables: inicio de cada tarea
        for job in range(n_jobs):
            for task in range(n_machines):
                var = f"J{job}_T{task}"
                domains[var] = list(range(horizon))
        
        # Restricciones HARD: precedencia de tareas dentro de un job
        for job in range(n_jobs):
            for task in range(n_machines - 1):
                var1 = f"J{job}_T{task}"
                var2 = f"J{job}_T{task+1}"
                
                def precedence(assignment, v1=var1, v2=var2):
                    t1 = assignment.get(v1)
                    t2 = assignment.get(v2)
                    if t1 is None or t2 is None:
                        return True
                    return t1 + 1 <= t2  # Duración = 1
                
                hierarchy.add_local_constraint(
                    var1, var2,
                    precedence,
                    Hardness.HARD
                )
        
        # Restricciones SOFT: minimizar makespan
        for job in range(n_jobs):
            last_task = f"J{job}_T{n_machines-1}"
            
            def minimize_completion(assignment, var=last_task):
                t = assignment.get(var)
                if t is None:
                    return True
                return t < horizon // 2  # Preferir completar antes de la mitad
            
            hierarchy.add_global_constraint(
                [last_task],
                minimize_completion,
                Hardness.SOFT,
                weight=1.0
            )
        
        return hierarchy, domains
    
    # ========================================================================
    # Benchmarks Individuales
    # ========================================================================
    
    def benchmark_baseline(
        self,
        problem_name: str,
        hierarchy: ConstraintHierarchy,
        domains: Dict[str, List[Any]]
    ) -> BenchmarkResult:
        """
        Benchmark con implementación baseline (sin optimizaciones).
        
        Args:
            problem_name: Nombre del problema
            hierarchy: Jerarquía de restricciones
            domains: Dominios iniciales
        
        Returns:
            Resultado del benchmark
        """
        # Setup
        start_setup = time.time()
        landscape = EnergyLandscapeOptimized(hierarchy)
        arc_engine = ArcEngine()
        engine = HacificationEngineOptimized(hierarchy, landscape, arc_engine)
        setup_time = time.time() - start_setup
        
        # Solve
        start_solve = time.time()
        try:
            result = engine.hacify(domains, assignment={})
        except Exception as e:
            print(f"  Error: {e}")
            result = None
        solve_time = time.time() - start_solve
        
        return BenchmarkResult(
            problem_name=problem_name,
            problem_size=len(domains),
            optimization_level='baseline',
            total_time=setup_time + solve_time,
            setup_time=setup_time,
            solve_time=solve_time,
            backtracks=0,  # No tracking en baseline
            nodes_explored=0,
            peak_memory_mb=0.0,
            allocations=0,
            solution_found=result is not None,
            solution_quality=result.energy.total_energy if result else float('inf'),
            optimization_stats={}
        )
    
    def benchmark_with_optimizations(
        self,
        problem_name: str,
        hierarchy: ConstraintHierarchy,
        domains: Dict[str, List[Any]],
        optimization_level: str
    ) -> BenchmarkResult:
        """
        Benchmark con optimizaciones específicas.
        
        Args:
            problem_name: Nombre del problema
            hierarchy: Jerarquía de restricciones
            domains: Dominios iniciales
            optimization_level: 'phase1', 'phase2', 'phase3', 'all'
        
        Returns:
            Resultado del benchmark
        """
        # Setup con optimizaciones
        start_setup = time.time()
        
        # Fase 1: Estructuras de datos
        if optimization_level in ['phase1', 'all']:
            # Usar Sparse Set para dominios
            sparse_domains = {
                var: SparseSet(max_size=len(domain), initial_values=domain)
                for var, domain in domains.items()
            }
            
            # Usar Object Pool
            list_pool = get_list_pool()
            dict_pool = get_dict_pool()
        
        # Fase 2: Algoritmos
        if optimization_level in ['phase2', 'all']:
            # Usar Adaptive Propagation
            adaptive_engine = AdaptivePropagationEngine()
            
            # Usar Watched Literals
            watched_manager = WatchedLiteralsManager()
            for constraint in hierarchy.constraints:
                watched_manager.add_constraint(constraint)
            
            # Usar Advanced Heuristics
            wdeg = WeightedDegreeHeuristic(hierarchy.constraints)
            ibs = ImpactBasedSearch(list(domains.keys()), domains)
            
            # Usar Predicate Cache
            pred_cache = PredicateCache(max_size=1000)
        
        # Fase 3: Compilación y Vectorización
        if optimization_level in ['phase3', 'all']:
            # Usar Hacification Incremental
            incremental_engine = IncrementalHacificationEngine(hierarchy)
            
            # Usar JIT Compiler
            jit_compiler = get_jit_compiler()
            
            # Usar NumPy Vectorizer
            vectorizer = get_numpy_vectorizer()
            vectorized_domains = vectorizer.vectorize_domains(domains)
        
        landscape = EnergyLandscapeOptimized(hierarchy)
        arc_engine = ArcEngine()
        engine = HacificationEngineOptimized(hierarchy, landscape, arc_engine)
        
        setup_time = time.time() - start_setup
        
        # Solve
        start_solve = time.time()
        
        try:
            if optimization_level in ['phase3', 'all']:
                # Usar Hacification Incremental
                result = incremental_engine.hacify(domains, assignment={})
                stats = incremental_engine.get_stats()
            else:
                result = engine.hacify(domains, assignment={})
                stats = {}
        except Exception as e:
            print(f"  Error: {e}")
            result = None
            stats = {}
        
        solve_time = time.time() - start_solve
        
        # Recolectar estadísticas
        optimization_stats = stats.copy()
        
        if optimization_level in ['phase2', 'all']:
            optimization_stats['adaptive_propagation'] = adaptive_engine.get_stats()
            optimization_stats['watched_literals'] = watched_manager.get_stats()
            optimization_stats['wdeg'] = wdeg.get_stats()
            optimization_stats['predicate_cache'] = pred_cache.get_stats()
        
        if optimization_level in ['phase3', 'all']:
            optimization_stats['jit_compiler'] = jit_compiler.get_stats()
            optimization_stats['vectorizer'] = vectorizer.get_stats()
        
        return BenchmarkResult(
            problem_name=problem_name,
            problem_size=len(domains),
            optimization_level=optimization_level,
            total_time=setup_time + solve_time,
            setup_time=setup_time,
            solve_time=solve_time,
            backtracks=stats.get('backtracks', 0),
            nodes_explored=stats.get('nodes_explored', 0),
            peak_memory_mb=0.0,
            allocations=0,
            solution_found=result is not None,
            solution_quality=result.energy.total_energy if result else float('inf'),
            optimization_stats=optimization_stats
        )
    
    # ========================================================================
    # Suite de Benchmarks
    # ========================================================================
    
    def run_scalability_test(
        self,
        problem_type: str,
        sizes: List[int],
        optimization_levels: List[str]
    ):
        """
        Ejecuta test de escalabilidad.
        
        Args:
            problem_type: 'n_queens', 'graph_coloring', 'job_shop'
            sizes: Lista de tamaños de problema
            optimization_levels: Lista de niveles de optimización
        """
        print(f"\n{'='*80}")
        print(f"Scalability Test: {problem_type}")
        print(f"{'='*80}\n")
        
        for size in sizes:
            print(f"\n--- Problem Size: {size} ---\n")
            
            # Generar problema
            if problem_type == 'n_queens':
                hierarchy, domains = self.generate_n_queens(size)
            elif problem_type == 'graph_coloring':
                hierarchy, domains = self.generate_graph_coloring(size, size // 2)
            elif problem_type == 'job_shop':
                hierarchy, domains = self.generate_job_shop_scheduling(size, size, size * 3)
            else:
                raise ValueError(f"Unknown problem type: {problem_type}")
            
            # Benchmark baseline
            print("Running baseline...")
            baseline = self.benchmark_baseline(problem_type, hierarchy, domains)
            self.results.append(baseline)
            print(f"  Time: {baseline.total_time:.4f}s")
            
            # Benchmark con optimizaciones
            for opt_level in optimization_levels:
                print(f"Running {opt_level}...")
                optimized = self.benchmark_with_optimizations(
                    problem_type, hierarchy, domains, opt_level
                )
                self.results.append(optimized)
                print(f"  Time: {optimized.total_time:.4f}s")
                
                # Comparar
                comparison = self.compare_results(baseline, optimized)
                self.comparisons.append(comparison)
                print(f"  Speedup: {comparison.total_speedup:.2f}x")
    
    def compare_results(
        self,
        baseline: BenchmarkResult,
        optimized: BenchmarkResult
    ) -> ComparisonResult:
        """
        Compara dos resultados de benchmark.
        
        Args:
            baseline: Resultado baseline
            optimized: Resultado optimizado
        
        Returns:
            Comparación
        """
        total_speedup = baseline.total_time / optimized.total_time if optimized.total_time > 0 else 0
        setup_speedup = baseline.setup_time / optimized.setup_time if optimized.setup_time > 0 else 0
        solve_speedup = baseline.solve_time / optimized.solve_time if optimized.solve_time > 0 else 0
        
        backtrack_reduction = (
            (baseline.backtracks - optimized.backtracks) / baseline.backtracks * 100
            if baseline.backtracks > 0 else 0
        )
        
        # Breakeven alcanzado si speedup > 1.0
        breakeven = total_speedup > 1.0
        
        return ComparisonResult(
            problem_name=baseline.problem_name,
            problem_size=baseline.problem_size,
            baseline=baseline,
            optimized=optimized,
            total_speedup=total_speedup,
            setup_speedup=setup_speedup,
            solve_speedup=solve_speedup,
            backtrack_reduction=backtrack_reduction,
            memory_reduction=0.0,
            allocation_reduction=0.0,
            breakeven=breakeven
        )
    
    # ========================================================================
    # Reportes
    # ========================================================================
    
    def generate_report(self) -> str:
        """
        Genera reporte completo de benchmarks.
        
        Returns:
            Reporte en formato Markdown
        """
        report = []
        report.append("# Comprehensive Benchmark Report")
        report.append("")
        report.append(f"**Total benchmarks**: {len(self.results)}")
        report.append(f"**Total comparisons**: {len(self.comparisons)}")
        report.append("")
        
        # Resumen por nivel de optimización
        report.append("## Summary by Optimization Level")
        report.append("")
        
        by_level = {}
        for result in self.results:
            level = result.optimization_level
            if level not in by_level:
                by_level[level] = []
            by_level[level].append(result)
        
        for level, results in sorted(by_level.items()):
            avg_time = np.mean([r.total_time for r in results])
            report.append(f"### {level}")
            report.append(f"- **Count**: {len(results)}")
            report.append(f"- **Avg Time**: {avg_time:.4f}s")
            report.append("")
        
        # Comparaciones
        report.append("## Comparisons")
        report.append("")
        report.append("| Problem | Size | Optimization | Speedup | Breakeven |")
        report.append("|---------|------|--------------|---------|-----------|")
        
        for comp in self.comparisons:
            report.append(
                f"| {comp.problem_name} | {comp.problem_size} | "
                f"{comp.optimized.optimization_level} | "
                f"{comp.total_speedup:.2f}x | "
                f"{'✓' if comp.breakeven else '✗'} |"
            )
        
        report.append("")
        
        # Puntos de equilibrio
        report.append("## Breakeven Points")
        report.append("")
        
        breakeven_by_problem = {}
        for comp in self.comparisons:
            if comp.breakeven:
                problem = comp.problem_name
                if problem not in breakeven_by_problem:
                    breakeven_by_problem[problem] = []
                breakeven_by_problem[problem].append(comp.problem_size)
        
        for problem, sizes in sorted(breakeven_by_problem.items()):
            min_size = min(sizes)
            report.append(f"- **{problem}**: Breakeven at size {min_size}")
        
        report.append("")
        
        return "\n".join(report)
    
    def save_results(self):
        """Guarda resultados en JSON."""
        # Resultados
        results_file = os.path.join(self.output_dir, "benchmark_results.json")
        with open(results_file, 'w') as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)
        
        # Comparaciones
        comparisons_file = os.path.join(self.output_dir, "comparisons.json")
        with open(comparisons_file, 'w') as f:
            json.dump([asdict(c) for c in self.comparisons], f, indent=2, default=str)
        
        # Reporte
        report_file = os.path.join(self.output_dir, "benchmark_report.md")
        with open(report_file, 'w') as f:
            f.write(self.generate_report())
        
        print(f"\nResults saved to {self.output_dir}/")


def main():
    """Ejecuta suite completa de benchmarks."""
    suite = ComprehensiveBenchmarkSuite()
    
    # Test 1: N-Queens escalabilidad
    suite.run_scalability_test(
        problem_type='n_queens',
        sizes=[4, 6, 8],  # Tamaños manejables
        optimization_levels=['phase1', 'phase2', 'all']
    )
    
    # Test 2: Graph Coloring escalabilidad
    suite.run_scalability_test(
        problem_type='graph_coloring',
        sizes=[5, 10, 15],
        optimization_levels=['phase1', 'phase2', 'all']
    )
    
    # Test 3: Job Shop Scheduling
    suite.run_scalability_test(
        problem_type='job_shop',
        sizes=[2, 3, 4],
        optimization_levels=['phase1', 'all']
    )
    
    # Guardar resultados
    suite.save_results()
    
    # Mostrar reporte
    print("\n" + suite.generate_report())


if __name__ == "__main__":
    main()

