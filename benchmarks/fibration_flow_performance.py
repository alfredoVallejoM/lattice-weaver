"""
Benchmarks Empíricos de Rendimiento para Fibration Flow

Este módulo implementa benchmarks completos para medir las mejoras de rendimiento
entre las versiones original y optimizada de Fibration Flow.

Métricas medidas:
1. Tiempo de ejecución (wall-clock time)
2. Uso de memoria (peak memory, allocations)
3. Número de backtracks
4. Número de nodos explorados
5. Calidad de soluciones (energía final)
6. Escalabilidad (problemas de diferentes tamaños)

Autor: Agente Autónomo - Lattice Weaver
Fecha: 15 de Octubre, 2025
"""

import time
import tracemalloc
import gc
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
import json

from lattice_weaver.fibration.fibration_search_solver import FibrationSearchSolver
# TODO: Descomentar cuando se integre el componente
# from lattice_weaver.fibration.fibration_search_solver_enhanced import FibrationSearchSolverEnhanced
from lattice_weaver.fibration.hacification_engine import HacificationEngine
# TODO: Descomentar cuando se integre el componente
# from lattice_weaver.fibration.hacification_engine_optimized import HacificationEngineOptimized
from lattice_weaver.fibration.constraint_hierarchy import ConstraintHierarchy, Hardness
from lattice_weaver.fibration.energy_landscape_optimized import EnergyLandscapeOptimized
# TODO: Descomentar cuando se integre el componente
# from lattice_weaver.arc_engine.core import ArcEngine


@dataclass
class BenchmarkResult:
    """Resultado de un benchmark individual."""
    name: str
    version: str  # 'original' o 'optimized'
    problem_size: int
    
    # Métricas de tiempo
    wall_time: float
    
    # Métricas de memoria
    peak_memory_mb: float
    memory_allocations: int
    
    # Métricas de búsqueda
    nodes_explored: int
    backtracks: int
    backjumps: int
    conflicts_analyzed: int
    
    # Métricas de solución
    solution_found: bool
    solution_energy: float
    num_solutions: int
    
    # Métricas de heurísticas
    mrv_calls: int = 0
    lcv_calls: int = 0
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComparisonReport:
    """Reporte de comparación entre versiones."""
    benchmark_name: str
    problem_size: int
    
    original: BenchmarkResult
    optimized: BenchmarkResult
    
    # Speedups
    time_speedup: float
    memory_reduction: float
    backtrack_reduction: float
    
    def to_dict(self) -> Dict:
        """Convierte el reporte a diccionario."""
        return {
            'benchmark': self.benchmark_name,
            'problem_size': self.problem_size,
            'original': {
                'wall_time': self.original.wall_time,
                'peak_memory_mb': self.original.peak_memory_mb,
                'backtracks': self.original.backtracks,
                'nodes_explored': self.original.nodes_explored,
                'solution_energy': self.original.solution_energy
            },
            'optimized': {
                'wall_time': self.optimized.wall_time,
                'peak_memory_mb': self.optimized.peak_memory_mb,
                'backtracks': self.optimized.backtracks,
                'nodes_explored': self.optimized.nodes_explored,
                'solution_energy': self.optimized.solution_energy,
                'backjumps': self.optimized.backjumps
            },
            'improvements': {
                'time_speedup': f"{self.time_speedup:.2f}x",
                'memory_reduction': f"{self.memory_reduction:.2f}x",
                'backtrack_reduction': f"{self.backtrack_reduction:.2f}x"
            }
        }


class FibrationFlowBenchmarkSuite:
    """Suite de benchmarks para Fibration Flow."""
    
    def __init__(self):
        """Inicializa la suite de benchmarks."""
        self.results: List[BenchmarkResult] = []
        self.comparisons: List[ComparisonReport] = []
    
    def run_all_benchmarks(self) -> List[ComparisonReport]:
        """
        Ejecuta todos los benchmarks y genera reportes de comparación.
        
        Returns:
            Lista de reportes de comparación
        """
        print("=" * 80)
        print("FIBRATION FLOW PERFORMANCE BENCHMARKS")
        print("=" * 80)
        print()
        
        # Benchmark 1: N-Queens (problema clásico)
        print("Benchmark 1: N-Queens Problem")
        print("-" * 80)
        for n in [4, 6, 8]:
            comparison = self._benchmark_nqueens(n)
            self.comparisons.append(comparison)
            self._print_comparison(comparison)
            print()
        
        # Benchmark 2: Graph Coloring
        print("\nBenchmark 2: Graph Coloring Problem")
        print("-" * 80)
        for n_nodes in [5, 10, 15]:
            comparison = self._benchmark_graph_coloring(n_nodes, n_colors=3)
            self.comparisons.append(comparison)
            self._print_comparison(comparison)
            print()
        
        # Benchmark 3: HacificationEngine stress test
        print("\nBenchmark 3: HacificationEngine Stress Test")
        print("-" * 80)
        for n_calls in [100, 500, 1000]:
            comparison = self._benchmark_hacification_stress(n_calls)
            self.comparisons.append(comparison)
            self._print_comparison(comparison)
            print()
        
        # Generar reporte final
        self._generate_final_report()
        
        return self.comparisons
    
    def _benchmark_nqueens(self, n: int) -> ComparisonReport:
        """
        Benchmark del problema N-Queens.
        
        Args:
            n: Tamaño del tablero (n x n)
        
        Returns:
            Reporte de comparación
        """
        print(f"  N-Queens (n={n})...")
        
        # Crear problema
        hierarchy, variables, domains = self._create_nqueens_problem(n)
        landscape = EnergyLandscapeOptimized(hierarchy)
        
        # Benchmark versión original
        print("    Ejecutando versión original...", end=" ", flush=True)
        arc_engine_orig = ArcEngine(use_tms=False, parallel=False)
        result_orig = self._run_solver_benchmark(
            solver_class=FibrationSearchSolver,
            hierarchy=hierarchy,
            landscape=landscape,
            arc_engine=arc_engine_orig,
            variables=variables,
            domains=domains,
            use_homotopy=False,
            use_tms=False,
            name=f"nqueens_{n}",
            version="original",
            problem_size=n
        )
        print(f"✓ ({result_orig.wall_time:.2f}s)")
        
        # Benchmark versión optimizada
        print("    Ejecutando versión optimizada...", end=" ", flush=True)
        arc_engine_opt = ArcEngine(use_tms=True, parallel=False)
        result_opt = self._run_solver_benchmark(
            solver_class=FibrationSearchSolverEnhanced,
            hierarchy=hierarchy,
            landscape=landscape,
            arc_engine=arc_engine_opt,
            variables=variables,
            domains=domains,
            use_homotopy=True,
            use_tms=True,
            name=f"nqueens_{n}",
            version="optimized",
            problem_size=n
        )
        print(f"✓ ({result_opt.wall_time:.2f}s)")
        
        # Crear reporte de comparación
        return self._create_comparison_report(f"N-Queens (n={n})", n, result_orig, result_opt)
    
    def _benchmark_graph_coloring(self, n_nodes: int, n_colors: int) -> ComparisonReport:
        """
        Benchmark del problema de coloreo de grafos.
        
        Args:
            n_nodes: Número de nodos
            n_colors: Número de colores disponibles
        
        Returns:
            Reporte de comparación
        """
        print(f"  Graph Coloring (nodes={n_nodes}, colors={n_colors})...")
        
        # Crear problema
        hierarchy, variables, domains = self._create_graph_coloring_problem(n_nodes, n_colors)
        landscape = EnergyLandscapeOptimized(hierarchy)
        
        # Benchmark versión original
        print("    Ejecutando versión original...", end=" ", flush=True)
        arc_engine_orig = ArcEngine(use_tms=False, parallel=False)
        result_orig = self._run_solver_benchmark(
            solver_class=FibrationSearchSolver,
            hierarchy=hierarchy,
            landscape=landscape,
            arc_engine=arc_engine_orig,
            variables=variables,
            domains=domains,
            use_homotopy=False,
            use_tms=False,
            name=f"graph_coloring_{n_nodes}",
            version="original",
            problem_size=n_nodes
        )
        print(f"✓ ({result_orig.wall_time:.2f}s)")
        
        # Benchmark versión optimizada
        print("    Ejecutando versión optimizada...", end=" ", flush=True)
        arc_engine_opt = ArcEngine(use_tms=True, parallel=False)
        result_opt = self._run_solver_benchmark(
            solver_class=FibrationSearchSolverEnhanced,
            hierarchy=hierarchy,
            landscape=landscape,
            arc_engine=arc_engine_opt,
            variables=variables,
            domains=domains,
            use_homotopy=True,
            use_tms=True,
            name=f"graph_coloring_{n_nodes}",
            version="optimized",
            problem_size=n_nodes
        )
        print(f"✓ ({result_opt.wall_time:.2f}s)")
        
        return self._create_comparison_report(
            f"Graph Coloring (nodes={n_nodes})", n_nodes, result_orig, result_opt
        )
    
    def _benchmark_hacification_stress(self, n_calls: int) -> ComparisonReport:
        """
        Stress test del HacificationEngine.
        
        Args:
            n_calls: Número de llamadas a hacify()
        
        Returns:
            Reporte de comparación
        """
        print(f"  HacificationEngine Stress (calls={n_calls})...")
        
        # Crear problema simple
        hierarchy, variables, domains = self._create_nqueens_problem(4)
        landscape = EnergyLandscapeOptimized(hierarchy)
        
        # Benchmark versión original
        print("    Ejecutando versión original...", end=" ", flush=True)
        arc_engine_orig = ArcEngine(use_tms=False, parallel=False)
        hacification_orig = HacificationEngine(hierarchy, landscape, arc_engine_orig)
        
        result_orig = self._run_hacification_stress_benchmark(
            hacification_orig, n_calls, "original", n_calls
        )
        print(f"✓ ({result_orig.wall_time:.2f}s)")
        
        # Benchmark versión optimizada
        print("    Ejecutando versión optimizada...", end=" ", flush=True)
        arc_engine_opt = ArcEngine(use_tms=False, parallel=False)
        hacification_opt = HacificationEngineOptimized(hierarchy, landscape, arc_engine_opt)
        
        result_opt = self._run_hacification_stress_benchmark(
            hacification_opt, n_calls, "optimized", n_calls
        )
        print(f"✓ ({result_opt.wall_time:.2f}s)")
        
        return self._create_comparison_report(
            f"HacificationEngine Stress (calls={n_calls})", n_calls, result_orig, result_opt
        )
    
    def _run_solver_benchmark(
        self,
        solver_class,
        hierarchy: ConstraintHierarchy,
        landscape: EnergyLandscapeOptimized,
        arc_engine: ArcEngine,
        variables: List[str],
        domains: Dict[str, List[Any]],
        use_homotopy: bool,
        use_tms: bool,
        name: str,
        version: str,
        problem_size: int
    ) -> BenchmarkResult:
        """Ejecuta un benchmark de solver."""
        # Limpiar memoria antes del benchmark
        gc.collect()
        
        # Iniciar medición de memoria
        tracemalloc.start()
        
        # Crear solver
        if solver_class == FibrationSearchSolver:
            solver = FibrationSearchSolver(
                variables=variables,
                domains=domains,
                hierarchy=hierarchy,
                use_homotopy=use_homotopy,
                use_tms=use_tms,
                max_backtracks=10000,
                max_iterations=10000
            )
        else:  # FibrationSearchSolverEnhanced
            solver = FibrationSearchSolverEnhanced(
                hierarchy=hierarchy,
                landscape=landscape,
                arc_engine=arc_engine,
                variables=variables,
                domains=domains,
                use_homotopy=use_homotopy,
                use_tms=use_tms,
                use_enhanced_heuristics=True,
                max_backtracks=10000,
                max_iterations=10000,
                time_limit_seconds=30.0
            )
        
        # Ejecutar solver
        start_time = time.time()
        solution = solver.solve(time_limit_seconds=30)
        wall_time = time.time() - start_time
        
        # Obtener estadísticas de memoria
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Obtener estadísticas del solver
        stats = solver.get_statistics() if hasattr(solver, 'get_statistics') else {}
        
        # Crear resultado
        return BenchmarkResult(
            name=name,
            version=version,
            problem_size=problem_size,
            wall_time=wall_time,
            peak_memory_mb=peak / (1024 * 1024),
            memory_allocations=current,
            nodes_explored=stats.get('search', {}).get('nodes_explored', solver.stats.get('nodes_explored', 0)) if hasattr(solver, 'stats') else 0,
            backtracks=solver.backtracks_count,
            backjumps=stats.get('search', {}).get('backjumps', 0),
            conflicts_analyzed=stats.get('search', {}).get('conflicts_analyzed', 0),
            solution_found=solution is not None,
            solution_energy=solver.best_energy if solver.best_energy != float('inf') else 0.0,
            num_solutions=solver.num_solutions_found,
            mrv_calls=stats.get('heuristics', {}).get('mrv_calls', 0),
            lcv_calls=stats.get('heuristics', {}).get('lcv_calls', 0)
        )
    
    def _run_hacification_stress_benchmark(
        self,
        hacification_engine,
        n_calls: int,
        version: str,
        problem_size: int
    ) -> BenchmarkResult:
        """Ejecuta un stress test del HacificationEngine."""
        # Limpiar memoria
        gc.collect()
        
        # Iniciar medición de memoria
        tracemalloc.start()
        
        # Generar asignaciones de prueba
        test_assignments = [
            {"Q0": 0, "Q1": 2, "Q2": 1, "Q3": 3},
            {"Q0": 1, "Q1": 3, "Q2": 0, "Q3": 2},
            {"Q0": 2, "Q1": 0, "Q2": 3, "Q1": 1},
        ]
        
        # Ejecutar múltiples llamadas
        start_time = time.time()
        for i in range(n_calls):
            assignment = test_assignments[i % len(test_assignments)]
            hacification_engine.hacify(assignment, strict=True)
        wall_time = time.time() - start_time
        
        # Obtener estadísticas de memoria
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Obtener estadísticas
        stats = hacification_engine.get_statistics()
        
        return BenchmarkResult(
            name=f"hacification_stress_{n_calls}",
            version=version,
            problem_size=problem_size,
            wall_time=wall_time,
            peak_memory_mb=peak / (1024 * 1024),
            memory_allocations=current,
            nodes_explored=0,
            backtracks=0,
            backjumps=0,
            conflicts_analyzed=0,
            solution_found=True,
            solution_energy=0.0,
            num_solutions=n_calls,
            metadata={'hacify_calls': stats.get('performance', {}).get('hacify_calls', 0)}
        )
    
    def _create_nqueens_problem(self, n: int) -> Tuple[ConstraintHierarchy, List[str], Dict[str, List[int]]]:
        """Crea un problema N-Queens."""
        hierarchy = ConstraintHierarchy()
        variables = [f"Q{i}" for i in range(n)]
        domains = {var: list(range(n)) for var in variables}
        
        # Añadir restricciones: no dos reinas en la misma fila, columna o diagonal
        for i in range(n):
            for j in range(i + 1, n):
                # No misma fila (implícito por el modelo)
                # No misma columna
                def ne_constraint(assignment, i=i, j=j):
                    return assignment[f"Q{i}"] != assignment[f"Q{j}"]
                
                hierarchy.add_local_constraint(
                    var1=f"Q{i}", var2=f"Q{j}",
                    predicate=ne_constraint,
                    hardness=Hardness.HARD,
                    metadata={"name": f"Q{i}_ne_Q{j}"}
                )
                
                # No misma diagonal
                def no_diagonal(assignment, i=i, j=j):
                    return abs(assignment[f"Q{i}"] - assignment[f"Q{j}"]) != abs(i - j)
                
                hierarchy.add_local_constraint(
                    var1=f"Q{i}", var2=f"Q{j}",
                    predicate=no_diagonal,
                    hardness=Hardness.HARD,
                    metadata={"name": f"Q{i}_nodiag_Q{j}"}
                )
        
        return hierarchy, variables, domains
    
    def _create_graph_coloring_problem(
        self, n_nodes: int, n_colors: int
    ) -> Tuple[ConstraintHierarchy, List[str], Dict[str, List[int]]]:
        """Crea un problema de coloreo de grafos."""
        hierarchy = ConstraintHierarchy()
        variables = [f"N{i}" for i in range(n_nodes)]
        domains = {var: list(range(n_colors)) for var in variables}
        
        # Crear grafo (ciclo simple para este benchmark)
        for i in range(n_nodes):
            j = (i + 1) % n_nodes
            
            def ne_constraint(assignment, i=i, j=j):
                return assignment[f"N{i}"] != assignment[f"N{j}"]
            
            hierarchy.add_local_constraint(
                var1=f"N{i}", var2=f"N{j}",
                predicate=ne_constraint,
                hardness=Hardness.HARD,
                metadata={"name": f"N{i}_ne_N{j}"}
            )
        
        return hierarchy, variables, domains
    
    def _create_comparison_report(
        self,
        name: str,
        problem_size: int,
        original: BenchmarkResult,
        optimized: BenchmarkResult
    ) -> ComparisonReport:
        """Crea un reporte de comparación."""
        time_speedup = original.wall_time / optimized.wall_time if optimized.wall_time > 0 else 1.0
        memory_reduction = original.peak_memory_mb / optimized.peak_memory_mb if optimized.peak_memory_mb > 0 else 1.0
        backtrack_reduction = original.backtracks / optimized.backtracks if optimized.backtracks > 0 else 1.0
        
        return ComparisonReport(
            benchmark_name=name,
            problem_size=problem_size,
            original=original,
            optimized=optimized,
            time_speedup=time_speedup,
            memory_reduction=memory_reduction,
            backtrack_reduction=backtrack_reduction
        )
    
    def _print_comparison(self, comparison: ComparisonReport):
        """Imprime un reporte de comparación."""
        print(f"    Resultados:")
        print(f"      Tiempo:    {comparison.original.wall_time:.3f}s → {comparison.optimized.wall_time:.3f}s  (speedup: {comparison.time_speedup:.2f}x)")
        print(f"      Memoria:   {comparison.original.peak_memory_mb:.2f}MB → {comparison.optimized.peak_memory_mb:.2f}MB  (reducción: {comparison.memory_reduction:.2f}x)")
        print(f"      Backtracks: {comparison.original.backtracks} → {comparison.optimized.backtracks}  (reducción: {comparison.backtrack_reduction:.2f}x)")
        if comparison.optimized.backjumps > 0:
            print(f"      Backjumps:  {comparison.optimized.backjumps}")
        print(f"      Energía:    {comparison.original.solution_energy:.4f} → {comparison.optimized.solution_energy:.4f}")
    
    def _generate_final_report(self):
        """Genera un reporte final consolidado."""
        print("\n" + "=" * 80)
        print("RESUMEN FINAL DE MEJORAS")
        print("=" * 80)
        
        if not self.comparisons:
            print("No hay comparaciones disponibles.")
            return
        
        # Calcular promedios
        avg_time_speedup = sum(c.time_speedup for c in self.comparisons) / len(self.comparisons)
        avg_memory_reduction = sum(c.memory_reduction for c in self.comparisons) / len(self.comparisons)
        avg_backtrack_reduction = sum(c.backtrack_reduction for c in self.comparisons) / len(self.comparisons)
        
        print(f"\nPromedios generales:")
        print(f"  Speedup de tiempo:      {avg_time_speedup:.2f}x")
        print(f"  Reducción de memoria:   {avg_memory_reduction:.2f}x")
        print(f"  Reducción de backtracks: {avg_backtrack_reduction:.2f}x")
        
        # Guardar resultados en JSON
        report_data = {
            'summary': {
                'avg_time_speedup': avg_time_speedup,
                'avg_memory_reduction': avg_memory_reduction,
                'avg_backtrack_reduction': avg_backtrack_reduction
            },
            'comparisons': [c.to_dict() for c in self.comparisons]
        }
        
        with open('/home/ubuntu/docs/fibration_flow_benchmark_results.json', 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n✓ Resultados guardados en: /home/ubuntu/docs/fibration_flow_benchmark_results.json")
        print("=" * 80)


def main():
    """Función principal para ejecutar los benchmarks."""
    suite = FibrationFlowBenchmarkSuite()
    suite.run_all_benchmarks()


if __name__ == "__main__":
    main()

