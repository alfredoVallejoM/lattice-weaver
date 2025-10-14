"""
Benchmark de Escalabilidad: Problemas Grandes

Prueba el Flujo de Fibración Optimizado con problemas grandes:
- N-Queens: n=12, 15, 20, 25, 30
- Graph Coloring: grafos de 20-50 nodos
- Sudoku: 9x9, 16x16

Objetivo: Verificar que el solver escala correctamente y mantiene eficiencia.
"""

import time
import sys
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import random

sys.path.insert(0, '/home/ubuntu/lattice-weaver')

from lattice_weaver.fibration import (
    ConstraintHierarchy,
    ConstraintLevel,
    Hardness,
    CoherenceSolverOptimized
)


@dataclass
class BenchmarkResult:
    """Resultado de un benchmark."""
    solver_name: str
    problem_name: str
    problem_size: int
    time_seconds: float
    nodes_explored: int
    solution_found: bool
    timeout: bool = False


# ============================================================================
# FORWARD CHECKING (para comparación)
# ============================================================================

class ForwardCheckingSolver:
    """Solver con forward checking para comparación."""
    
    def __init__(self, variables: List[str], domains: Dict[str, List], 
                 constraints: List[Callable]):
        self.variables = variables
        self.domains = domains
        self.constraints = constraints
        self.nodes_explored = 0
        
    def solve(self, timeout: float = 60.0) -> Optional[Dict]:
        self.nodes_explored = 0
        self.start_time = time.time()
        self.timeout = timeout
        return self._backtrack({}, {var: list(domain) for var, domain in self.domains.items()})
    
    def _backtrack(self, assignment: Dict, domains: Dict) -> Optional[Dict]:
        if time.time() - self.start_time > self.timeout:
            return None
        
        self.nodes_explored += 1
        
        if len(assignment) == len(self.variables):
            return assignment
        
        var = self._select_unassigned_variable(assignment, domains)
        
        for value in domains[var]:
            assignment[var] = value
            
            if self._is_consistent(assignment):
                new_domains = self._forward_check(assignment, domains, var, value)
                
                if new_domains is not None:
                    result = self._backtrack(assignment, new_domains)
                    if result is not None:
                        return result
            
            del assignment[var]
        
        return None
    
    def _select_unassigned_variable(self, assignment: Dict, domains: Dict) -> str:
        unassigned = [v for v in self.variables if v not in assignment]
        return min(unassigned, key=lambda v: len(domains[v]))
    
    def _is_consistent(self, assignment: Dict) -> bool:
        for constraint in self.constraints:
            if not constraint(assignment):
                return False
        return True
    
    def _forward_check(self, assignment: Dict, domains: Dict, 
                      var: str, value) -> Optional[Dict]:
        new_domains = {v: list(d) for v, d in domains.items()}
        
        for other_var in self.variables:
            if other_var in assignment or other_var == var:
                continue
            
            new_domains[other_var] = [
                v for v in new_domains[other_var]
                if self._is_consistent({**assignment, other_var: v})
            ]
            
            if not new_domains[other_var]:
                return None
        
        return new_domains


# ============================================================================
# PROBLEMAS DE PRUEBA
# ============================================================================

def create_nqueens_problem(n: int) -> Tuple[List[str], Dict, List[Callable], ConstraintHierarchy]:
    """Crea un problema de N-Queens."""
    variables = [f"Q{i}" for i in range(n)]
    domains = {var: list(range(n)) for var in variables}
    
    constraints_list = []
    hierarchy = ConstraintHierarchy()
    
    for i in range(n):
        for j in range(i+1, n):
            # No misma fila
            def no_same_row(a, i=i, j=j):
                if f"Q{i}" not in a or f"Q{j}" not in a:
                    return True
                return a[f"Q{i}"] != a[f"Q{j}"]
            
            constraints_list.append(no_same_row)
            
            hierarchy.add_local_constraint(
                f"Q{i}", f"Q{j}",
                lambda a, i=i, j=j: a[f"Q{i}"] != a[f"Q{j}"],
                weight=1.0,
                hardness=Hardness.HARD
            )
            
            # No misma diagonal
            def no_same_diagonal(a, i=i, j=j):
                if f"Q{i}" not in a or f"Q{j}" not in a:
                    return True
                qi = a[f"Q{i}"]
                qj = a[f"Q{j}"]
                return abs(qi - qj) != abs(i - j)
            
            constraints_list.append(no_same_diagonal)
            
            hierarchy.add_local_constraint(
                f"Q{i}", f"Q{j}",
                lambda a, i=i, j=j: abs(a[f"Q{i}"] - a[f"Q{j}"]) != abs(i - j),
                weight=1.0,
                hardness=Hardness.HARD
            )
    
    return variables, domains, constraints_list, hierarchy


def create_sudoku_problem(size: int = 9) -> Tuple[List[str], Dict, List[Callable], ConstraintHierarchy]:
    """
    Crea un problema de Sudoku.
    
    Args:
        size: Tamaño del Sudoku (9 o 16)
    """
    variables = [f"C{i}_{j}" for i in range(size) for j in range(size)]
    domains = {var: list(range(1, size+1)) for var in variables}
    
    constraints_list = []
    hierarchy = ConstraintHierarchy()
    
    # Restricciones de filas
    for i in range(size):
        row_vars = [f"C{i}_{j}" for j in range(size)]
        
        def row_constraint(a, row_vars=row_vars):
            assigned = [a[v] for v in row_vars if v in a]
            return len(assigned) == len(set(assigned))
        
        constraints_list.append(row_constraint)
        
        # Añadir restricciones binarias para la jerarquía
        for j1 in range(size):
            for j2 in range(j1+1, size):
                v1, v2 = f"C{i}_{j1}", f"C{i}_{j2}"
                hierarchy.add_local_constraint(
                    v1, v2,
                    lambda a, v1=v1, v2=v2: a[v1] != a[v2],
                    weight=1.0,
                    hardness=Hardness.HARD
                )
    
    # Restricciones de columnas
    for j in range(size):
        col_vars = [f"C{i}_{j}" for i in range(size)]
        
        def col_constraint(a, col_vars=col_vars):
            assigned = [a[v] for v in col_vars if v in a]
            return len(assigned) == len(set(assigned))
        
        constraints_list.append(col_constraint)
        
        # Añadir restricciones binarias
        for i1 in range(size):
            for i2 in range(i1+1, size):
                v1, v2 = f"C{i1}_{j}", f"C{i2}_{j}"
                hierarchy.add_local_constraint(
                    v1, v2,
                    lambda a, v1=v1, v2=v2: a[v1] != a[v2],
                    weight=1.0,
                    hardness=Hardness.HARD
                )
    
    # Restricciones de bloques (solo para 9x9)
    if size == 9:
        block_size = 3
        for block_i in range(3):
            for block_j in range(3):
                block_vars = [
                    f"C{i}_{j}" 
                    for i in range(block_i*3, (block_i+1)*3)
                    for j in range(block_j*3, (block_j+1)*3)
                ]
                
                def block_constraint(a, block_vars=block_vars):
                    assigned = [a[v] for v in block_vars if v in a]
                    return len(assigned) == len(set(assigned))
                
                constraints_list.append(block_constraint)
    
    return variables, domains, constraints_list, hierarchy


def create_graph_coloring_problem(n_nodes: int, n_colors: int, 
                                  edge_probability: float = 0.3,
                                  seed: int = 42) -> Tuple[List[str], Dict, List[Callable], ConstraintHierarchy]:
    """Crea un problema de coloración de grafos aleatorio."""
    random.seed(seed)
    
    variables = [f"N{i}" for i in range(n_nodes)]
    domains = {var: list(range(n_colors)) for var in variables}
    
    # Generar grafo aleatorio
    edges = []
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            if random.random() < edge_probability:
                edges.append((i, j))
    
    constraints_list = []
    hierarchy = ConstraintHierarchy()
    
    for i, j in edges:
        def edge_constraint(a, i=i, j=j):
            if f"N{i}" not in a or f"N{j}" not in a:
                return True
            return a[f"N{i}"] != a[f"N{j}"]
        
        constraints_list.append(edge_constraint)
        
        hierarchy.add_local_constraint(
            f"N{i}", f"N{j}",
            lambda a, i=i, j=j: a[f"N{i}"] != a[f"N{j}"],
            weight=1.0,
            hardness=Hardness.HARD
        )
    
    return variables, domains, constraints_list, hierarchy


# ============================================================================
# BENCHMARK RUNNER
# ============================================================================

def run_benchmark(solver_name: str, solver, problem_name: str, 
                 problem_size: int, timeout: float = 120.0) -> BenchmarkResult:
    """Ejecuta un benchmark con timeout."""
    print(f"  {solver_name:<30}", end=" ", flush=True)
    
    start_time = time.time()
    
    try:
        if hasattr(solver, 'solve'):
            if 'timeout' in solver.solve.__code__.co_varnames:
                solution = solver.solve(timeout=timeout)
            else:
                solution = solver.solve(max_nodes=1000000)
        else:
            solution = None
        
        elapsed = time.time() - start_time
        
        if elapsed > timeout:
            print(f"⏱️  TIMEOUT ({timeout:.0f}s)")
            return BenchmarkResult(
                solver_name=solver_name,
                problem_name=problem_name,
                problem_size=problem_size,
                time_seconds=timeout,
                nodes_explored=solver.nodes_explored if hasattr(solver, 'nodes_explored') else 0,
                solution_found=False,
                timeout=True
            )
        
        nodes = solver.nodes_explored if hasattr(solver, 'nodes_explored') else 0
        found = solution is not None
        
        if found:
            print(f"✓ {elapsed:7.2f}s  {nodes:10d} nodos")
        else:
            print(f"✗ {elapsed:7.2f}s  {nodes:10d} nodos")
        
        return BenchmarkResult(
            solver_name=solver_name,
            problem_name=problem_name,
            problem_size=problem_size,
            time_seconds=elapsed,
            nodes_explored=nodes,
            solution_found=found,
            timeout=False
        )
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ ERROR: {str(e)[:50]}")
        return BenchmarkResult(
            solver_name=solver_name,
            problem_name=problem_name,
            problem_size=problem_size,
            time_seconds=elapsed,
            nodes_explored=0,
            solution_found=False,
            timeout=False
        )


def benchmark_nqueens_scalability() -> List[BenchmarkResult]:
    """Benchmark de escalabilidad con N-Queens."""
    results = []
    sizes = [12, 15, 20, 25]
    
    for n in sizes:
        print(f"\n{'='*70}")
        print(f"N-Queens: n={n} ({n*(n-1)} restricciones)")
        print(f"{'='*70}")
        
        variables, domains, constraints_list, hierarchy = create_nqueens_problem(n)
        
        # Forward Checking
        solver = ForwardCheckingSolver(variables, domains, constraints_list)
        result = run_benchmark("Forward Checking", solver, "N-Queens", n, timeout=120.0)
        results.append(result)
        
        # Flujo de Fibración Optimizado
        solver = CoherenceSolverOptimized(variables, domains)
        solver.hierarchy = hierarchy
        from lattice_weaver.fibration import EnergyLandscapeOptimized
        solver.landscape = EnergyLandscapeOptimized(hierarchy)
        result = run_benchmark("Flujo de Fibración Optimizado", solver, "N-Queens", n, timeout=120.0)
        results.append(result)
        
        # Mostrar estadísticas si encontró solución
        if result.solution_found:
            stats = solver.get_statistics()
            print(f"\n  📊 Estadísticas:")
            print(f"    Nodos:           {stats['nodes_explored']:,}")
            print(f"    Propagaciones:   {stats['propagations']:,}")
            print(f"    Cache hit rate:  {stats['landscape_stats']['hit_rate']:.1%}")
            print(f"    Incremental:     {stats['landscape_stats']['incremental_rate']:.1%}")
    
    return results


def benchmark_graph_coloring_scalability() -> List[BenchmarkResult]:
    """Benchmark de escalabilidad con Graph Coloring."""
    results = []
    sizes = [20, 30, 40, 50]
    
    for n in sizes:
        print(f"\n{'='*70}")
        print(f"Graph Coloring: {n} nodos, 3 colores, p=0.3")
        print(f"{'='*70}")
        
        variables, domains, constraints_list, hierarchy = create_graph_coloring_problem(
            n_nodes=n, n_colors=3, edge_probability=0.3
        )
        
        # Forward Checking
        solver = ForwardCheckingSolver(variables, domains, constraints_list)
        result = run_benchmark("Forward Checking", solver, "Graph Coloring", n, timeout=60.0)
        results.append(result)
        
        # Flujo de Fibración Optimizado
        solver = CoherenceSolverOptimized(variables, domains)
        solver.hierarchy = hierarchy
        from lattice_weaver.fibration import EnergyLandscapeOptimized
        solver.landscape = EnergyLandscapeOptimized(hierarchy)
        result = run_benchmark("Flujo de Fibración Optimizado", solver, "Graph Coloring", n, timeout=60.0)
        results.append(result)
        
        if result.solution_found:
            stats = solver.get_statistics()
            print(f"\n  📊 Estadísticas:")
            print(f"    Nodos:           {stats['nodes_explored']:,}")
            print(f"    Cache hit rate:  {stats['landscape_stats']['hit_rate']:.1%}")
    
    return results


def print_summary_table(results: List[BenchmarkResult]):
    """Imprime tabla resumen."""
    print("\n" + "="*90)
    print("RESUMEN DE ESCALABILIDAD")
    print("="*90)
    
    by_problem = {}
    for result in results:
        key = result.problem_name
        if key not in by_problem:
            by_problem[key] = []
        by_problem[key].append(result)
    
    for problem, group in sorted(by_problem.items()):
        print(f"\n{problem}:")
        print(f"{'Tamaño':<8} {'Solver':<30} {'Tiempo (s)':<12} {'Nodos':<15} {'Estado':<10}")
        print("-" * 90)
        
        by_size = {}
        for r in group:
            if r.problem_size not in by_size:
                by_size[r.problem_size] = []
            by_size[r.problem_size].append(r)
        
        for size in sorted(by_size.keys()):
            for i, result in enumerate(by_size[size]):
                size_str = f"n={size}" if i == 0 else ""
                status = "✓" if result.solution_found else ("TIMEOUT" if result.timeout else "✗")
                
                print(f"{size_str:<8} {result.solver_name:<30} {result.time_seconds:<12.2f} "
                      f"{result.nodes_explored:<15,} {status:<10}")


def main():
    """Función principal."""
    print("="*90)
    print("BENCHMARK DE ESCALABILIDAD: PROBLEMAS GRANDES")
    print("="*90)
    print("\nObjetivo: Verificar que el Flujo de Fibración Optimizado escala correctamente")
    print("Timeout: 120s para N-Queens, 60s para Graph Coloring")
    
    all_results = []
    
    # Benchmark 1: N-Queens (escalabilidad)
    print("\n" + "="*90)
    print("BENCHMARK 1: N-QUEENS (ESCALABILIDAD)")
    print("="*90)
    results = benchmark_nqueens_scalability()
    all_results.extend(results)
    
    # Benchmark 2: Graph Coloring (escalabilidad)
    print("\n" + "="*90)
    print("BENCHMARK 2: GRAPH COLORING (ESCALABILIDAD)")
    print("="*90)
    results = benchmark_graph_coloring_scalability()
    all_results.extend(results)
    
    # Resumen
    print_summary_table(all_results)
    
    # Análisis de escalabilidad
    print("\n" + "="*90)
    print("ANÁLISIS DE ESCALABILIDAD")
    print("="*90)
    
    fibration_results = [r for r in all_results if "Optimizado" in r.solver_name and r.solution_found]
    fc_results = [r for r in all_results if r.solver_name == "Forward Checking" and r.solution_found]
    
    if fibration_results:
        print(f"\nFlujo de Fibración Optimizado:")
        print(f"  Problemas resueltos: {len(fibration_results)}/{len([r for r in all_results if 'Optimizado' in r.solver_name])}")
        
        if len(fibration_results) >= 2:
            # Calcular factor de crecimiento
            sorted_fib = sorted(fibration_results, key=lambda r: r.problem_size)
            growth_factors = []
            for i in range(1, len(sorted_fib)):
                if sorted_fib[i].problem_name == sorted_fib[i-1].problem_name:
                    size_ratio = sorted_fib[i].problem_size / sorted_fib[i-1].problem_size
                    time_ratio = sorted_fib[i].time_seconds / sorted_fib[i-1].time_seconds
                    node_ratio = sorted_fib[i].nodes_explored / sorted_fib[i-1].nodes_explored if sorted_fib[i-1].nodes_explored > 0 else 0
                    growth_factors.append((size_ratio, time_ratio, node_ratio))
            
            if growth_factors:
                avg_time_growth = sum(t for _, t, _ in growth_factors) / len(growth_factors)
                avg_node_growth = sum(n for _, _, n in growth_factors if n > 0) / len([n for _, _, n in growth_factors if n > 0])
                print(f"  Factor de crecimiento promedio (tiempo): {avg_time_growth:.2f}x")
                print(f"  Factor de crecimiento promedio (nodos): {avg_node_growth:.2f}x")


if __name__ == "__main__":
    main()

