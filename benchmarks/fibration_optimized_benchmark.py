"""
Benchmark: Flujo de Fibración Optimizado vs. Baseline

Compara:
1. Backtracking Simple (baseline)
2. Forward Checking (estado del arte)
3. Flujo de Fibración Original (sin optimizar)
4. Flujo de Fibración Optimizado (con todas las optimizaciones)

Optimizaciones implementadas:
- Cálculo incremental de energía
- Propagación de restricciones
- Heurística MRV
- Poda agresiva
- Cache habilitado
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
    EnergyLandscape,
    EnergyLandscapeOptimized,
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


# ============================================================================
# SOLVERS BASELINE (copiados del benchmark anterior)
# ============================================================================

class SimpleBacktrackingSolver:
    """Solver baseline: Backtracking simple."""
    
    def __init__(self, variables: List[str], domains: Dict[str, List], 
                 constraints: List[Callable]):
        self.variables = variables
        self.domains = domains
        self.constraints = constraints
        self.nodes_explored = 0
        
    def solve(self) -> Optional[Dict]:
        self.nodes_explored = 0
        return self._backtrack({})
    
    def _backtrack(self, assignment: Dict) -> Optional[Dict]:
        self.nodes_explored += 1
        
        if len(assignment) == len(self.variables):
            return assignment
        
        var = self._select_unassigned_variable(assignment)
        
        for value in self.domains[var]:
            assignment[var] = value
            
            if self._is_consistent(assignment):
                result = self._backtrack(assignment)
                if result is not None:
                    return result
            
            del assignment[var]
        
        return None
    
    def _select_unassigned_variable(self, assignment: Dict) -> str:
        for var in self.variables:
            if var not in assignment:
                return var
    
    def _is_consistent(self, assignment: Dict) -> bool:
        for constraint in self.constraints:
            if not constraint(assignment):
                return False
        return True


class ForwardCheckingSolver:
    """Solver con forward checking."""
    
    def __init__(self, variables: List[str], domains: Dict[str, List], 
                 constraints: List[Callable]):
        self.variables = variables
        self.domains = domains
        self.constraints = constraints
        self.nodes_explored = 0
        
    def solve(self) -> Optional[Dict]:
        self.nodes_explored = 0
        return self._backtrack({}, {var: list(domain) for var, domain in self.domains.items()})
    
    def _backtrack(self, assignment: Dict, domains: Dict) -> Optional[Dict]:
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


# ============================================================================
# BENCHMARK RUNNER
# ============================================================================

def run_benchmark(solver_name: str, solver, problem_name: str, 
                 problem_size: int, timeout: float = 30.0) -> BenchmarkResult:
    """Ejecuta un benchmark con timeout."""
    print(f"  {solver_name:<30}", end=" ", flush=True)
    
    start_time = time.time()
    
    try:
        solution = solver.solve()
        elapsed = time.time() - start_time
        
        if elapsed > timeout:
            print(f"⏱️  TIMEOUT")
            return BenchmarkResult(
                solver_name=solver_name,
                problem_name=problem_name,
                problem_size=problem_size,
                time_seconds=timeout,
                nodes_explored=solver.nodes_explored,
                solution_found=False
            )
        
        nodes = solver.nodes_explored
        found = solution is not None
        
        if found:
            print(f"✓ {elapsed:6.3f}s  {nodes:8d} nodos")
        else:
            print(f"✗ {elapsed:6.3f}s  {nodes:8d} nodos")
        
        return BenchmarkResult(
            solver_name=solver_name,
            problem_name=problem_name,
            problem_size=problem_size,
            time_seconds=elapsed,
            nodes_explored=nodes,
            solution_found=found
        )
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ ERROR: {e}")
        return BenchmarkResult(
            solver_name=solver_name,
            problem_name=problem_name,
            problem_size=problem_size,
            time_seconds=elapsed,
            nodes_explored=0,
            solution_found=False
        )


def benchmark_nqueens(sizes: List[int]) -> List[BenchmarkResult]:
    """Benchmark N-Queens."""
    results = []
    
    for n in sizes:
        print(f"\n{'='*70}")
        print(f"N-Queens: n={n}")
        print(f"{'='*70}")
        
        variables, domains, constraints_list, hierarchy = create_nqueens_problem(n)
        
        # Backtracking simple
        solver = SimpleBacktrackingSolver(variables, domains, constraints_list)
        result = run_benchmark("Backtracking Simple", solver, "N-Queens", n)
        results.append(result)
        
        # Forward checking
        solver = ForwardCheckingSolver(variables, domains, constraints_list)
        result = run_benchmark("Forward Checking", solver, "N-Queens", n)
        results.append(result)
        
        # Flujo de Fibración Optimizado
        solver = CoherenceSolverOptimized(variables, domains)
        solver.hierarchy = hierarchy
        solver.landscape = EnergyLandscapeOptimized(hierarchy)
        result = run_benchmark("Flujo de Fibración Optimizado", solver, "N-Queens", n)
        results.append(result)
        
        # Mostrar estadísticas del solver optimizado
        if result.solution_found:
            stats = solver.get_statistics()
            print(f"\n  Estadísticas del Solver Optimizado:")
            print(f"    Nodos explorados:     {stats['nodes_explored']}")
            print(f"    Nodos podados:        {stats['nodes_pruned']}")
            print(f"    Propagaciones:        {stats['propagations']}")
            print(f"    Conflictos detectados: {stats['conflicts_detected']}")
            print(f"    Tasa de poda:         {stats['pruning_rate']:.2%}")
            
            landscape_stats = stats['landscape_stats']
            print(f"\n  Estadísticas del Paisaje de Energía:")
            print(f"    Hit rate del cache:   {landscape_stats['hit_rate']:.2%}")
            print(f"    Cálculos incrementales: {landscape_stats['incremental_rate']:.2%}")
    
    return results


def print_summary_table(results: List[BenchmarkResult]):
    """Imprime tabla resumen."""
    print("\n" + "="*80)
    print("RESUMEN COMPARATIVO")
    print("="*80)
    
    by_problem = {}
    for result in results:
        key = (result.problem_name, result.problem_size)
        if key not in by_problem:
            by_problem[key] = []
        by_problem[key].append(result)
    
    for (problem, size), group in sorted(by_problem.items()):
        print(f"\n{problem} (n={size}):")
        print(f"{'Solver':<35} {'Tiempo (s)':<12} {'Nodos':<12} {'Speedup vs BT':<15}")
        print("-" * 80)
        
        baseline_time = next((r.time_seconds for r in group if r.solver_name == "Backtracking Simple"), None)
        baseline_nodes = next((r.nodes_explored for r in group if r.solver_name == "Backtracking Simple"), None)
        
        for result in group:
            speedup_time = ""
            speedup_nodes = ""
            
            if baseline_time and result.time_seconds > 0 and result.solution_found:
                speedup_time = f"{baseline_time / result.time_seconds:.2f}x"
            
            if baseline_nodes and result.nodes_explored > 0 and result.solution_found:
                speedup_nodes = f"({baseline_nodes / result.nodes_explored:.2f}x nodos)"
            
            speedup_str = f"{speedup_time} {speedup_nodes}"
            
            print(f"{result.solver_name:<35} {result.time_seconds:<12.3f} {result.nodes_explored:<12} {speedup_str:<15}")


def main():
    """Función principal."""
    print("="*80)
    print("BENCHMARK: FLUJO DE FIBRACIÓN OPTIMIZADO")
    print("="*80)
    print("\nComparando:")
    print("  1. Backtracking Simple (baseline)")
    print("  2. Forward Checking (estado del arte)")
    print("  3. Flujo de Fibración Optimizado (nuestro método)")
    print("\nOptimizaciones implementadas:")
    print("  ✓ Cálculo incremental de energía")
    print("  ✓ Propagación de restricciones")
    print("  ✓ Heurística MRV (Minimum Remaining Values)")
    print("  ✓ Poda agresiva basada en energía")
    print("  ✓ Cache habilitado")
    
    all_results = []
    
    # Benchmark: N-Queens
    print("\n" + "="*80)
    print("PROBLEMA: N-QUEENS")
    print("="*80)
    results = benchmark_nqueens([4, 6, 8, 10])
    all_results.extend(results)
    
    # Resumen
    print_summary_table(all_results)
    
    # Análisis final
    print("\n" + "="*80)
    print("ANÁLISIS FINAL")
    print("="*80)
    
    fibration_results = [r for r in all_results if "Optimizado" in r.solver_name and r.solution_found]
    baseline_results = [r for r in all_results if r.solver_name == "Backtracking Simple" and r.solution_found]
    fc_results = [r for r in all_results if r.solver_name == "Forward Checking" and r.solution_found]
    
    if fibration_results and baseline_results:
        # Speedup vs Backtracking
        avg_speedup_bt = sum(
            next((b.time_seconds / f.time_seconds for b in baseline_results 
                  if b.problem_size == f.problem_size), 0)
            for f in fibration_results
        ) / len(fibration_results)
        
        avg_node_reduction_bt = sum(
            (1 - f.nodes_explored / next((b.nodes_explored for b in baseline_results 
                                         if b.problem_size == f.problem_size), 1))
            for f in fibration_results
        ) / len(fibration_results)
        
        print(f"\nVs. Backtracking Simple:")
        print(f"  Speedup promedio:              {avg_speedup_bt:.2f}x")
        print(f"  Reducción de nodos explorados: {avg_node_reduction_bt*100:.1f}%")
    
    if fibration_results and fc_results:
        # Speedup vs Forward Checking
        avg_speedup_fc = sum(
            next((fc.time_seconds / f.time_seconds for fc in fc_results 
                  if fc.problem_size == f.problem_size), 0)
            for f in fibration_results
        ) / len(fibration_results)
        
        avg_node_reduction_fc = sum(
            (1 - f.nodes_explored / next((fc.nodes_explored for fc in fc_results 
                                         if fc.problem_size == f.problem_size), 1))
            for f in fibration_results
        ) / len(fibration_results)
        
        print(f"\nVs. Forward Checking:")
        print(f"  Speedup promedio:              {avg_speedup_fc:.2f}x")
        print(f"  Reducción de nodos explorados: {avg_node_reduction_fc*100:.1f}%")


if __name__ == "__main__":
    main()

