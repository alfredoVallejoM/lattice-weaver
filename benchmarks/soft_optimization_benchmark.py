"""
Benchmark: Optimización con Restricciones SOFT

Demuestra las ventajas del Flujo de Fibración en optimización multi-objetivo.

Comparación:
- Forward Checking: Primera solución factible (ignora SOFT)
- Flujo de Fibración (Optimization): Mejor solución (optimiza SOFT)
"""

import time
import sys
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass

sys.path.insert(0, '/home/ubuntu/lattice-weaver')

from lattice_weaver.fibration import (
    ConstraintHierarchy,
    ConstraintLevel,
    Hardness,
    OptimizationSolver,
    EnergyLandscapeOptimized
)


@dataclass
class SolutionQuality:
    """Métricas de calidad de una solución."""
    hard_violations: int
    soft_violations: float
    total_energy: float
    local_energy: float
    pattern_energy: float
    global_energy: float
    
    def is_valid(self) -> bool:
        return self.hard_violations == 0


@dataclass
class BenchmarkResult:
    """Resultado de un benchmark."""
    solver_name: str
    problem_name: str
    problem_size: int
    time_seconds: float
    nodes_explored: int
    solutions_found: int
    solution_found: bool
    solution_quality: Optional[SolutionQuality] = None


# ============================================================================
# FORWARD CHECKING (baseline)
# ============================================================================

class ForwardCheckingSolver:
    """Forward Checking: primera solución factible."""
    
    def __init__(self, variables: List[str], domains: Dict[str, List], 
                 hard_constraints: List[Callable]):
        self.variables = variables
        self.domains = domains
        self.hard_constraints = hard_constraints
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
        for constraint in self.hard_constraints:
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
# PROBLEMAS
# ============================================================================

def create_nqueens_with_preferences(n: int) -> Tuple[List[str], Dict, List[Callable], ConstraintHierarchy]:
    """N-Queens con preferencias de posición."""
    variables = [f"Q{i}" for i in range(n)]
    domains = {var: list(range(n)) for var in variables}
    
    hard_constraints = []
    hierarchy = ConstraintHierarchy()
    
    # HARD: No atacarse
    for i in range(n):
        for j in range(i+1, n):
            def no_same_row(a, i=i, j=j):
                if f"Q{i}" not in a or f"Q{j}" not in a:
                    return True
                return a[f"Q{i}"] != a[f"Q{j}"]
            
            hard_constraints.append(no_same_row)
            
            hierarchy.add_local_constraint(
                f"Q{i}", f"Q{j}",
                lambda a, i=i, j=j: a[f"Q{i}"] != a[f"Q{j}"],
                weight=1.0,
                hardness=Hardness.HARD
            )
            
            def no_same_diagonal(a, i=i, j=j):
                if f"Q{i}" not in a or f"Q{j}" not in a:
                    return True
                qi = a[f"Q{i}"]
                qj = a[f"Q{j}"]
                return abs(qi - qj) != abs(i - j)
            
            hard_constraints.append(no_same_diagonal)
            
            hierarchy.add_local_constraint(
                f"Q{i}", f"Q{j}",
                lambda a, i=i, j=j: abs(a[f"Q{i}"] - a[f"Q{j}"]) != abs(i - j),
                weight=1.0,
                hardness=Hardness.HARD
            )
    
    # SOFT: Preferir posiciones centrales
    center = n / 2.0
    
    def prefer_center(a):
        if len(a) < n:
            return 0.0
        
        total_distance = sum(abs(a[f"Q{i}"] - center) for i in range(n))
        max_distance = n * (n / 2.0)
        return total_distance / max_distance
    
    hierarchy.add_global_constraint(
        variables,
        prefer_center,
        objective="minimize",
        weight=2.0,  # Peso alto
        hardness=Hardness.SOFT
    )
    
    # SOFT: Distribución balanceada
    def prefer_balanced(a):
        if len(a) < n:
            return 0.0
        
        positions = [a[f"Q{i}"] for i in range(n)]
        mean_pos = sum(positions) / len(positions)
        variance = sum((p - mean_pos) ** 2 for p in positions) / len(positions)
        max_variance = (n ** 2) / 4
        return variance / max_variance
    
    hierarchy.add_pattern_constraint(
        variables,
        prefer_balanced,
        pattern_type="balanced",
        weight=1.0,
        hardness=Hardness.SOFT
    )
    
    return variables, domains, hard_constraints, hierarchy


# ============================================================================
# EVALUACIÓN
# ============================================================================

def evaluate_solution_quality(solution: Dict, hierarchy: ConstraintHierarchy,
                              landscape: EnergyLandscapeOptimized) -> SolutionQuality:
    """Evalúa la calidad de una solución."""
    energy = landscape.compute_energy(solution)
    
    hard_violations = 0
    for level in ConstraintLevel:
        constraints = hierarchy.get_constraints_at_level(level)
        for constraint in constraints:
            if constraint.hardness == Hardness.HARD:
                satisfied, violation = constraint.evaluate(solution)
                if not satisfied or violation > 0:
                    hard_violations += 1
    
    soft_violations = 0.0
    for level in ConstraintLevel:
        constraints = hierarchy.get_constraints_at_level(level)
        for constraint in constraints:
            if constraint.hardness == Hardness.SOFT:
                satisfied, violation = constraint.evaluate(solution)
                soft_violations += constraint.weight * violation
    
    return SolutionQuality(
        hard_violations=hard_violations,
        soft_violations=soft_violations,
        total_energy=energy.total_energy,
        local_energy=energy.local_energy,
        pattern_energy=energy.pattern_energy,
        global_energy=energy.global_energy
    )


# ============================================================================
# BENCHMARK
# ============================================================================

def run_benchmark(solver_name: str, solver, problem_name: str,
                 problem_size: int, hierarchy: ConstraintHierarchy,
                 landscape: EnergyLandscapeOptimized,
                 timeout: float = 60.0) -> BenchmarkResult:
    """Ejecuta benchmark."""
    print(f"  {solver_name:<40}", end=" ", flush=True)
    
    start_time = time.time()
    
    try:
        if hasattr(solver, 'solve'):
            if 'timeout' in solver.solve.__code__.co_varnames:
                solution = solver.solve(timeout=timeout)
            elif 'max_nodes' in solver.solve.__code__.co_varnames:
                solution = solver.solve(max_nodes=10000)
            else:
                solution = solver.solve()
        else:
            solution = None
        
        elapsed = time.time() - start_time
        nodes = solver.nodes_explored if hasattr(solver, 'nodes_explored') else 0
        solutions = solver.solutions_found if hasattr(solver, 'solutions_found') else (1 if solution else 0)
        found = solution is not None
        
        quality = None
        if found and solution:
            quality = evaluate_solution_quality(solution, hierarchy, landscape)
            print(f"✓ {elapsed:6.2f}s  {nodes:8d} nodos  {solutions:4d} sols  E={quality.total_energy:.3f} (SOFT={quality.soft_violations:.3f})")
        else:
            print(f"✗ {elapsed:6.2f}s  {nodes:8d} nodos")
        
        return BenchmarkResult(
            solver_name=solver_name,
            problem_name=problem_name,
            problem_size=problem_size,
            time_seconds=elapsed,
            nodes_explored=nodes,
            solutions_found=solutions,
            solution_found=found,
            solution_quality=quality
        )
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ ERROR: {str(e)[:50]}")
        import traceback
        traceback.print_exc()
        return BenchmarkResult(
            solver_name=solver_name,
            problem_name=problem_name,
            problem_size=problem_size,
            time_seconds=elapsed,
            nodes_explored=0,
            solutions_found=0,
            solution_found=False
        )


def main():
    """Función principal."""
    print("="*100)
    print("BENCHMARK: OPTIMIZACIÓN CON RESTRICCIONES SOFT")
    print("="*100)
    print("\nObjetivo: Demostrar que el Flujo de Fibración encuentra soluciones ÓPTIMAS")
    print("          mientras que Forward Checking solo encuentra soluciones FACTIBLES.")
    print("\nForward Checking: Primera solución factible (ignora SOFT)")
    print("Flujo de Fibración: Explora múltiples soluciones y devuelve la mejor (optimiza SOFT)")
    
    results = []
    sizes = [8, 10]
    
    for n in sizes:
        print(f"\n{'='*100}")
        print(f"N-Queens con Preferencias: n={n}")
        print(f"  HARD: No atacarse")
        print(f"  SOFT: Preferir centro (peso=2.0) + distribución balanceada (peso=1.0)")
        print(f"{'='*100}")
        
        variables, domains, hard_constraints, hierarchy = create_nqueens_with_preferences(n)
        landscape = EnergyLandscapeOptimized(hierarchy)
        
        # Forward Checking (primera solución)
        solver = ForwardCheckingSolver(variables, domains, hard_constraints)
        result = run_benchmark(
            "Forward Checking (primera solución)", solver, "N-Queens+Prefs", n,
            hierarchy, landscape, timeout=30.0
        )
        results.append(result)
        
        # Flujo de Fibración - Beam Search
        solver = OptimizationSolver(variables, domains, hierarchy)
        solver.beam_width = 10
        solver.k_best_values = 5
        result = run_benchmark(
            "Fibración Beam Search (k=3, beam=5)", solver, "N-Queens+Prefs", n,
            hierarchy, landscape, timeout=30.0
        )
        results.append(result)
        
        # Flujo de Fibración - K-Best
        solver = OptimizationSolver(variables, domains, hierarchy)
        solver.k_best_values = 8
        result = run_benchmark(
            "Fibración K-Best (k=5)", solver, "N-Queens+Prefs", n,
            hierarchy, landscape, timeout=30.0
        )
        results.append(result)
    
    # Comparación de calidad
    print("\n" + "="*100)
    print("COMPARACIÓN DE CALIDAD DE SOLUCIONES")
    print("="*100)
    
    by_size = {}
    for result in results:
        if result.problem_size not in by_size:
            by_size[result.problem_size] = []
        by_size[result.problem_size].append(result)
    
    for size in sorted(by_size.keys()):
        print(f"\nN-Queens (n={size}):")
        print(f"{'Solver':<45} {'Energía Total':<15} {'SOFT Violations':<18} {'Nodos':<10} {'Soluciones':<12}")
        print("-" * 100)
        
        fc_result = next((r for r in by_size[size] if "Forward Checking" in r.solver_name), None)
        fc_soft = fc_result.solution_quality.soft_violations if fc_result and fc_result.solution_quality else float('inf')
        
        for result in by_size[size]:
            if not result.solution_found or not result.solution_quality:
                continue
            
            q = result.solution_quality
            improvement = ""
            if fc_soft < float('inf') and q.soft_violations < fc_soft:
                improvement = f"({((fc_soft - q.soft_violations) / fc_soft * 100):.1f}% mejor)"
            
            print(f"{result.solver_name:<45} {q.total_energy:<15.3f} {q.soft_violations:<18.3f} {result.nodes_explored:<10} {result.solutions_found:<12} {improvement}")
    
    # Estadísticas finales
    print("\n" + "="*100)
    print("ESTADÍSTICAS FINALES")
    print("="*100)
    
    fc_results = [r for r in results if "Forward Checking" in r.solver_name and r.solution_found]
    fib_results = [r for r in results if "Fibración" in r.solver_name and r.solution_found]
    
    if fc_results and fib_results:
        fc_avg_soft = sum(r.solution_quality.soft_violations for r in fc_results) / len(fc_results)
        fib_avg_soft = sum(r.solution_quality.soft_violations for r in fib_results) / len(fib_results)
        
        improvement = ((fc_avg_soft - fib_avg_soft) / fc_avg_soft * 100) if fc_avg_soft > 0 else 0
        
        print(f"\nPromedio de violaciones SOFT:")
        print(f"  Forward Checking:      {fc_avg_soft:.3f}")
        print(f"  Flujo de Fibración:    {fib_avg_soft:.3f}")
        print(f"  Mejora:                {improvement:.1f}%")
        
        if improvement > 0:
            print(f"\n✨ El Flujo de Fibración encuentra soluciones {improvement:.1f}% mejores en promedio")
            print(f"   explorando múltiples alternativas y optimizando restricciones SOFT.")
        else:
            print(f"\n⚠️  Ambos solvers encuentran soluciones de calidad similar.")
            print(f"   Esto puede indicar que el problema tiene pocas soluciones o que las restricciones SOFT")
            print(f"   no discriminan suficientemente entre soluciones.")


if __name__ == "__main__":
    main()

