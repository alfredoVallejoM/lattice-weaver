"""
Benchmark: Restricciones SOFT - Ventajas del Flujo de Fibración

Este benchmark demuestra las ventajas únicas del Flujo de Fibración en problemas
con restricciones SOFT (optimización multi-objetivo).

Problemas:
1. N-Queens con preferencias de posición
2. Graph Coloring con preferencias de color
3. Scheduling con múltiples objetivos

Comparación:
- Forward Checking: Solo encuentra solución factible (ignora SOFT)
- Flujo de Fibración: Encuentra solución óptima (optimiza SOFT)
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
    CoherenceSolverOptimized,
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
        """Una solución es válida si no viola restricciones HARD."""
        return self.hard_violations == 0


@dataclass
class BenchmarkResult:
    """Resultado de un benchmark con métricas de calidad."""
    solver_name: str
    problem_name: str
    problem_size: int
    time_seconds: float
    nodes_explored: int
    solution_found: bool
    solution_quality: Optional[SolutionQuality] = None


# ============================================================================
# FORWARD CHECKING (baseline - solo encuentra solución factible)
# ============================================================================

class ForwardCheckingSolver:
    """Forward Checking: encuentra primera solución factible, ignora SOFT."""
    
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
# PROBLEMAS CON RESTRICCIONES SOFT
# ============================================================================

def create_nqueens_with_preferences(n: int) -> Tuple[List[str], Dict, List[Callable], ConstraintHierarchy]:
    """
    N-Queens con preferencias de posición.
    
    HARD: No atacarse (restricciones clásicas)
    SOFT: Preferir ciertas posiciones (centro del tablero, simetría)
    """
    variables = [f"Q{i}" for i in range(n)]
    domains = {var: list(range(n)) for var in variables}
    
    hard_constraints = []
    hierarchy = ConstraintHierarchy()
    
    # RESTRICCIONES HARD: No atacarse
    for i in range(n):
        for j in range(i+1, n):
            # No misma fila
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
            
            # No misma diagonal
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
    
    # RESTRICCIÓN SOFT 1: Preferir posiciones centrales
    center = n / 2.0
    
    def prefer_center(a):
        """Penaliza reinas alejadas del centro."""
        if len(a) < n:
            return 0.0
        
        total_distance = sum(abs(a[f"Q{i}"] - center) for i in range(n))
        max_distance = n * (n / 2.0)  # Máxima distancia posible
        return total_distance / max_distance  # Normalizar a [0, 1]
    
    hierarchy.add_global_constraint(
        variables,
        prefer_center,
        objective="minimize",
        weight=1.0,
        hardness=Hardness.SOFT
    )
    
    # RESTRICCIÓN SOFT 2: Preferir distribución balanceada
    def prefer_balanced_distribution(a):
        """Penaliza distribución desbalanceada de reinas."""
        if len(a) < n:
            return 0.0
        
        positions = [a[f"Q{i}"] for i in range(n)]
        mean_pos = sum(positions) / len(positions)
        variance = sum((p - mean_pos) ** 2 for p in positions) / len(positions)
        max_variance = (n ** 2) / 4  # Varianza máxima
        return variance / max_variance  # Normalizar
    
    hierarchy.add_pattern_constraint(
        variables,
        prefer_balanced_distribution,
        pattern_type="balanced",
        weight=0.5,
        hardness=Hardness.SOFT
    )
    
    return variables, domains, hard_constraints, hierarchy


def create_graph_coloring_with_preferences(n_nodes: int, n_colors: int,
                                          edge_probability: float = 0.25,
                                          seed: int = 42) -> Tuple[List[str], Dict, List[Callable], ConstraintHierarchy]:
    """
    Graph Coloring con preferencias de color.
    
    HARD: Nodos adyacentes con colores diferentes
    SOFT: Minimizar número de colores usados, preferir ciertos colores
    """
    random.seed(seed)
    
    variables = [f"N{i}" for i in range(n_nodes)]
    domains = {var: list(range(n_colors)) for var in variables}
    
    # Generar grafo aleatorio
    edges = []
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            if random.random() < edge_probability:
                edges.append((i, j))
    
    hard_constraints = []
    hierarchy = ConstraintHierarchy()
    
    # RESTRICCIONES HARD: Nodos adyacentes diferentes
    for i, j in edges:
        def edge_constraint(a, i=i, j=j):
            if f"N{i}" not in a or f"N{j}" not in a:
                return True
            return a[f"N{i}"] != a[f"N{j}"]
        
        hard_constraints.append(edge_constraint)
        
        hierarchy.add_local_constraint(
            f"N{i}", f"N{j}",
            lambda a, i=i, j=j: a[f"N{i}"] != a[f"N{j}"],
            weight=1.0,
            hardness=Hardness.HARD
        )
    
    # RESTRICCIÓN SOFT 1: Minimizar número de colores
    def minimize_colors(a):
        """Penaliza usar muchos colores."""
        if len(a) < n_nodes:
            return 0.0
        
        unique_colors = len(set(a.values()))
        return (unique_colors - 1) / (n_colors - 1)  # Normalizar
    
    hierarchy.add_global_constraint(
        variables,
        minimize_colors,
        objective="minimize",
        weight=2.0,  # Peso alto: objetivo principal
        hardness=Hardness.SOFT
    )
    
    # RESTRICCIÓN SOFT 2: Preferir colores bajos (0, 1, 2...)
    def prefer_low_colors(a):
        """Penaliza usar colores altos."""
        if len(a) < n_nodes:
            return 0.0
        
        total_color = sum(a.values())
        max_total = n_nodes * (n_colors - 1)  # Máximo posible
        return total_color / max_total if max_total > 0 else 0.0
    
    hierarchy.add_global_constraint(
        variables,
        prefer_low_colors,
        objective="minimize",
        weight=0.5,
        hardness=Hardness.SOFT
    )
    
    # RESTRICCIÓN SOFT 3: Distribución balanceada de colores
    def balanced_colors(a):
        """Penaliza distribución desbalanceada."""
        if len(a) < n_nodes:
            return 0.0
        
        colors = list(a.values())
        color_counts = {c: colors.count(c) for c in range(n_colors)}
        counts = list(color_counts.values())
        
        if not counts:
            return 0.0
        
        mean = sum(counts) / len(counts)
        variance = sum((c - mean) ** 2 for c in counts) / len(counts)
        max_variance = (n_nodes ** 2) / n_colors
        return variance / max_variance if max_variance > 0 else 0.0
    
    hierarchy.add_pattern_constraint(
        variables,
        balanced_colors,
        pattern_type="balanced",
        weight=0.3,
        hardness=Hardness.SOFT
    )
    
    return variables, domains, hard_constraints, hierarchy


# ============================================================================
# EVALUACIÓN DE CALIDAD
# ============================================================================

def evaluate_solution_quality(solution: Dict, hierarchy: ConstraintHierarchy,
                              landscape: EnergyLandscapeOptimized) -> SolutionQuality:
    """Evalúa la calidad de una solución."""
    energy = landscape.compute_energy(solution)
    
    # Contar violaciones HARD
    hard_violations = 0
    for level in ConstraintLevel:
        constraints = hierarchy.get_constraints_at_level(level)
        for constraint in constraints:
            if constraint.hardness == Hardness.HARD:
                satisfied, violation = constraint.evaluate(solution)
                if not satisfied or violation > 0:
                    hard_violations += 1
    
    # Contar violaciones SOFT (suma ponderada)
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
# BENCHMARK RUNNER
# ============================================================================

def run_benchmark_with_quality(solver_name: str, solver, problem_name: str,
                               problem_size: int, hierarchy: ConstraintHierarchy,
                               landscape: EnergyLandscapeOptimized,
                               timeout: float = 60.0) -> BenchmarkResult:
    """Ejecuta benchmark y evalúa calidad de solución."""
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
        nodes = solver.nodes_explored if hasattr(solver, 'nodes_explored') else 0
        found = solution is not None
        
        quality = None
        if found and solution:
            quality = evaluate_solution_quality(solution, hierarchy, landscape)
            print(f"✓ {elapsed:6.2f}s  {nodes:8d} nodos  E={quality.total_energy:.3f} (SOFT={quality.soft_violations:.3f})")
        else:
            print(f"✗ {elapsed:6.2f}s  {nodes:8d} nodos")
        
        return BenchmarkResult(
            solver_name=solver_name,
            problem_name=problem_name,
            problem_size=problem_size,
            time_seconds=elapsed,
            nodes_explored=nodes,
            solution_found=found,
            solution_quality=quality
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
            solution_found=False
        )


def benchmark_nqueens_with_preferences() -> List[BenchmarkResult]:
    """Benchmark N-Queens con preferencias."""
    results = []
    sizes = [8, 10, 12]
    
    for n in sizes:
        print(f"\n{'='*80}")
        print(f"N-Queens con Preferencias: n={n}")
        print(f"  HARD: No atacarse")
        print(f"  SOFT: Preferir centro + distribución balanceada")
        print(f"{'='*80}")
        
        variables, domains, hard_constraints, hierarchy = create_nqueens_with_preferences(n)
        landscape = EnergyLandscapeOptimized(hierarchy)
        
        # Forward Checking (solo HARD)
        solver = ForwardCheckingSolver(variables, domains, hard_constraints)
        result = run_benchmark_with_quality(
            "Forward Checking (HARD only)", solver, "N-Queens+Prefs", n,
            hierarchy, landscape, timeout=30.0
        )
        results.append(result)
        
        # Flujo de Fibración (HARD + SOFT)
        solver = CoherenceSolverOptimized(variables, domains)
        solver.hierarchy = hierarchy
        solver.landscape = landscape
        result = run_benchmark_with_quality(
            "Flujo de Fibración (HARD+SOFT)", solver, "N-Queens+Prefs", n,
            hierarchy, landscape, timeout=30.0
        )
        results.append(result)
    
    return results


def benchmark_graph_coloring_with_preferences() -> List[BenchmarkResult]:
    """Benchmark Graph Coloring con preferencias."""
    results = []
    sizes = [15, 20, 25]
    
    for n in sizes:
        print(f"\n{'='*80}")
        print(f"Graph Coloring con Preferencias: {n} nodos, 4 colores")
        print(f"  HARD: Nodos adyacentes diferentes")
        print(f"  SOFT: Minimizar colores + preferir bajos + balancear")
        print(f"{'='*80}")
        
        variables, domains, hard_constraints, hierarchy = create_graph_coloring_with_preferences(
            n_nodes=n, n_colors=4, edge_probability=0.25
        )
        landscape = EnergyLandscapeOptimized(hierarchy)
        
        # Forward Checking (solo HARD)
        solver = ForwardCheckingSolver(variables, domains, hard_constraints)
        result = run_benchmark_with_quality(
            "Forward Checking (HARD only)", solver, "GraphColor+Prefs", n,
            hierarchy, landscape, timeout=30.0
        )
        results.append(result)
        
        # Flujo de Fibración (HARD + SOFT)
        solver = CoherenceSolverOptimized(variables, domains)
        solver.hierarchy = hierarchy
        solver.landscape = landscape
        result = run_benchmark_with_quality(
            "Flujo de Fibración (HARD+SOFT)", solver, "GraphColor+Prefs", n,
            hierarchy, landscape, timeout=30.0
        )
        results.append(result)
    
    return results


def print_quality_comparison(results: List[BenchmarkResult]):
    """Imprime comparación de calidad de soluciones."""
    print("\n" + "="*90)
    print("COMPARACIÓN DE CALIDAD DE SOLUCIONES")
    print("="*90)
    
    by_problem = {}
    for result in results:
        key = (result.problem_name, result.problem_size)
        if key not in by_problem:
            by_problem[key] = []
        by_problem[key].append(result)
    
    for (problem, size), group in sorted(by_problem.items()):
        print(f"\n{problem} (n={size}):")
        print(f"{'Solver':<35} {'Energía Total':<15} {'SOFT Violations':<18} {'Mejora':<10}")
        print("-" * 90)
        
        # Baseline: Forward Checking
        fc_result = next((r for r in group if "Forward Checking" in r.solver_name), None)
        fc_soft = fc_result.solution_quality.soft_violations if fc_result and fc_result.solution_quality else float('inf')
        
        for result in group:
            if not result.solution_found or not result.solution_quality:
                continue
            
            q = result.solution_quality
            
            improvement = ""
            if fc_soft < float('inf') and q.soft_violations < fc_soft:
                improvement = f"{((fc_soft - q.soft_violations) / fc_soft * 100):.1f}% mejor"
            
            print(f"{result.solver_name:<35} {q.total_energy:<15.3f} {q.soft_violations:<18.3f} {improvement:<10}")


def main():
    """Función principal."""
    print("="*90)
    print("BENCHMARK: RESTRICCIONES SOFT - VENTAJAS DEL FLUJO DE FIBRACIÓN")
    print("="*90)
    print("\nObjetivo: Demostrar que el Flujo de Fibración encuentra soluciones de MEJOR CALIDAD")
    print("          que Forward Checking en problemas con restricciones SOFT.")
    print("\nForward Checking: Encuentra primera solución factible (ignora SOFT)")
    print("Flujo de Fibración: Optimiza restricciones SOFT (mejor calidad)")
    
    all_results = []
    
    # Benchmark 1: N-Queens con preferencias
    print("\n" + "="*90)
    print("BENCHMARK 1: N-QUEENS CON PREFERENCIAS")
    print("="*90)
    results = benchmark_nqueens_with_preferences()
    all_results.extend(results)
    
    # Benchmark 2: Graph Coloring con preferencias
    print("\n" + "="*90)
    print("BENCHMARK 2: GRAPH COLORING CON PREFERENCIAS")
    print("="*90)
    results = benchmark_graph_coloring_with_preferences()
    all_results.extend(results)
    
    # Comparación de calidad
    print_quality_comparison(all_results)
    
    # Estadísticas finales
    print("\n" + "="*90)
    print("ESTADÍSTICAS FINALES")
    print("="*90)
    
    fc_results = [r for r in all_results if "Forward Checking" in r.solver_name and r.solution_found]
    fib_results = [r for r in all_results if "Fibración" in r.solver_name and r.solution_found]
    
    if fc_results and fib_results:
        fc_avg_soft = sum(r.solution_quality.soft_violations for r in fc_results) / len(fc_results)
        fib_avg_soft = sum(r.solution_quality.soft_violations for r in fib_results) / len(fib_results)
        
        improvement = ((fc_avg_soft - fib_avg_soft) / fc_avg_soft * 100) if fc_avg_soft > 0 else 0
        
        print(f"\nPromedio de violaciones SOFT:")
        print(f"  Forward Checking:      {fc_avg_soft:.3f}")
        print(f"  Flujo de Fibración:    {fib_avg_soft:.3f}")
        print(f"  Mejora:                {improvement:.1f}%")
        
        print(f"\n✨ El Flujo de Fibración encuentra soluciones {improvement:.1f}% mejores en promedio")


if __name__ == "__main__":
    main()

