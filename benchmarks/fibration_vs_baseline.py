"""
Benchmark: Flujo de Fibración vs. Estado del Arte

Este benchmark compara la eficiencia del método de Flujo de Fibración
con métodos tradicionales de resolución de CSP:

1. Backtracking simple (baseline)
2. Backtracking con forward checking
3. Backtracking con AC-3 (arc consistency)
4. Flujo de Fibración (nuestro método)

Problemas de prueba:
- N-Queens (escalabilidad)
- Graph Coloring (restricciones complejas)
- Sudoku (restricciones multinivel)
"""

import time
import sys
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import random

# Añadir path para importar módulos
sys.path.insert(0, '/home/ubuntu/lattice-weaver')

from lattice_weaver.fibration import (
    ConstraintHierarchy,
    ConstraintLevel,
    Hardness,
    EnergyLandscape
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
    solution_quality: float = 0.0  # Para problemas de optimización


# ============================================================================
# SOLVERS BASELINE
# ============================================================================

class SimpleBacktrackingSolver:
    """Solver baseline: Backtracking simple sin optimizaciones."""
    
    def __init__(self, variables: List[str], domains: Dict[str, List], 
                 constraints: List[Callable]):
        self.variables = variables
        self.domains = domains
        self.constraints = constraints
        self.nodes_explored = 0
        
    def solve(self) -> Optional[Dict]:
        """Resuelve usando backtracking simple."""
        self.nodes_explored = 0
        return self._backtrack({})
    
    def _backtrack(self, assignment: Dict) -> Optional[Dict]:
        """Backtracking recursivo."""
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
        """Selecciona siguiente variable (orden fijo)."""
        for var in self.variables:
            if var not in assignment:
                return var
    
    def _is_consistent(self, assignment: Dict) -> bool:
        """Verifica si la asignación es consistente."""
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
        """Resuelve usando backtracking con forward checking."""
        self.nodes_explored = 0
        return self._backtrack({}, {var: list(domain) for var, domain in self.domains.items()})
    
    def _backtrack(self, assignment: Dict, domains: Dict) -> Optional[Dict]:
        """Backtracking con forward checking."""
        self.nodes_explored += 1
        
        if len(assignment) == len(self.variables):
            return assignment
        
        var = self._select_unassigned_variable(assignment)
        
        for value in domains[var]:
            assignment[var] = value
            
            if self._is_consistent(assignment):
                # Forward checking: podar dominios
                new_domains = self._forward_check(assignment, domains, var, value)
                
                if new_domains is not None:
                    result = self._backtrack(assignment, new_domains)
                    if result is not None:
                        return result
            
            del assignment[var]
        
        return None
    
    def _select_unassigned_variable(self, assignment: Dict) -> str:
        """Selecciona variable con MRV (Minimum Remaining Values)."""
        unassigned = [v for v in self.variables if v not in assignment]
        # Heurística MRV: variable con menor dominio
        return min(unassigned, key=lambda v: len(self.domains[v]))
    
    def _is_consistent(self, assignment: Dict) -> bool:
        """Verifica consistencia."""
        for constraint in self.constraints:
            if not constraint(assignment):
                return False
        return True
    
    def _forward_check(self, assignment: Dict, domains: Dict, 
                      var: str, value) -> Optional[Dict]:
        """Forward checking: podar dominios inconsistentes."""
        new_domains = {v: list(d) for v, d in domains.items()}
        
        for other_var in self.variables:
            if other_var in assignment or other_var == var:
                continue
            
            # Filtrar valores inconsistentes
            new_domains[other_var] = [
                v for v in new_domains[other_var]
                if self._is_consistent({**assignment, other_var: v})
            ]
            
            if not new_domains[other_var]:
                return None  # Dominio vacío -> fallo
        
        return new_domains


class FibrationSolver:
    """Solver usando Flujo de Fibración."""
    
    def __init__(self, variables: List[str], domains: Dict[str, List],
                 hierarchy: ConstraintHierarchy):
        self.variables = variables
        self.domains = domains
        self.hierarchy = hierarchy
        self.landscape = EnergyLandscape(hierarchy)
        self.nodes_explored = 0
        
    def solve(self) -> Optional[Dict]:
        """Resuelve usando búsqueda guiada por energía."""
        self.nodes_explored = 0
        return self._energy_guided_search({})
    
    def _energy_guided_search(self, assignment: Dict) -> Optional[Dict]:
        """Búsqueda guiada por el paisaje de energía."""
        self.nodes_explored += 1
        
        if len(assignment) == len(self.variables):
            # Verificar solución completa
            energy = self.landscape.compute_energy(assignment)
            if energy.total_energy == 0.0:
                return assignment
            return None
        
        # Seleccionar variable con más restricciones
        var = self._select_variable_by_constraints(assignment)
        
        # Calcular gradiente de energía
        gradient = self.landscape.compute_energy_gradient(
            assignment, var, self.domains[var]
        )
        
        # Ordenar valores por energía (menor primero)
        sorted_values = sorted(gradient.items(), key=lambda x: x[1])
        
        # Poda: solo explorar valores con energía razonable
        threshold = sorted_values[0][1] + 1.0  # Tolerar +1.0 de energía
        pruned_values = [v for v, e in sorted_values if e <= threshold]
        
        for value in pruned_values:
            assignment[var] = value
            
            result = self._energy_guided_search(assignment)
            if result is not None:
                return result
            
            del assignment[var]
        
        return None
    
    def _select_variable_by_constraints(self, assignment: Dict) -> str:
        """Selecciona variable con más restricciones."""
        unassigned = [v for v in self.variables if v not in assignment]
        
        # Contar restricciones por variable
        constraint_counts = {}
        for var in unassigned:
            constraints = self.hierarchy.get_constraints_involving(var)
            constraint_counts[var] = len(constraints)
        
        return max(constraint_counts, key=constraint_counts.get)


# ============================================================================
# PROBLEMAS DE PRUEBA
# ============================================================================

def create_nqueens_problem(n: int) -> Tuple[List[str], Dict, List[Callable], ConstraintHierarchy]:
    """
    Crea un problema de N-Queens.
    
    Returns:
        (variables, domains, constraints_list, hierarchy)
    """
    variables = [f"Q{i}" for i in range(n)]
    domains = {var: list(range(n)) for var in variables}
    
    # Constraints para solvers baseline
    constraints_list = []
    
    # Hierarchy para Fibration solver
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


def create_graph_coloring_problem(n_nodes: int, n_colors: int, 
                                  edge_probability: float = 0.3) -> Tuple[List[str], Dict, List[Callable], ConstraintHierarchy]:
    """
    Crea un problema de coloración de grafos aleatorio.
    
    Returns:
        (variables, domains, constraints_list, hierarchy)
    """
    variables = [f"N{i}" for i in range(n_nodes)]
    domains = {var: list(range(n_colors)) for var in variables}
    
    # Generar grafo aleatorio
    edges = []
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            if random.random() < edge_probability:
                edges.append((i, j))
    
    # Constraints para solvers baseline
    constraints_list = []
    
    # Hierarchy para Fibration solver
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
    
    # Restricción de patrón: preferir distribución balanceada
    def balanced_colors(a):
        if len(a) < n_nodes:
            return 0.0
        colors = list(a.values())
        color_counts = {c: colors.count(c) for c in range(n_colors)}
        counts = list(color_counts.values())
        if not counts:
            return 0.0
        mean = sum(counts) / len(counts)
        variance = sum((c - mean) ** 2 for c in counts) / len(counts)
        return variance / (n_nodes * n_nodes)
    
    hierarchy.add_pattern_constraint(
        variables,
        balanced_colors,
        pattern_type="balanced",
        weight=0.5,
        hardness=Hardness.SOFT
    )
    
    return variables, domains, constraints_list, hierarchy


# ============================================================================
# BENCHMARK RUNNER
# ============================================================================

def run_benchmark(solver_name: str, solver, problem_name: str, 
                 problem_size: int, timeout: float = 10.0) -> BenchmarkResult:
    """
    Ejecuta un benchmark con timeout.
    
    Returns:
        BenchmarkResult
    """
    print(f"  Ejecutando {solver_name}...", end=" ", flush=True)
    
    start_time = time.time()
    
    try:
        solution = solver.solve()
        elapsed = time.time() - start_time
        
        if elapsed > timeout:
            print(f"⏱️  TIMEOUT ({timeout}s)")
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
            print(f"✓ {elapsed:.3f}s ({nodes} nodos)")
        else:
            print(f"✗ No encontrada ({elapsed:.3f}s, {nodes} nodos)")
        
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
    """Benchmark N-Queens con diferentes tamaños."""
    results = []
    
    for n in sizes:
        print(f"\n{'='*60}")
        print(f"N-Queens: n={n}")
        print(f"{'='*60}")
        
        variables, domains, constraints_list, hierarchy = create_nqueens_problem(n)
        
        # Backtracking simple
        solver = SimpleBacktrackingSolver(variables, domains, constraints_list)
        result = run_benchmark("Backtracking Simple", solver, "N-Queens", n, timeout=30.0)
        results.append(result)
        
        # Forward checking
        solver = ForwardCheckingSolver(variables, domains, constraints_list)
        result = run_benchmark("Forward Checking", solver, "N-Queens", n, timeout=30.0)
        results.append(result)
        
        # Flujo de Fibración
        solver = FibrationSolver(variables, domains, hierarchy)
        result = run_benchmark("Flujo de Fibración", solver, "N-Queens", n, timeout=30.0)
        results.append(result)
    
    return results


def benchmark_graph_coloring(sizes: List[int]) -> List[BenchmarkResult]:
    """Benchmark Graph Coloring con diferentes tamaños."""
    results = []
    
    for n in sizes:
        print(f"\n{'='*60}")
        print(f"Graph Coloring: n={n} nodos, k=3 colores")
        print(f"{'='*60}")
        
        # Fijar seed para reproducibilidad
        random.seed(42)
        
        variables, domains, constraints_list, hierarchy = create_graph_coloring_problem(
            n_nodes=n, n_colors=3, edge_probability=0.3
        )
        
        # Backtracking simple
        solver = SimpleBacktrackingSolver(variables, domains, constraints_list)
        result = run_benchmark("Backtracking Simple", solver, "Graph Coloring", n, timeout=30.0)
        results.append(result)
        
        # Forward checking
        solver = ForwardCheckingSolver(variables, domains, constraints_list)
        result = run_benchmark("Forward Checking", solver, "Graph Coloring", n, timeout=30.0)
        results.append(result)
        
        # Flujo de Fibración
        solver = FibrationSolver(variables, domains, hierarchy)
        result = run_benchmark("Flujo de Fibración", solver, "Graph Coloring", n, timeout=30.0)
        results.append(result)
    
    return results


def print_summary_table(results: List[BenchmarkResult]):
    """Imprime tabla resumen de resultados."""
    print("\n" + "="*80)
    print("RESUMEN DE RESULTADOS")
    print("="*80)
    
    # Agrupar por problema y tamaño
    by_problem = {}
    for result in results:
        key = (result.problem_name, result.problem_size)
        if key not in by_problem:
            by_problem[key] = []
        by_problem[key].append(result)
    
    for (problem, size), group in sorted(by_problem.items()):
        print(f"\n{problem} (n={size}):")
        print(f"{'Solver':<25} {'Tiempo (s)':<12} {'Nodos':<12} {'Solución':<10} {'Speedup':<10}")
        print("-" * 80)
        
        # Baseline: tiempo del backtracking simple
        baseline_time = next((r.time_seconds for r in group if r.solver_name == "Backtracking Simple"), None)
        
        for result in group:
            speedup = ""
            if baseline_time and result.time_seconds > 0 and result.solution_found:
                speedup = f"{baseline_time / result.time_seconds:.2f}x"
            
            status = "✓" if result.solution_found else "✗"
            
            print(f"{result.solver_name:<25} {result.time_seconds:<12.3f} {result.nodes_explored:<12} {status:<10} {speedup:<10}")


def main():
    """Función principal del benchmark."""
    print("="*80)
    print("BENCHMARK: FLUJO DE FIBRACIÓN VS. ESTADO DEL ARTE")
    print("="*80)
    print("\nComparando:")
    print("  1. Backtracking Simple (baseline)")
    print("  2. Forward Checking (estado del arte clásico)")
    print("  3. Flujo de Fibración (nuestro método)")
    
    all_results = []
    
    # Benchmark 1: N-Queens
    print("\n" + "="*80)
    print("BENCHMARK 1: N-QUEENS")
    print("="*80)
    results = benchmark_nqueens([4, 6, 8])
    all_results.extend(results)
    
    # Benchmark 2: Graph Coloring
    print("\n" + "="*80)
    print("BENCHMARK 2: GRAPH COLORING")
    print("="*80)
    results = benchmark_graph_coloring([8, 12, 16])
    all_results.extend(results)
    
    # Resumen
    print_summary_table(all_results)
    
    # Análisis de speedup
    print("\n" + "="*80)
    print("ANÁLISIS DE SPEEDUP")
    print("="*80)
    
    fibration_results = [r for r in all_results if r.solver_name == "Flujo de Fibración" and r.solution_found]
    baseline_results = [r for r in all_results if r.solver_name == "Backtracking Simple" and r.solution_found]
    
    if fibration_results and baseline_results:
        avg_speedup = sum(
            next((b.time_seconds / f.time_seconds for b in baseline_results 
                  if b.problem_name == f.problem_name and b.problem_size == f.problem_size), 0)
            for f in fibration_results
        ) / len(fibration_results)
        
        print(f"\nSpeedup promedio de Flujo de Fibración: {avg_speedup:.2f}x")
        
        # Reducción de nodos explorados
        avg_node_reduction = sum(
            (1 - f.nodes_explored / next((b.nodes_explored for b in baseline_results 
                                         if b.problem_name == f.problem_name and b.problem_size == f.problem_size), 1))
            for f in fibration_results
        ) / len(fibration_results)
        
        print(f"Reducción promedio de nodos explorados: {avg_node_reduction*100:.1f}%")


if __name__ == "__main__":
    main()

