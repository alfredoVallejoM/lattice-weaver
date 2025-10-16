"""
Benchmarks Comparativos: Fibration Flow vs Estado del Arte en CSP

Este módulo compara Fibration Flow con técnicas clásicas de CSP para identificar
en qué tipos de problemas destaca nuestra implementación.

Solvers comparados:
1. Backtracking Simple (baseline)
2. Forward Checking
3. AC-3 + Backtracking
4. Fibration Flow (optimizado completo)

Problemas benchmark:
- N-Queens (4, 6, 8, 10, 12)
- Graph Coloring (diferentes topologías)
- Sudoku (4x4, 9x9)
- Map Coloring

Autor: Agente Autónomo - Lattice Weaver
Fecha: 15 de Octubre, 2025
"""

import time
import json
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field

# TODO: Descomentar cuando se integre el componente
# from lattice_weaver.fibration.fibration_search_solver_enhanced import FibrationSearchSolverEnhanced
from lattice_weaver.fibration.constraint_hierarchy import ConstraintHierarchy, Hardness, ConstraintLevel
from lattice_weaver.fibration.energy_landscape_optimized import EnergyLandscapeOptimized
# TODO: Descomentar cuando se integre el componente
# from lattice_weaver.arc_engine.core import ArcEngine


@dataclass
class BenchmarkResult:
    """Resultado de un benchmark."""
    solver_name: str
    problem_name: str
    problem_size: int
    
    # Métricas de tiempo
    time_seconds: float
    
    # Métricas de búsqueda
    nodes_explored: int
    backtracks: int
    
    # Solución
    solution_found: bool
    solution_quality: float = 0.0
    
    # Metadata
    timeout: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class SimpleBacktrackingSolver:
    """Solver de backtracking simple (baseline)."""
    
    def __init__(self, variables: List[str], domains: Dict[str, List[Any]], 
                 constraints: List[Tuple[List[str], Any]]):
        self.variables = variables
        self.domains = domains
        self.constraints = constraints
        self.nodes_explored = 0
        self.backtracks = 0
        self.solution = None
    
    def solve(self, time_limit: float = 30.0) -> Optional[Dict[str, Any]]:
        """Resuelve el CSP con backtracking simple."""
        self.start_time = time.time()
        self.time_limit = time_limit
        self.nodes_explored = 0
        self.backtracks = 0
        
        assignment = {}
        if self._backtrack(assignment):
            return self.solution
        return None
    
    def _backtrack(self, assignment: Dict[str, Any]) -> bool:
        """Backtracking recursivo."""
        if time.time() - self.start_time > self.time_limit:
            return False
        
        self.nodes_explored += 1
        
        if len(assignment) == len(self.variables):
            self.solution = assignment.copy()
            return True
        
        var = self._select_unassigned_variable(assignment)
        
        for value in self.domains[var]:
            assignment[var] = value
            
            if self._is_consistent(assignment):
                if self._backtrack(assignment):
                    return True
            
            del assignment[var]
            self.backtracks += 1
        
        return False
    
    def _select_unassigned_variable(self, assignment: Dict[str, Any]) -> str:
        """Selecciona la primera variable no asignada."""
        for var in self.variables:
            if var not in assignment:
                return var
        return None
    
    def _is_consistent(self, assignment: Dict[str, Any]) -> bool:
        """Verifica si la asignación es consistente."""
        for variables, predicate in self.constraints:
            if all(v in assignment for v in variables):
                if not predicate(assignment):
                    return False
        return True


class ForwardCheckingSolver:
    """Solver con Forward Checking."""
    
    def __init__(self, variables: List[str], domains: Dict[str, List[Any]], 
                 constraints: List[Tuple[List[str], Any]]):
        self.variables = variables
        self.original_domains = {v: list(d) for v, d in domains.items()}
        self.constraints = constraints
        self.nodes_explored = 0
        self.backtracks = 0
        self.solution = None
    
    def solve(self, time_limit: float = 30.0) -> Optional[Dict[str, Any]]:
        """Resuelve el CSP con forward checking."""
        self.start_time = time.time()
        self.time_limit = time_limit
        self.nodes_explored = 0
        self.backtracks = 0
        
        assignment = {}
        domains = {v: list(d) for v, d in self.original_domains.items()}
        
        if self._backtrack(assignment, domains):
            return self.solution
        return None
    
    def _backtrack(self, assignment: Dict[str, Any], domains: Dict[str, List[Any]]) -> bool:
        """Backtracking con forward checking."""
        if time.time() - self.start_time > self.time_limit:
            return False
        
        self.nodes_explored += 1
        
        if len(assignment) == len(self.variables):
            self.solution = assignment.copy()
            return True
        
        var = self._select_unassigned_variable(assignment, domains)
        
        for value in list(domains[var]):
            assignment[var] = value
            
            # Forward checking: podar dominios
            new_domains = self._forward_check(var, value, assignment, domains)
            
            if new_domains is not None:
                if self._backtrack(assignment, new_domains):
                    return True
            
            del assignment[var]
            self.backtracks += 1
        
        return False
    
    def _select_unassigned_variable(self, assignment: Dict[str, Any], 
                                   domains: Dict[str, List[Any]]) -> str:
        """Selecciona variable con MRV (Minimum Remaining Values)."""
        unassigned = [v for v in self.variables if v not in assignment]
        return min(unassigned, key=lambda v: len(domains[v]))
    
    def _forward_check(self, var: str, value: Any, assignment: Dict[str, Any],
                      domains: Dict[str, List[Any]]) -> Optional[Dict[str, List[Any]]]:
        """Poda dominios de variables futuras."""
        new_domains = {v: list(d) for v, d in domains.items()}
        
        for variables, predicate in self.constraints:
            if var in variables:
                for other_var in variables:
                    if other_var != var and other_var not in assignment:
                        # Podar valores inconsistentes
                        new_domain = []
                        for other_value in new_domains[other_var]:
                            temp_assignment = assignment.copy()
                            temp_assignment[other_var] = other_value
                            if predicate(temp_assignment):
                                new_domain.append(other_value)
                        
                        new_domains[other_var] = new_domain
                        
                        if not new_domain:
                            return None  # Domain wipeout
        
        return new_domains
    
    def _is_consistent(self, assignment: Dict[str, Any]) -> bool:
        """Verifica consistencia."""
        for variables, predicate in self.constraints:
            if all(v in assignment for v in variables):
                if not predicate(assignment):
                    return False
        return True


class StateOfTheArtComparison:
    """Comparación de Fibration Flow con estado del arte."""
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
    
    def run_all_benchmarks(self):
        """Ejecuta todos los benchmarks comparativos."""
        print("=" * 100)
        print("FIBRATION FLOW VS ESTADO DEL ARTE EN CSP")
        print("=" * 100)
        print()
        
        # Benchmark 1: N-Queens (escalabilidad)
        print("\n" + "=" * 100)
        print("BENCHMARK 1: N-QUEENS (Escalabilidad)")
        print("=" * 100)
        self._benchmark_nqueens_scalability()
        
        # Benchmark 2: Graph Coloring (diferentes topologías)
        print("\n" + "=" * 100)
        print("BENCHMARK 2: GRAPH COLORING (Topologías)")
        print("=" * 100)
        self._benchmark_graph_coloring_topologies()
        
        # Generar reporte final
        self._generate_final_report()
    
    def _benchmark_nqueens_scalability(self):
        """Benchmark de escalabilidad en N-Queens."""
        sizes = [4, 6, 8, 10]
        
        for n in sizes:
            print(f"\n{n}-Queens:")
            print("-" * 100)
            
            # Crear problema
            hierarchy, variables, domains, constraints = self._create_nqueens_problem(n)
            
            # 1. Backtracking Simple
            result_bt = self._run_simple_backtracking(
                f"{n}-Queens", n, variables, domains, constraints
            )
            self.results.append(result_bt)
            self._print_result(result_bt)
            
            # 2. Forward Checking
            result_fc = self._run_forward_checking(
                f"{n}-Queens", n, variables, domains, constraints
            )
            self.results.append(result_fc)
            self._print_result(result_fc)
            
            # 3. Fibration Flow
            result_ff = self._run_fibration_flow(
                f"{n}-Queens", n, hierarchy, variables, domains
            )
            self.results.append(result_ff)
            self._print_result(result_ff)
            
            # Comparación
            self._print_comparison(n, [result_bt, result_fc, result_ff])
    
    def _benchmark_graph_coloring_topologies(self):
        """Benchmark de diferentes topologías de grafos."""
        test_cases = [
            ("cycle", 10, 3),
            ("bipartite", 12, 2),
            ("random", 10, 4),
        ]
        
        for graph_type, n_nodes, n_colors in test_cases:
            print(f"\nGraph Coloring ({graph_type}, {n_nodes} nodos, {n_colors} colores):")
            print("-" * 100)
            
            # Crear problema
            hierarchy, variables, domains, constraints = self._create_graph_coloring_problem(
                n_nodes, n_colors, graph_type
            )
            
            # 1. Backtracking Simple
            result_bt = self._run_simple_backtracking(
                f"GC-{graph_type}", n_nodes, variables, domains, constraints
            )
            self.results.append(result_bt)
            self._print_result(result_bt)
            
            # 2. Forward Checking
            result_fc = self._run_forward_checking(
                f"GC-{graph_type}", n_nodes, variables, domains, constraints
            )
            self.results.append(result_fc)
            self._print_result(result_fc)
            
            # 3. Fibration Flow
            result_ff = self._run_fibration_flow(
                f"GC-{graph_type}", n_nodes, hierarchy, variables, domains
            )
            self.results.append(result_ff)
            self._print_result(result_ff)
            
            # Comparación
            self._print_comparison(f"{graph_type}-{n_nodes}", [result_bt, result_fc, result_ff])
    
    def _create_nqueens_problem(self, n: int):
        """Crea problema N-Queens."""
        hierarchy = ConstraintHierarchy()
        variables = [f"Q{i}" for i in range(n)]
        domains = {var: list(range(n)) for var in variables}
        constraints = []
        
        for i in range(n):
            for j in range(i + 1, n):
                # No misma columna
                def ne_constraint(assignment, i=i, j=j):
                    qi, qj = f"Q{i}", f"Q{j}"
                    if qi in assignment and qj in assignment:
                        return assignment[qi] != assignment[qj]
                    return True
                
                hierarchy.add_local_constraint(
                    var1=f"Q{i}", var2=f"Q{j}",
                    predicate=ne_constraint,
                    hardness=Hardness.HARD,
                    metadata={"name": f"Q{i}_ne_Q{j}"}
                )
                constraints.append(([f"Q{i}", f"Q{j}"], ne_constraint))
                
                # No misma diagonal
                def no_diagonal(assignment, i=i, j=j):
                    qi, qj = f"Q{i}", f"Q{j}"
                    if qi in assignment and qj in assignment:
                        return abs(assignment[qi] - assignment[qj]) != abs(i - j)
                    return True
                
                hierarchy.add_local_constraint(
                    var1=f"Q{i}", var2=f"Q{j}",
                    predicate=no_diagonal,
                    hardness=Hardness.HARD,
                    metadata={"name": f"Q{i}_nodiag_Q{j}"}
                )
                constraints.append(([f"Q{i}", f"Q{j}"], no_diagonal))
        
        return hierarchy, variables, domains, constraints
    
    def _create_graph_coloring_problem(self, n_nodes: int, n_colors: int, graph_type: str):
        """Crea problema de coloreo de grafos."""
        hierarchy = ConstraintHierarchy()
        variables = [f"N{i}" for i in range(n_nodes)]
        domains = {var: list(range(n_colors)) for var in variables}
        constraints = []
        
        # Generar aristas
        edges = []
        if graph_type == "cycle":
            for i in range(n_nodes):
                j = (i + 1) % n_nodes
                edges.append((i, j))
        elif graph_type == "bipartite":
            half = n_nodes // 2
            for i in range(half):
                for j in range(half, n_nodes):
                    edges.append((i, j))
        elif graph_type == "random":
            import random
            random.seed(42)
            for i in range(n_nodes):
                for j in range(i + 1, n_nodes):
                    if random.random() < 0.3:
                        edges.append((i, j))
        
        # Añadir restricciones
        for i, j in edges:
            def ne_constraint(assignment, i=i, j=j):
                ni, nj = f"N{i}", f"N{j}"
                if ni in assignment and nj in assignment:
                    return assignment[ni] != assignment[nj]
                return True
            
            hierarchy.add_local_constraint(
                var1=f"N{i}", var2=f"N{j}",
                predicate=ne_constraint,
                hardness=Hardness.HARD,
                metadata={"name": f"N{i}_ne_N{j}"}
            )
            constraints.append(([f"N{i}", f"N{j}"], ne_constraint))
        
        return hierarchy, variables, domains, constraints
    
    def _run_simple_backtracking(self, problem_name: str, problem_size: int,
                                variables: List[str], domains: Dict[str, List[Any]],
                                constraints: List) -> BenchmarkResult:
        """Ejecuta backtracking simple."""
        solver = SimpleBacktrackingSolver(variables, domains, constraints)
        
        start_time = time.time()
        solution = solver.solve(time_limit=30.0)
        elapsed = time.time() - start_time
        
        return BenchmarkResult(
            solver_name="Backtracking Simple",
            problem_name=problem_name,
            problem_size=problem_size,
            time_seconds=elapsed,
            nodes_explored=solver.nodes_explored,
            backtracks=solver.backtracks,
            solution_found=solution is not None,
            timeout=elapsed >= 30.0
        )
    
    def _run_forward_checking(self, problem_name: str, problem_size: int,
                            variables: List[str], domains: Dict[str, List[Any]],
                            constraints: List) -> BenchmarkResult:
        """Ejecuta forward checking."""
        solver = ForwardCheckingSolver(variables, domains, constraints)
        
        start_time = time.time()
        solution = solver.solve(time_limit=30.0)
        elapsed = time.time() - start_time
        
        return BenchmarkResult(
            solver_name="Forward Checking",
            problem_name=problem_name,
            problem_size=problem_size,
            time_seconds=elapsed,
            nodes_explored=solver.nodes_explored,
            backtracks=solver.backtracks,
            solution_found=solution is not None,
            timeout=elapsed >= 30.0
        )
    
    def _run_fibration_flow(self, problem_name: str, problem_size: int,
                           hierarchy: ConstraintHierarchy, variables: List[str],
                           domains: Dict[str, List[Any]]) -> BenchmarkResult:
        """Ejecuta Fibration Flow."""
        landscape = EnergyLandscapeOptimized(hierarchy)
        arc_engine = ArcEngine(use_tms=True, parallel=False)
        
        solver = FibrationSearchSolverEnhanced(
            hierarchy=hierarchy,
            landscape=landscape,
            arc_engine=arc_engine,
            variables=variables,
            domains=domains,
            use_homotopy=True,
            use_tms=True,
            use_enhanced_heuristics=True,
            max_backtracks=50000,
            max_iterations=50000,
            time_limit_seconds=30.0
        )
        
        start_time = time.time()
        solution = solver.solve()
        elapsed = time.time() - start_time
        
        stats = solver.get_statistics()
        
        return BenchmarkResult(
            solver_name="Fibration Flow",
            problem_name=problem_name,
            problem_size=problem_size,
            time_seconds=elapsed,
            nodes_explored=stats['search']['nodes_explored'],
            backtracks=stats['search']['backtracks'],
            solution_found=solution is not None,
            solution_quality=stats['solution']['energy'],
            timeout=elapsed >= 30.0,
            metadata={'backjumps': stats['search']['backjumps']}
        )
    
    def _print_result(self, result: BenchmarkResult):
        """Imprime un resultado."""
        timeout_str = " (TIMEOUT)" if result.timeout else ""
        found_str = "✓" if result.solution_found else "✗"
        
        print(f"  {result.solver_name:<25} {found_str}  "
              f"Tiempo: {result.time_seconds:>7.3f}s  "
              f"Nodos: {result.nodes_explored:>8}  "
              f"Backtracks: {result.backtracks:>8}{timeout_str}")
    
    def _print_comparison(self, problem_id: str, results: List[BenchmarkResult]):
        """Imprime comparación de resultados."""
        print(f"\n  Comparación ({problem_id}):")
        
        if all(r.solution_found for r in results):
            baseline = results[0]
            
            for r in results[1:]:
                time_speedup = baseline.time_seconds / r.time_seconds if r.time_seconds > 0 else float('inf')
                backtrack_reduction = baseline.backtracks / r.backtracks if r.backtracks > 0 else float('inf')
                
                print(f"    {r.solver_name} vs Baseline: "
                      f"Speedup={time_speedup:.2f}x, "
                      f"Backtrack reduction={backtrack_reduction:.2f}x")
        print()
    
    def _generate_final_report(self):
        """Genera reporte final."""
        print("\n" + "=" * 100)
        print("RESUMEN FINAL")
        print("=" * 100)
        
        # Agrupar por solver
        by_solver = {}
        for r in self.results:
            if r.solver_name not in by_solver:
                by_solver[r.solver_name] = []
            by_solver[r.solver_name].append(r)
        
        # Estadísticas por solver
        print("\nEstadísticas por Solver:")
        print("-" * 100)
        print(f"{'Solver':<25} {'Problemas resueltos':<20} {'Tiempo promedio':<20} {'Backtracks promedio':<20}")
        print("-" * 100)
        
        for solver_name, results in by_solver.items():
            solved = sum(1 for r in results if r.solution_found)
            avg_time = sum(r.time_seconds for r in results) / len(results)
            avg_backtracks = sum(r.backtracks for r in results) / len(results)
            
            print(f"{solver_name:<25} {solved}/{len(results):<19} {avg_time:>10.3f}s{'':<9} {avg_backtracks:>10.0f}")
        
        print("=" * 100)
        
        # Guardar resultados
        with open('/home/ubuntu/docs/state_of_the_art_comparison.json', 'w') as f:
            json.dump([{
                'solver_name': r.solver_name,
                'problem_name': r.problem_name,
                'problem_size': r.problem_size,
                'time_seconds': r.time_seconds,
                'nodes_explored': r.nodes_explored,
                'backtracks': r.backtracks,
                'solution_found': r.solution_found,
                'timeout': r.timeout
            } for r in self.results], f, indent=2)
        
        print(f"\n✓ Resultados guardados en: /home/ubuntu/docs/state_of_the_art_comparison.json")


def main():
    """Función principal."""
    comparison = StateOfTheArtComparison()
    comparison.run_all_benchmarks()


if __name__ == "__main__":
    main()

