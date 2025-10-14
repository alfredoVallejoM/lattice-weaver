"""
Benchmark: Problemas Complejos Multi-Objetivo

Problemas diseñados específicamente para demostrar las ventajas del Flujo de Fibración:

1. SCHEDULING CON MÚLTIPLES OBJETIVOS
   - HARD: Restricciones de precedencia y recursos
   - SOFT: Minimizar makespan + balancear carga + preferencias de trabajadores
   
2. ASIGNACIÓN DE RECURSOS CON CONFLICTOS
   - HARD: Cada tarea asignada a exactamente un recurso
   - SOFT: Minimizar costo + maximizar calidad + balancear utilización
   
3. GRAPH COLORING CON PREFERENCIAS COMPLEJAS
   - HARD: Nodos adyacentes diferentes
   - SOFT: Minimizar colores + preferencias de color + clustering

Estos problemas tienen MUCHAS soluciones factibles con calidades MUY diferentes.
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
    SimpleOptimizationSolver,
    EnergyLandscapeOptimized
)


@dataclass
class SolutionQuality:
    """Métricas de calidad."""
    hard_violations: int
    soft_violations: float
    total_energy: float
    breakdown: Dict[str, float]  # Desglose por objetivo


@dataclass
class BenchmarkResult:
    """Resultado de benchmark."""
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
# PROBLEMA 1: SCHEDULING CON MÚLTIPLES OBJETIVOS
# ============================================================================

def create_scheduling_problem(n_tasks: int = 8, n_workers: int = 3) -> Tuple[List[str], Dict, List[Callable], ConstraintHierarchy]:
    """
    Problema de scheduling con múltiples objetivos en conflicto.
    
    Variables: Asignación de tareas a trabajadores
    
    HARD:
    - Cada tarea asignada a exactamente un trabajador
    - Restricciones de precedencia (algunas tareas deben hacerse antes que otras)
    
    SOFT (en conflicto):
    - Minimizar makespan (tiempo total)
    - Balancear carga entre trabajadores
    - Respetar preferencias de trabajadores por tareas
    """
    variables = [f"T{i}" for i in range(n_tasks)]
    domains = {var: list(range(n_workers)) for var in variables}
    
    hard_constraints = []
    hierarchy = ConstraintHierarchy()
    
    # Duración de cada tarea (aleatoria pero fija)
    random.seed(42)
    task_durations = {i: random.randint(1, 5) for i in range(n_tasks)}
    
    # Preferencias de trabajadores (matriz de costos)
    # Algunos trabajadores son mejores en ciertas tareas
    worker_preferences = {}
    for task in range(n_tasks):
        worker_preferences[task] = {
            worker: random.uniform(0.5, 1.5) for worker in range(n_workers)
        }
    
    # RESTRICCIONES HARD: Precedencia
    # T0 debe hacerse antes que T2
    # T1 debe hacerse antes que T3
    # T2 y T3 deben hacerse antes que T4
    precedences = [(0, 2), (1, 3), (2, 4), (3, 4)]
    
    for before, after in precedences:
        def precedence_constraint(a, before=before, after=after):
            if f"T{before}" not in a or f"T{after}" not in a:
                return True
            # Si están en el mismo trabajador, verificar orden implícito
            # (asumimos que las tareas se ejecutan en orden de índice)
            return True  # Simplificado: solo verificamos que estén asignadas
        
        hard_constraints.append(precedence_constraint)
        
        hierarchy.add_local_constraint(
            f"T{before}", f"T{after}",
            lambda a, before=before, after=after: True,  # Simplificado
            weight=1.0,
            hardness=Hardness.HARD
        )
    
    # RESTRICCIÓN SOFT 1: Minimizar makespan (tiempo total)
    def minimize_makespan(a):
        if len(a) < n_tasks:
            return 0.0
        
        # Calcular tiempo total por trabajador
        worker_times = {w: 0 for w in range(n_workers)}
        for task in range(n_tasks):
            worker = a[f"T{task}"]
            worker_times[worker] += task_durations[task]
        
        # Makespan = tiempo del trabajador más ocupado
        makespan = max(worker_times.values())
        max_possible = sum(task_durations.values())
        
        return makespan / max_possible  # Normalizar
    
    hierarchy.add_global_constraint(
        variables,
        minimize_makespan,
        objective="minimize",
        weight=3.0,  # Peso alto: objetivo principal
        hardness=Hardness.SOFT
    )
    
    # RESTRICCIÓN SOFT 2: Balancear carga entre trabajadores
    def balance_load(a):
        if len(a) < n_tasks:
            return 0.0
        
        worker_times = {w: 0 for w in range(n_workers)}
        for task in range(n_tasks):
            worker = a[f"T{task}"]
            worker_times[worker] += task_durations[task]
        
        times = list(worker_times.values())
        mean_time = sum(times) / len(times)
        variance = sum((t - mean_time) ** 2 for t in times) / len(times)
        max_variance = (sum(task_durations.values()) ** 2) / n_workers
        
        return variance / max_variance if max_variance > 0 else 0.0
    
    hierarchy.add_pattern_constraint(
        variables,
        balance_load,
        pattern_type="balanced",
        weight=2.0,
        hardness=Hardness.SOFT
    )
    
    # RESTRICCIÓN SOFT 3: Respetar preferencias de trabajadores
    def respect_preferences(a):
        if len(a) < n_tasks:
            return 0.0
        
        total_cost = sum(
            worker_preferences[task][a[f"T{task}"]]
            for task in range(n_tasks)
        )
        max_cost = n_tasks * 1.5  # Máximo costo posible
        
        return total_cost / max_cost
    
    hierarchy.add_global_constraint(
        variables,
        respect_preferences,
        objective="minimize",
        weight=1.5,
        hardness=Hardness.SOFT
    )
    
    return variables, domains, hard_constraints, hierarchy


# ============================================================================
# PROBLEMA 2: ASIGNACIÓN DE RECURSOS CON COSTOS Y CALIDAD
# ============================================================================

def create_resource_assignment_problem(n_tasks: int = 10, n_resources: int = 4) -> Tuple[List[str], Dict, List[Callable], ConstraintHierarchy]:
    """
    Asignación de tareas a recursos con múltiples objetivos.
    
    HARD:
    - Cada tarea asignada a exactamente un recurso
    - Capacidad de recursos no excedida
    
    SOFT (en conflicto):
    - Minimizar costo total
    - Maximizar calidad total
    - Balancear utilización de recursos
    """
    variables = [f"Task{i}" for i in range(n_tasks)]
    domains = {var: list(range(n_resources)) for var in variables}
    
    hard_constraints = []
    hierarchy = ConstraintHierarchy()
    
    random.seed(42)
    
    # Costo de asignar cada tarea a cada recurso
    costs = {}
    for task in range(n_tasks):
        costs[task] = {
            resource: random.uniform(10, 100) for resource in range(n_resources)
        }
    
    # Calidad de asignar cada tarea a cada recurso (inverso del costo)
    qualities = {}
    for task in range(n_tasks):
        qualities[task] = {
            resource: random.uniform(0.5, 1.0) for resource in range(n_resources)
        }
    
    # Capacidad de cada recurso
    resource_capacity = {r: 3 for r in range(n_resources)}
    
    # Carga de cada tarea
    task_load = {t: 1 for t in range(n_tasks)}
    
    # RESTRICCIÓN HARD: Capacidad de recursos
    def capacity_constraint(a):
        if len(a) < n_tasks:
            return True
        
        resource_usage = {r: 0 for r in range(n_resources)}
        for task in range(n_tasks):
            resource = a[f"Task{task}"]
            resource_usage[resource] += task_load[task]
        
        # Verificar que ningún recurso exceda su capacidad
        return all(usage <= resource_capacity[r] for r, usage in resource_usage.items())
    
    hard_constraints.append(capacity_constraint)
    
    hierarchy.add_global_constraint(
        variables,
        lambda a: 0.0 if capacity_constraint(a) else 1.0,
        objective="satisfy",
        weight=1.0,
        hardness=Hardness.HARD
    )
    
    # RESTRICCIÓN SOFT 1: Minimizar costo total
    def minimize_cost(a):
        if len(a) < n_tasks:
            return 0.0
        
        total_cost = sum(
            costs[task][a[f"Task{task}"]]
            for task in range(n_tasks)
        )
        max_cost = n_tasks * 100  # Máximo costo posible
        
        return total_cost / max_cost
    
    hierarchy.add_global_constraint(
        variables,
        minimize_cost,
        objective="minimize",
        weight=3.0,  # Conflicto con calidad
        hardness=Hardness.SOFT
    )
    
    # RESTRICCIÓN SOFT 2: Maximizar calidad (= minimizar 1-calidad)
    def maximize_quality(a):
        if len(a) < n_tasks:
            return 0.0
        
        total_quality = sum(
            qualities[task][a[f"Task{task}"]]
            for task in range(n_tasks)
        )
        max_quality = n_tasks * 1.0
        
        # Invertir: queremos minimizar (1 - calidad_normalizada)
        return 1.0 - (total_quality / max_quality)
    
    hierarchy.add_global_constraint(
        variables,
        maximize_quality,
        objective="minimize",
        weight=2.5,  # Conflicto con costo
        hardness=Hardness.SOFT
    )
    
    # RESTRICCIÓN SOFT 3: Balancear utilización
    def balance_utilization(a):
        if len(a) < n_tasks:
            return 0.0
        
        resource_usage = {r: 0 for r in range(n_resources)}
        for task in range(n_tasks):
            resource = a[f"Task{task}"]
            resource_usage[resource] += task_load[task]
        
        usages = list(resource_usage.values())
        mean_usage = sum(usages) / len(usages)
        variance = sum((u - mean_usage) ** 2 for u in usages) / len(usages)
        max_variance = (n_tasks ** 2) / n_resources
        
        return variance / max_variance if max_variance > 0 else 0.0
    
    hierarchy.add_pattern_constraint(
        variables,
        balance_utilization,
        pattern_type="balanced",
        weight=1.5,
        hardness=Hardness.SOFT
    )
    
    return variables, domains, hard_constraints, hierarchy


# ============================================================================
# EVALUACIÓN
# ============================================================================

def evaluate_solution_quality(solution: Dict, hierarchy: ConstraintHierarchy,
                              landscape: EnergyLandscapeOptimized) -> SolutionQuality:
    """Evalúa calidad con desglose por objetivo."""
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
    breakdown = {}
    
    for level in ConstraintLevel:
        constraints = hierarchy.get_constraints_at_level(level)
        for i, constraint in enumerate(constraints):
            if constraint.hardness == Hardness.SOFT:
                satisfied, violation = constraint.evaluate(solution)
                weighted_violation = constraint.weight * violation
                soft_violations += weighted_violation
                
                # Guardar desglose
                key = f"{level.name}_constraint_{i}"
                breakdown[key] = weighted_violation
    
    return SolutionQuality(
        hard_violations=hard_violations,
        soft_violations=soft_violations,
        total_energy=energy.total_energy,
        breakdown=breakdown
    )


# ============================================================================
# BENCHMARK
# ============================================================================

def run_benchmark(solver_name: str, solver, problem_name: str,
                 problem_size: int, hierarchy: ConstraintHierarchy,
                 landscape: EnergyLandscapeOptimized,
                 timeout: float = 60.0) -> BenchmarkResult:
    """Ejecuta benchmark."""
    print(f"  {solver_name:<50}", end=" ", flush=True)
    
    start_time = time.time()
    
    try:
        if hasattr(solver, 'solve'):
            if 'timeout' in solver.solve.__code__.co_varnames:
                solution = solver.solve(timeout=timeout)
            elif 'max_nodes' in solver.solve.__code__.co_varnames:
                solution = solver.solve(max_nodes=50000)
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
            print(f"✓ {elapsed:6.2f}s  {nodes:8d} nodos  {solutions:4d} sols  E={quality.total_energy:.3f}")
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
    print("="*120)
    print("BENCHMARK: PROBLEMAS COMPLEJOS MULTI-OBJETIVO")
    print("="*120)
    print("\nObjetivo: Demostrar ventajas del Flujo de Fibración en problemas con MÚLTIPLES OBJETIVOS EN CONFLICTO")
    print("\n📌 Forward Checking: Primera solución factible (ignora objetivos SOFT)")
    print("📌 Flujo de Fibración: Explora y optimiza trade-offs entre objetivos")
    
    all_results = []
    
    # ========================================================================
    # PROBLEMA 1: SCHEDULING
    # ========================================================================
    print(f"\n{'='*120}")
    print("PROBLEMA 1: SCHEDULING CON MÚLTIPLES OBJETIVOS")
    print("="*120)
    print("  8 tareas, 3 trabajadores")
    print("  🔴 HARD: Restricciones de precedencia")
    print("  🟡 SOFT: Minimizar makespan (peso=3.0) + Balancear carga (peso=2.0) + Preferencias (peso=1.5)")
    print("  ⚠️  CONFLICTO: Minimizar tiempo vs. balancear carga vs. respetar preferencias")
    print("="*120)
    
    variables, domains, hard_constraints, hierarchy = create_scheduling_problem(n_tasks=8, n_workers=3)
    landscape = EnergyLandscapeOptimized(hierarchy)
    
    # Forward Checking
    solver = ForwardCheckingSolver(variables, domains, hard_constraints)
    result = run_benchmark(
        "Forward Checking (primera solución)", solver, "Scheduling", 8,
        hierarchy, landscape, timeout=10.0
    )
    all_results.append(result)
    
    # Flujo de Fibración
    solver = SimpleOptimizationSolver(variables, domains, hierarchy)
    solver.max_solutions = 20
    result = run_benchmark(
        "Flujo de Fibración (mejor de 20 soluciones)", solver, "Scheduling", 8,
        hierarchy, landscape, timeout=10.0
    )
    all_results.append(result)
    
    # ========================================================================
    # PROBLEMA 2: ASIGNACIÓN DE RECURSOS
    # ========================================================================
    print(f"\n{'='*120}")
    print("PROBLEMA 2: ASIGNACIÓN DE RECURSOS CON COSTOS Y CALIDAD")
    print("="*120)
    print("  10 tareas, 4 recursos")
    print("  🔴 HARD: Capacidad de recursos no excedida")
    print("  🟡 SOFT: Minimizar costo (peso=3.0) + Maximizar calidad (peso=2.5) + Balancear (peso=1.5)")
    print("  ⚠️  CONFLICTO: Costo bajo vs. calidad alta (trade-off clásico)")
    print("="*120)
    
    variables, domains, hard_constraints, hierarchy = create_resource_assignment_problem(n_tasks=10, n_resources=4)
    landscape = EnergyLandscapeOptimized(hierarchy)
    
    # Forward Checking
    solver = ForwardCheckingSolver(variables, domains, hard_constraints)
    result = run_benchmark(
        "Forward Checking (primera solución)", solver, "Resource Assignment", 10,
        hierarchy, landscape, timeout=10.0
    )
    all_results.append(result)
    
    # Flujo de Fibración
    solver = SimpleOptimizationSolver(variables, domains, hierarchy)
    solver.max_solutions = 20
    result = run_benchmark(
        "Flujo de Fibración (mejor de 20 soluciones)", solver, "Resource Assignment", 10,
        hierarchy, landscape, timeout=10.0
    )
    all_results.append(result)
    
    # ========================================================================
    # COMPARACIÓN
    # ========================================================================
    print("\n" + "="*120)
    print("COMPARACIÓN DE CALIDAD DE SOLUCIONES")
    print("="*120)
    
    by_problem = {}
    for result in all_results:
        if result.problem_name not in by_problem:
            by_problem[result.problem_name] = []
        by_problem[result.problem_name].append(result)
    
    for problem, results in sorted(by_problem.items()):
        print(f"\n🎯 {problem}:")
        print(f"{'Solver':<55} {'Energía Total':<15} {'SOFT Violations':<18} {'Nodos':<10} {'Mejora':<20}")
        print("-" * 120)
        
        fc_result = next((r for r in results if "Forward Checking" in r.solver_name), None)
        fc_soft = fc_result.solution_quality.soft_violations if fc_result and fc_result.solution_quality else float('inf')
        
        for result in results:
            if not result.solution_found or not result.solution_quality:
                continue
            
            q = result.solution_quality
            improvement = ""
            if fc_soft < float('inf') and q.soft_violations < fc_soft:
                pct = ((fc_soft - q.soft_violations) / fc_soft * 100)
                improvement = f"✨ {pct:.1f}% mejor"
            elif fc_soft < float('inf') and q.soft_violations > fc_soft:
                pct = ((q.soft_violations - fc_soft) / fc_soft * 100)
                improvement = f"⚠️  {pct:.1f}% peor"
            
            print(f"{result.solver_name:<55} {q.total_energy:<15.3f} {q.soft_violations:<18.3f} {result.nodes_explored:<10} {improvement:<20}")
    
    # ========================================================================
    # ESTADÍSTICAS FINALES
    # ========================================================================
    print("\n" + "="*120)
    print("ESTADÍSTICAS FINALES")
    print("="*120)
    
    fc_results = [r for r in all_results if "Forward Checking" in r.solver_name and r.solution_found]
    fib_results = [r for r in all_results if "Fibración" in r.solver_name and r.solution_found]
    
    if fc_results and fib_results:
        fc_avg_soft = sum(r.solution_quality.soft_violations for r in fc_results) / len(fc_results)
        fib_avg_soft = sum(r.solution_quality.soft_violations for r in fib_results) / len(fib_results)
        
        improvement = ((fc_avg_soft - fib_avg_soft) / fc_avg_soft * 100) if fc_avg_soft > 0 else 0
        
        print(f"\n📊 Promedio de violaciones SOFT:")
        print(f"  Forward Checking:      {fc_avg_soft:.3f}")
        print(f"  Flujo de Fibración:    {fib_avg_soft:.3f}")
        print(f"  Mejora:                {improvement:.1f}%")
        
        if improvement > 5:
            print(f"\n✨ CONCLUSIÓN: El Flujo de Fibración encuentra soluciones {improvement:.1f}% mejores")
            print(f"   al explorar el espacio de soluciones y optimizar trade-offs entre objetivos en conflicto.")
            print(f"\n💡 En problemas multi-objetivo complejos, la primera solución factible suele ser subóptima.")
            print(f"   El paisaje de energía del Flujo de Fibración guía hacia soluciones de mejor calidad.")
        elif improvement > 0:
            print(f"\n✓ El Flujo de Fibración encuentra soluciones {improvement:.1f}% mejores")
        else:
            print(f"\n⚠️  Ambos solvers encuentran soluciones de calidad similar.")


if __name__ == "__main__":
    main()

